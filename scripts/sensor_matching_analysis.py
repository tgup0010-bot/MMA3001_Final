"""Attempt to link environmental sensors to occupancy zones by data, not metadata.

The supplied location spreadsheet cannot map most occupancy zones to the
environmental sensors (see the report, Section 6). This script
tests the alternative Keenan Granland suggested on the EdStem forum
(thread #81, "Confirming scope"): rather than assume a room correspondence, look for
one in the data itself, by correlating each zone's occupancy pattern
against each sensor's readings.

See occupancy.sensor_matching for the method (lagged correlation of the
CO2 *rate of change*, not its raw level -- see that module's docstring for
why the raw level would be misleading).

Usage:
    python scripts/sensor_matching_analysis.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from occupancy.processed_data import (
    env_readings_raw_or_processed,
    resampled_occupancy_raw_or_processed,
)
from occupancy.rooms import room_label
from occupancy.sensor_matching import build_correlation_matrix, best_match_per_room
from occupancy.sensors import sensor_label

OCC_RAW_PATH = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
RESULTS_PATH = Path("reports/sensor_matching_results.json")

BIN_SIZE = "15min"
VARIABLES_TESTED = ("Carbon dioxide", "Temperature", "Humidity")


def timed(label: str, fn, *args, **kwargs):
    """Call ``fn`` and print how long it took; returns ``(result, seconds)``."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[{elapsed:6.1f}s] {label}")
    return result, elapsed


def main() -> None:
    """Correlate every zone with every sensor for each variable and save the matrices."""
    timings: dict[str, float] = {}

    env_long, t = timed("load env readings", env_readings_raw_or_processed, ENV_RAW_PATH)
    timings["load_env"] = t
    print(f"  env rows={len(env_long):,}  sensors={env_long['sensorid'].nunique()}")

    occ_resampled, t = timed(
        "load + resample occupancy", resampled_occupancy_raw_or_processed, OCC_RAW_PATH, BIN_SIZE
    )
    timings["load_and_resample_occupancy"] = t
    rooms = sorted(occ_resampled["floorspaceid"].unique())
    print(f"  zones={len(rooms)}: {[room_label(r) for r in rooms]}")

    all_results = {}
    # CO2 is the main test; temperature and humidity are checked too, because
    # a shared HVAC schedule could link a sensor to a zone without shared people.
    for variable in VARIABLES_TESTED:
        print(f"\n=== Testing variable: {variable} ===")
        (matrix, detail), t = timed(
            f"build_correlation_matrix ({variable})",
            build_correlation_matrix,
            occ_resampled,
            env_long,
            variable=variable,
            bin_size=BIN_SIZE,
        )
        timings[f"correlate_{variable}"] = t

        # Human-readable labels for printing only; the JSON keeps the raw ids.
        display = matrix.rename(index=room_label, columns=sensor_label)
        print(display.to_string(float_format=lambda x: f"{x:+.3f}"))

        # For each zone, the sensor (and lag) with the strongest correlation.
        # Strongest does not mean strong: |r| must be judged as well.
        summary = best_match_per_room(matrix, detail)
        summary_display = summary.copy()
        summary_display.index = [room_label(r) for r in summary_display.index]
        summary_display["best_sensor"] = summary_display["best_sensor"].map(
            lambda s: sensor_label(s) if s is not None else None
        )
        print("\nBest-correlated sensor per zone:")
        print(summary_display.to_string(float_format=lambda x: f"{x:+.3f}"))

        all_results[variable] = {
            "matrix": matrix.to_dict(),
            "best_match_per_room": summary.reset_index().to_dict(orient="records"),
        }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "bin_size": BIN_SIZE,
                "variables_tested": list(VARIABLES_TESTED),
                "zones": {r: room_label(r) for r in rooms},
                "sensors": {s: sensor_label(s) for s in env_long["sensorid"].unique()},
                "timings_seconds": timings,
                "results": all_results,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
