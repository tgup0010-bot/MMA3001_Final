"""Accumulated CO2 exposure via Newton-Cotes integration (MMA3001 Week 6).

Computes total accumulated CO2 exposure (ppm.h) for sensor 6012002000326
two ways:

1. Trapezoidal integration directly on the raw, irregularly-spaced sensor
   readings -- honouring the true gap between each pair of readings
   (Week 6.1/7.1), rather than assuming a uniform sampling rate.
2. Trapezoidal and Simpson's 1/3 rule on regular grids at several bin
   sizes (matching scripts/sensitivity_analysis.py's own 5/15/30/60-minute
   sweep), to see how the accumulated estimate -- and the agreement
   between the two rules -- changes as sampling resolution changes
   (Week 6.2's refinement/convergence-evidence idea).

Usage:
    python scripts/co2_integration_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from occupancy.co2_integration import (
    richardson_order_check,
    simpsons_integral,
    trapezoidal_integral,
    trapezoidal_integral_gap_aware,
)
from occupancy.co2_regression import co2_series
from occupancy.processed_data import env_readings_raw_or_processed

ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
RESULTS_PATH = Path("reports/co2_integration_results.json")

SENSOR_ID = "6012002000326"
#: 5-minute bins are deliberately excluded from the convergence comparison:
#: this sensor's real reporting cadence is ~10 minutes, so at 5-minute
#: resolution most bins are empty and no meaningfully long contiguous run
#: exists (checked directly -- the longest run is 2 points). That is
#: itself a real, reportable finding, not something to force past.
BIN_SIZES_MIN = (15, 30, 60)
#: Matches occupancy.preprocessing.resample_occupancy's own default -- a
#: panel spanning longer than this is a sensor outage, not a measurement
#: to integrate across as if CO2 varied linearly through it.
MAX_GAP_HOURS = 2.0


def main() -> None:
    """Compute cumulative CO2 exposure with both rules and save the cross-check."""
    env_long = env_readings_raw_or_processed(ENV_RAW_PATH)

    from occupancy.env_sensors import pivot_variable

    co2_raw = pivot_variable(env_long, "Carbon dioxide")
    sensor_readings = co2_raw[co2_raw["sensorid"] == SENSOR_ID].sort_values("time")
    print(
        f"Raw readings for sensor {SENSOR_ID}: {len(sensor_readings):,}  "
        f"{sensor_readings['time'].min()} to {sensor_readings['time'].max()}"
    )

    # --- 1. Trapezoidal on the true, irregularly-spaced raw readings ---
    naive_total = trapezoidal_integral(sensor_readings["time"], sensor_readings["value"])
    duration_hours = (
        sensor_readings["time"].iloc[-1] - sensor_readings["time"].iloc[0]
    ).total_seconds() / 3600.0
    print(f"\nNaive trapezoidal integral (bridges every gap, including outages): "
          f"{naive_total:,.1f} ppm.h over {duration_hours:,.1f} h span "
          f"-> implied mean {naive_total / duration_hours:.1f} ppm")

    gap_aware = trapezoidal_integral_gap_aware(
        sensor_readings["time"], sensor_readings["value"], max_gap_hours=MAX_GAP_HOURS
    )
    print(
        f"Gap-aware trapezoidal (max_gap={MAX_GAP_HOURS}h, excluding outage panels): "
        f"{gap_aware['total']:,.1f} ppm.h over {gap_aware['hours_included']:,.1f} h "
        f"actually covered ({gap_aware['n_panels_excluded']} outage panel(s) totalling "
        f"{gap_aware['hours_excluded']:,.1f} h excluded)"
    )
    print(
        f"  -> implied mean over covered time: "
        f"{gap_aware['total'] / gap_aware['hours_included']:.1f} ppm "
        f"(vs raw reading mean {sensor_readings['value'].mean():.1f} ppm)"
    )

    # --- 2. Regular-grid trapezoidal + Simpson, refining ONE fixed window ---
    #
    # A naive dropna() on the resampled series looks like a clean regular
    # grid but is not one: this sensor has real reporting outages (one gap
    # alone spans ~636 days), and dropna() silently concatenates readings
    # from either side of a gap as if they were one bin apart. That is
    # exactly the "pretend the spacing is uniform" mistake Week 6/7 warn
    # against -- caught here by Simpson and trapezoidal disagreeing wildly
    # on the naive version (Simpson is far more sensitive to a violated
    # uniform-spacing assumption than trapezoidal is).
    #
    # The fix has two parts. First, find one genuinely gap-free window at
    # the finest (15-min) resolution -- coarser bin sizes are then derived
    # from *that same window*, not from independently-chosen "longest run"
    # windows (which would each cover a different stretch of calendar
    # time, making a refinement comparison meaningless: halving h only
    # means something if both estimates approximate the same interval).
    finest_series = co2_series(env_long, SENSOR_ID, f"{min(BIN_SIZES_MIN)}min")
    run_id = finest_series.notna().ne(finest_series.notna().shift()).cumsum()
    longest_run_id = finest_series[finest_series.notna()].groupby(run_id).size().idxmax()
    window = finest_series[(run_id == longest_run_id) & finest_series.notna()]
    window_start, window_end = window.index.min(), window.index.max()
    print(
        f"\nFixed comparison window (longest gap-free run at "
        f"{min(BIN_SIZES_MIN)}min resolution): {window_start} to {window_end} "
        f"({len(window)} points, {(window_end - window_start).total_seconds()/3600:.1f} h)"
    )

    raw_in_window = sensor_readings[
        (sensor_readings["time"] >= window_start) & (sensor_readings["time"] <= window_end)
    ]

    grid_results = {}
    for bin_min in BIN_SIZES_MIN:
        series = raw_in_window.set_index("time")["value"].resample(f"{bin_min}min").mean()
        values = series.to_numpy()
        if pd.isna(values).any() or len(values) < 3:
            grid_results[bin_min] = {
                "n_points": len(values),
                "skipped": "insufficient data in the fixed window at this resolution",
            }
            print(f"  bin={bin_min:3}min  n={len(values):4}  -- insufficient data, skipped")
            continue
        if len(values) % 2 == 0:
            values = values[:-1]  # Simpson needs an even number of intervals

        trap = trapezoidal_integral(series.index[: len(values)], values)
        simpson = simpsons_integral(values, bin_hours=bin_min / 60.0)
        grid_results[bin_min] = {"n_points": len(values), "trapezoidal": trap, "simpson": simpson}
        print(
            f"  bin={bin_min:3}min  n={len(values):4}  "
            f"trap={trap:10,.1f} ppm.h  simpson={simpson:10,.1f} ppm.h  "
            f"diff={abs(trap - simpson):6.2f} ({100*abs(trap-simpson)/trap:.4f}%)"
        )

    # --- 3. Convergence check: does refining the grid change the trapezoidal estimate as expected? ---
    print("\nConvergence (trapezoidal estimate vs bin size, finer first):")
    sorted_bins = sorted(BIN_SIZES_MIN)
    convergence = []
    for coarse, fine in zip(sorted_bins[1:], sorted_bins[:-1]):
        if "trapezoidal" not in grid_results[coarse] or "trapezoidal" not in grid_results[fine]:
            print(f"  {fine}min vs {coarse}min: skipped (one side had insufficient contiguous data)")
            continue
        check = richardson_order_check(
            estimate_h=grid_results[coarse]["trapezoidal"],
            estimate_h_half=grid_results[fine]["trapezoidal"],
            order=2,
        )
        convergence.append({"coarse_min": coarse, "fine_min": fine, **check})
        print(
            f"  {fine}min vs {coarse}min: change={check['change']:+.1f} ppm.h "
            f"({check['pct_change']:+.2f}%)"
        )

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "sensor_id": SENSOR_ID,
                "raw_reading_count": len(sensor_readings),
                "duration_hours": duration_hours,
                "naive_trapezoidal_ppmh": naive_total,
                "gap_aware_trapezoidal": {"max_gap_hours": MAX_GAP_HOURS, **gap_aware},
                "grid_results": grid_results,
                "convergence": convergence,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
