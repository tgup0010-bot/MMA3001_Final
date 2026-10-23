"""Compare environmental sensor readings against real BoM outdoor weather.

Keenan Granland's third EdStem suggestion ("irming scope #81"): pull in an
external dataset (he named BoM) to compare against the environmental
sensor readings. This only works for whichever sensor(s) actually overlap
BoM's fetchable date range (see occupancy.external_weather for why that
range is limited to a rolling ~15 months) -- this script checks every
sensor and reports which ones qualify, rather than assuming.

Usage:
    python tools/fetch_bom_weather.py          # once, to get the data
    python scripts/weather_comparison_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

from occupancy.env_sensors import sensor_ids
from occupancy.external_weather import load_bom_weather
from occupancy.processed_data import bom_dir_raw_or_processed, env_readings_raw_or_processed
from occupancy.sensors import sensor_label
from occupancy.weather_comparison import compare_to_bom, daily_indoor_series

ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
BOM_DIR = Path("data/raw/bom")
RESULTS_PATH = Path("reports/weather_comparison_results.json")

MIN_OVERLAP_DAYS = 14
COMPARISONS = [
    ("Temperature", "temp_9am_c"),
    ("Humidity", "humidity_9am_pct"),
]


def main() -> None:
    env_long = env_readings_raw_or_processed(ENV_RAW_PATH)
    bom_df = load_bom_weather(bom_dir_raw_or_processed(BOM_DIR))
    print(
        f"BoM data: {len(bom_df)} days, "
        f"{bom_df['date'].min()} to {bom_df['date'].max()}"
    )

    all_results: dict = {}
    for sensorid in sensor_ids(env_long):
        label = sensor_label(sensorid)
        all_results[sensorid] = {}
        for variable, outdoor_col in COMPARISONS:
            try:
                indoor_daily = daily_indoor_series(env_long, sensorid, variable)
            except Exception:
                continue
            result = compare_to_bom(
                indoor_daily, bom_df, outdoor_col, min_overlap_days=MIN_OVERLAP_DAYS
            )
            all_results[sensorid][variable] = result
            status = (
                "no usable overlap"
                if result["n_days"] < MIN_OVERLAP_DAYS
                else f"r={result['correlation']:+.3f}"
            )
            print(
                f"  {label:30} {variable:12} n_days={result['n_days']:4}  {status}"
            )

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "bom_station": "Moorabbin Airport (086077)",
                "bom_date_range": [str(bom_df["date"].min()), str(bom_df["date"].max())],
                "min_overlap_days": MIN_OVERLAP_DAYS,
                "results": all_results,
                "sensor_labels": {s: sensor_label(s) for s in sensor_ids(env_long)},
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
