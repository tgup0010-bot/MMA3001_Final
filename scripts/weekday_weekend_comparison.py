"""Compare weekday and weekend CO2 for the two sensors with genuine readings.

Checks the assumption behind the ``is_weekend`` feature -- that weekends
look different from weekdays -- against the data, for sensor 0326 (main
analysis) and sensor 0869 (room G.38). Uses each sensor's genuine-reading
period only (see the report, Section 4.1), on the same 15-minute grid as
the models, and also reports each sensor's weekday hour-of-day profile.

Writes ``reports/weekday_weekend_results.json``.

Usage:
    python scripts/weekday_weekend_comparison.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from occupancy.co2_regression import co2_series
from occupancy.processed_data import env_readings_raw_or_processed

ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
RESULTS_PATH = Path("reports/weekday_weekend_results.json")
BIN_SIZE = "15min"
#: Genuine-reading period of each sensor (local time), from the interval audit.
SENSORS = {
    "6012002000326": ("2025-09-07", "2026-04-23"),
    "6012002000869": ("2024-02-29", "2024-06-14"),
}
WORKING_HOURS = (9, 17)


def main() -> None:
    env_long = env_readings_raw_or_processed(ENV_RAW_PATH)
    results = {}
    for sensorid, (start, end) in SENSORS.items():
        co2 = co2_series(env_long, sensorid, BIN_SIZE).dropna()
        co2 = co2[(co2.index >= pd.Timestamp(start, tz=co2.index.tz)) & (co2.index < pd.Timestamp(end, tz=co2.index.tz))]
        weekend = co2.index.dayofweek >= 5
        working = (co2.index.hour >= WORKING_HOURS[0]) & (co2.index.hour < WORKING_HOURS[1])
        weekday_profile = co2[~weekend].groupby(co2[~weekend].index.hour).mean()
        results[sensorid] = {
            "period": [start, end],
            "mean_weekday": float(co2[~weekend].mean()),
            "mean_weekend": float(co2[weekend].mean()),
            "working_hours_weekday": float(co2[~weekend & working].mean()),
            "working_hours_weekend": float(co2[weekend & working].mean()),
            "weekday_peak_hour": int(weekday_profile.idxmax()),
            "weekday_lowest_hour": int(weekday_profile.idxmin()),
            "weekday_hourly_mean": {int(h): float(v) for h, v in weekday_profile.items()},
        }
        r = results[sensorid]
        print(
            f"{sensorid}: all hours weekday {r['mean_weekday']:.0f} / weekend {r['mean_weekend']:.0f} ppm; "
            f"9am-5pm weekday {r['working_hours_weekday']:.0f} / weekend {r['working_hours_weekend']:.0f} ppm; "
            f"weekday peak at {r['weekday_peak_hour']}:00, lowest at {r['weekday_lowest_hour']}:00"
        )
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
