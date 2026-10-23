"""Compare environmental sensor readings against real BoM outdoor weather.

Keenan Granland's third EdStem suggestion (thread #81, "Confirming scope"): pull in an
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from occupancy.env_sensors import sensor_ids
from occupancy.external_weather import load_bom_weather
from occupancy.processed_data import bom_dir_raw_or_processed, env_readings_raw_or_processed
from occupancy.sensors import sensor_label
from occupancy.weather_comparison import compare_to_bom, daily_indoor_series

ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
BOM_DIR = Path("data/raw/bom")
RESULTS_PATH = Path("reports/weather_comparison_results.json")
CHART_PATH = Path("doc/report/chart18_bom_validation.png")
CHART_SENSOR = "6012002000326"

MIN_OVERLAP_DAYS = 14
COMPARISONS = [
    ("Temperature", "temp_9am_c"),
    ("Humidity", "humidity_9am_pct"),
]


def main() -> None:
    """Compare every sensor with BoM weather, save the results and draw chart 18."""
    env_long = env_readings_raw_or_processed(ENV_RAW_PATH)
    bom_df = load_bom_weather(bom_dir_raw_or_processed(BOM_DIR))
    print(
        f"BoM data: {len(bom_df)} days, "
        f"{bom_df['date'].min()} to {bom_df['date'].max()}"
    )

    all_results: dict = {}
    # Every sensor is checked, not just 0326: sensors with no genuine readings
    # in BoM's date range are reported as having no usable overlap.
    for sensorid in sensor_ids(env_long):
        label = sensor_label(sensorid)
        all_results[sensorid] = {}
        for variable, outdoor_col in COMPARISONS:
            try:
                indoor_daily = daily_indoor_series(env_long, sensorid, variable)
            except Exception:
                # This sensor never reported this variable: nothing to compare.
                continue
            result = compare_to_bom(
                indoor_daily, bom_df, outdoor_col, min_overlap_days=MIN_OVERLAP_DAYS
            )
            all_results[sensorid][variable] = result
            status = (
                "no usable overlap"
                if result["n_days"] < MIN_OVERLAP_DAYS
                else f"r={result['correlation']:+.3f}, day-to-day change r={result['change_correlation']:+.3f}"
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

    plot_comparison(env_long, bom_df, all_results[CHART_SENSOR])


def plot_comparison(env_long: pd.DataFrame, bom_df: pd.DataFrame, results: dict) -> None:
    """Plot the main sensor's daily mean against BoM's 9 am value, one panel per variable."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": "#898781",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )
    outdoor_all = bom_df.set_index("date")
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    units = {"Temperature": "\u00b0C", "Humidity": "%"}
    for ax, (variable, outdoor_col) in zip(axes, COMPARISONS):
        indoor = daily_indoor_series(env_long, CHART_SENSOR, variable)
        both = pd.concat(
            [indoor.rename("indoor"), outdoor_all[outdoor_col].rename("outdoor")], axis=1
        ).dropna()
        # Reindex to every day so the lines break across days missing from either series.
        both.index = pd.to_datetime(both.index)
        both = both.asfreq("D")
        ax.plot(both.index, both["outdoor"], color="#eb6834", linewidth=1.2,
                label="Outdoor: BoM Moorabbin Airport, 9 am")
        ax.plot(both.index, both["indoor"], color="#0b0b0b", linewidth=2.0,
                label=f"Indoor: sensor {CHART_SENSOR[-4:]}, daily mean")
        r = results[variable]
        ax.set_title(
            f"{variable}: r = {r['correlation']:.2f} (day-to-day changes {r['change_correlation']:.2f}),"
            f" {r['n_days']} days"
        )
        ax.set_ylabel(f"{variable} ({units[variable]})")
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#c3c2b7")
        ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved -> {CHART_PATH}")


if __name__ == "__main__":
    main()
