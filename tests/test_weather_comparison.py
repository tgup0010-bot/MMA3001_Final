"""Unit tests for occupancy.weather_comparison.

Uses small hand-built series/tables (not real data) so the suite runs in
well under a second and isolates the aggregation and correlation logic
from network access and file parsing.
"""

import json

import numpy as np
import pandas as pd
import pytest

from occupancy.env_sensors import _flatten
from occupancy.weather_comparison import compare_to_bom, daily_indoor_series


def _env_row(sensorid: str, t: pd.Timestamp, value: float) -> dict:
    return {
        "sensorid": sensorid,
        "jsondata": json.dumps(
            [
                {
                    "values": [{"time": int(t.timestamp()), "value": value}],
                    "variable": {"name": "Temperature", "unit": "\xb0C"},
                }
            ]
        ),
    }


def test_daily_indoor_series_averages_within_each_day():
    times = pd.date_range("2024-01-01 00:00", periods=4, freq="12h", tz="Australia/Melbourne")
    # Two readings on day 1 (20, 22 -> mean 21), two on day 2 (24, 26 -> mean 25).
    values = [20.0, 22.0, 24.0, 26.0]
    raw = pd.DataFrame([_env_row("s1", t, v) for t, v in zip(times, values)])
    env_long = _flatten(raw)

    daily = daily_indoor_series(env_long, "s1", "Temperature")
    assert daily.loc[pd.Timestamp("2024-01-01").date()] == pytest.approx(21.0)
    assert daily.loc[pd.Timestamp("2024-01-02").date()] == pytest.approx(25.0)


def test_compare_to_bom_detects_a_real_relationship():
    dates = pd.date_range("2025-07-01", periods=20, freq="D").date
    # Indoor temp tracks outdoor temp loosely (climate-controlled but not
    # fully independent), plus a constant offset -- a realistic pattern.
    outdoor = [10 + (i % 5) for i in range(20)]
    indoor = [18 + 0.5 * o for o in outdoor]

    indoor_daily = pd.Series(indoor, index=dates)
    bom_df = pd.DataFrame({"date": dates, "temp_9am_c": outdoor})

    result = compare_to_bom(indoor_daily, bom_df, "temp_9am_c")
    assert result["correlation"] > 0.9
    assert result["n_days"] == 20
    assert result["date_range"] == (dates[0], dates[-1])


def test_compare_to_bom_returns_nan_with_insufficient_overlap():
    dates = pd.date_range("2025-07-01", periods=3, freq="D").date
    indoor_daily = pd.Series([20.0, 21.0, 22.0], index=dates)
    bom_df = pd.DataFrame({"date": dates, "temp_9am_c": [10.0, 11.0, 12.0]})

    result = compare_to_bom(indoor_daily, bom_df, "temp_9am_c", min_overlap_days=14)
    assert pd.isna(result["correlation"])
    assert result["date_range"] is None


def test_compare_to_bom_only_uses_overlapping_dates():
    indoor_dates = pd.date_range("2025-07-01", periods=20, freq="D").date
    bom_dates = pd.date_range("2025-07-11", periods=20, freq="D").date  # 10-day overlap

    indoor_daily = pd.Series([20.0] * 20, index=indoor_dates)
    bom_df = pd.DataFrame({"date": bom_dates, "temp_9am_c": [10.0] * 20})

    result = compare_to_bom(indoor_daily, bom_df, "temp_9am_c", min_overlap_days=5)
    assert result["n_days"] == 10
    assert np.isnan(result["correlation"])  # constant series: r undefined, not an error
