"""Unit tests for occupancy.sensor_matching.

Uses small, hand-built time series (not the full raw data) so the suite
runs in well under a second and each case isolates one behaviour: that a
genuinely linked (occupancy, sensor) pair is detected, that an unrelated
pair is not, and that the lag search finds the right offset.
"""

import pandas as pd

from occupancy.sensor_matching import (
    best_match_per_room,
    build_correlation_matrix,
    lagged_correlation,
)


def _occupancy_series(pattern: list[float], start: str = "2024-06-01 09:00") -> pd.Series:
    idx = pd.date_range(start, periods=len(pattern), freq="15min", tz="Australia/Melbourne")
    return pd.Series(pattern, index=idx)


def test_lagged_correlation_detects_a_genuine_co2_response():
    """CO2 rises the bin after occupancy starts -- should be found at lag=1."""
    occupied = _occupancy_series([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    # CO2 level climbs the bin *after* each occupied bin, and falls back
    # down after each vacancy -- a delayed, but real, response.
    co2_level = pd.Series(
        [600, 600, 600, 650, 700, 750, 700, 650, 700, 750],
        index=occupied.index,
    )
    result = lagged_correlation(
        occupied, co2_level, max_lag_bins=3, use_diff=True, min_overlap_bins=5
    )
    assert result["best_lag"] == 1
    assert result["correlation"] > 0.5
    assert result["n_bins"] > 0


def test_lagged_correlation_is_weak_for_unrelated_series():
    occupied = _occupancy_series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    # A flat, noise-only signal that never moves in step with occupancy.
    flat_noise = pd.Series(
        [5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0], index=occupied.index
    )
    result = lagged_correlation(
        occupied, flat_noise, max_lag_bins=3, use_diff=True, min_overlap_bins=5
    )
    assert abs(result["correlation"]) < 0.5


def test_lagged_correlation_returns_nan_with_insufficient_overlap():
    occupied = _occupancy_series([0, 1, 1])
    variable = pd.Series([600, 650, 700], index=occupied.index)
    result = lagged_correlation(occupied, variable, max_lag_bins=1)
    assert result["best_lag"] is None
    assert pd.isna(result["correlation"])
    assert result["n_bins"] == 0


def test_build_correlation_matrix_shape_and_best_match():
    time_col = "collecteddate"
    idx = pd.date_range(
        "2024-06-01 09:00", periods=10, freq="15min", tz="Australia/Melbourne"
    )
    occ_pattern = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    occ_resampled = pd.DataFrame(
        {
            "floorspaceid": ["room-a"] * 10,
            time_col: idx,
            "occupied": occ_pattern,
        }
    )

    # Build a tiny env_long table with two sensors: one that tracks the
    # room's occupancy transitions, one that doesn't.
    import json

    def env_row(sensorid, t, value):
        return {
            "sensorid": sensorid,
            "jsondata": json.dumps(
                [
                    {
                        "values": [{"time": int(t.timestamp()), "value": value}],
                        "variable": {"name": "Carbon dioxide", "unit": "ppm"},
                    }
                ]
            ),
        }

    linked_values = [600, 600, 600, 650, 700, 750, 700, 650, 700, 750]
    unrelated_values = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0]

    from occupancy.env_sensors import _flatten

    raw = pd.DataFrame(
        [env_row("sensor-linked", t, v) for t, v in zip(idx, linked_values)]
        + [env_row("sensor-unrelated", t, v) for t, v in zip(idx, unrelated_values)]
    )
    env_long = _flatten(raw)

    matrix, detail = build_correlation_matrix(
        occ_resampled,
        env_long,
        variable="Carbon dioxide",
        bin_size="15min",
        max_lag_bins=3,
        room_col="floorspaceid",
        time_col=time_col,
        min_overlap_bins=5,
    )

    assert matrix.shape == (1, 2)
    assert set(matrix.columns) == {"sensor-linked", "sensor-unrelated"}

    summary = best_match_per_room(matrix, detail)
    assert summary.loc["room-a", "best_sensor"] == "sensor-linked"
