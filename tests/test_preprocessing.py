"""Unit tests for occupancy.preprocessing.

Uses small, hand-built event logs (not the full raw CSV) so the suite runs
in well under a second and each case tests one specific behaviour.
"""

import pandas as pd
import pytest

from occupancy.preprocessing import coverage_summary, resample_occupancy


def _event(room: str, ts: str, status: str) -> dict:
    return {
        "floorspaceid": room,
        "collecteddate": pd.Timestamp(ts, tz="Australia/Melbourne"),
        "occupancystatus": status,
    }


def test_binary_status_mapping():
    """CurrentlyOccupied -> 1, NotOccupied -> 0, by default."""
    df = pd.DataFrame(
        [
            _event("room-a", "2024-06-01 09:00", "CurrentlyOccupied"),
            _event("room-a", "2024-06-01 09:15", "NotOccupied"),
        ]
    )
    out = resample_occupancy(df, bin_size="15min")
    values = out.sort_values("collecteddate")["occupied"].tolist()
    assert values == [1.0, 0.0]


def test_forward_fill_within_max_gap():
    """A state should hold across bins until the next event, within max_gap."""
    df = pd.DataFrame(
        [
            _event("room-a", "2024-06-01 09:00", "CurrentlyOccupied"),
            _event("room-a", "2024-06-01 10:00", "NotOccupied"),
        ]
    )
    out = resample_occupancy(df, bin_size="15min", max_gap="2h")
    out = out.sort_values("collecteddate")
    # Bins at 09:00, 09:15, 09:30, 09:45 should all be forward-filled as
    # occupied (1.0); the gap to the next real event is only 1h, well
    # inside the 2h max_gap.
    assert (out.loc[out["collecteddate"] < pd.Timestamp("2024-06-01 10:00", tz="Australia/Melbourne"), "occupied"] == 1.0).all()


def test_gap_beyond_max_gap_becomes_missing():
    """A gap longer than max_gap must be NaN, not silently forward-filled."""
    df = pd.DataFrame(
        [
            _event("room-a", "2024-06-01 09:00", "CurrentlyOccupied"),
            # Next event is 5 hours later -- beyond a 2h max_gap.
            _event("room-a", "2024-06-01 14:00", "NotOccupied"),
        ]
    )
    out = resample_occupancy(df, bin_size="15min", max_gap="2h")
    out = out.sort_values("collecteddate")
    midpoint = pd.Timestamp("2024-06-01 12:00", tz="Australia/Melbourne")
    row = out[out["collecteddate"] == midpoint]
    assert row["occupied"].isna().all()


def test_rooms_are_independent():
    """Two rooms must not leak state into each other's grid."""
    df = pd.DataFrame(
        [
            _event("room-a", "2024-06-01 09:00", "CurrentlyOccupied"),
            _event("room-b", "2024-06-01 09:00", "NotOccupied"),
        ]
    )
    out = resample_occupancy(df, bin_size="15min")
    a = out[out["floorspaceid"] == "room-a"]["occupied"]
    b = out[out["floorspaceid"] == "room-b"]["occupied"]
    assert (a == 1.0).all()
    assert (b == 0.0).all()


def test_coverage_summary_reports_missing_fraction():
    df = pd.DataFrame(
        [
            _event("room-a", "2024-06-01 09:00", "CurrentlyOccupied"),
            _event("room-a", "2024-06-01 14:00", "NotOccupied"),  # 5h gap
        ]
    )
    resampled = resample_occupancy(df, bin_size="15min", max_gap="2h")
    summary = coverage_summary(resampled)
    assert summary.loc["room-a", "n_missing"] > 0
    assert 0 < summary.loc["room-a", "pct_missing"] < 100


def test_unknown_occupied_statuses_raises_via_data_loading():
    """occupancy.data_loading._validate should reject unknown statuses."""
    from occupancy.data_loading import _validate

    df = pd.DataFrame(
        [
            {
                "deviceid": "d1",
                "floorspaceid": "room-a",
                "occupancystatus": "SomeNewStatus",
                "headcount": 1,
                "collecteddate": pd.Timestamp("2024-06-01 09:00"),
            }
        ]
    )
    with pytest.raises(ValueError):
        _validate(df)
