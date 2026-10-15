"""Unit tests for occupancy.sensor_intervals.

Uses small hand-built series and long-format tables (not the real large
files) so the suite runs quickly and isolates the interval logic.
"""

import numpy as np
import pandas as pd

from occupancy.sensor_intervals import find_unbroken_intervals, interval_audit, usable_rows

TZ = "Australia/Melbourne"


def _series(values):
    idx = pd.date_range("2024-06-01 09:00", periods=len(values), freq="15min", tz=TZ)
    return pd.Series(values, index=idx, dtype=float)


def test_find_unbroken_intervals_splits_at_gaps():
    s = _series([1, 2, np.nan, 3, 4, 5, np.nan, np.nan, 6])
    runs = find_unbroken_intervals(s)
    assert runs["n_bins"].tolist() == [2, 3, 1]
    assert runs["start"].iloc[1] == s.index[3]
    assert runs["end"].iloc[1] == s.index[5]


def test_find_unbroken_intervals_empty_series():
    assert find_unbroken_intervals(_series([np.nan, np.nan])).empty


def test_usable_rows_matches_rolling_window_plus_target():
    runs = pd.DataFrame({"n_bins": [3, 4, 5, 10]})
    # A 4-bin window plus a target bin needs 5 bins; n bins yield n - 4 rows.
    assert usable_rows(runs, rolling_window=4) == 0 + 0 + 1 + 6


def _long(sensorid, times, values):
    return pd.DataFrame(
        {
            "sensorid": sensorid,
            "time": pd.to_datetime(times).tz_localize(TZ),
            "variable": "Carbon dioxide",
            "unit": "ppm",
            "value": values,
        }
    )


def test_interval_audit_replayed_packet_yields_no_intervals():
    start = pd.Timestamp("2024-06-01 09:00")
    live_times = [start + pd.Timedelta(minutes=15 * i) for i in range(8)]
    live = _long("live", live_times, [450 + i for i in range(8)])
    # A dead sensor re-sends one packet: same timestamp and value every row.
    dead = _long("dead", [start] * 50, [666] * 50)
    audit = interval_audit(pd.concat([live, dead]), ["live", "dead"]).set_index("sensorid")

    assert audit.loc["live", "n_intervals"] == 1
    assert audit.loc["live", "usable_rows"] == 4
    assert audit.loc["live", "hours"] == 2.0

    assert audit.loc["dead", "raw_rows"] == 50
    assert audit.loc["dead", "distinct_timestamps"] == 1
    assert audit.loc["dead", "distinct_values"] == 1
    assert audit.loc["dead", "usable_rows"] == 0
