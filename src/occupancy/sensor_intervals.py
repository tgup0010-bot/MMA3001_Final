"""Unbroken-interval audit of the environmental sensors.

Rather than accepting or rejecting a sensor as a whole, each sensor's
binned CO2 series is cut into *unbroken intervals* -- runs of consecutive
non-empty bins with no gap. Any interval long enough to build one
supervised row (the 1-hour rolling window plus the 15-minute-ahead target,
i.e. ``rolling_window + 1`` bins) contributes training data, whichever
sensor it comes from.

This is also what exposes the dataset's dominant failure mode. A sensor
that stops producing new readings keeps re-transmitting its last packet
(same internal timestamp, same value) roughly every 11 minutes; on a
regular time grid those thousands of rows all collapse into a single bin,
so a sensor that only ever replays one packet yields no intervals at all.
"""

from __future__ import annotations

import pandas as pd

from occupancy.co2_regression import co2_series


def find_unbroken_intervals(series: pd.Series) -> pd.DataFrame:
    """Find runs of consecutive non-missing values on a regular time grid.

    Args:
        series: A regularly binned series (e.g. the output of
            :func:`occupancy.co2_regression.co2_series`), where missing
            bins are ``NaN``.

    Returns:
        A DataFrame with one row per unbroken interval and columns
        ``start``, ``end`` (index labels of the first and last bin) and
        ``n_bins``, in time order. Empty if ``series`` has no values.
    """
    present = series.notna()
    run_id = (present != present.shift()).cumsum()
    runs = (
        series.index.to_series()[present]
        .groupby(run_id[present])
        .agg(start="min", end="max", n_bins="size")
    )
    return runs.reset_index(drop=True)


def usable_rows(intervals: pd.DataFrame, rolling_window: int = 4) -> int:
    """Number of supervised rows the intervals can produce.

    A row needs ``rolling_window`` bins of history (which includes the
    current bin) plus the next bin as its target, so an interval of
    ``n`` bins yields ``n - rolling_window`` rows (none if shorter).
    """
    return int((intervals["n_bins"] - rolling_window).clip(lower=0).sum())


def interval_audit(
    env_long: pd.DataFrame,
    sensorids: list[str],
    *,
    bin_size: str = "15min",
    rolling_window: int = 4,
) -> pd.DataFrame:
    """Summarise every sensor's usable unbroken intervals.

    Args:
        env_long: Output of :func:`occupancy.env_sensors.load_env_sensor_log`.
        sensorids: Sensors to audit.
        bin_size: Pandas frequency string for the regular grid.
        rolling_window: Bins of history each supervised row needs.

    Returns:
        One row per sensor with ``raw_rows`` (CO2 readings in the log),
        ``distinct_timestamps``, ``distinct_values``, the first/last
        distinct reading time, ``n_intervals`` and ``hours`` (intervals
        long enough to yield at least one row) and ``usable_rows``.
    """
    co2 = env_long[env_long["variable"] == "Carbon dioxide"]
    min_bins = rolling_window + 1
    bin_hours = pd.Timedelta(bin_size) / pd.Timedelta("1h")

    rows = []
    for sid in sensorids:
        readings = co2[co2["sensorid"] == sid]
        intervals = find_unbroken_intervals(co2_series(env_long, sid, bin_size))
        long_enough = intervals[intervals["n_bins"] >= min_bins]
        rows.append(
            {
                "sensorid": sid,
                "raw_rows": len(readings),
                "distinct_timestamps": readings["time"].nunique(),
                "distinct_values": readings["value"].nunique(),
                "first_reading": readings["time"].min(),
                "last_reading": readings["time"].max(),
                "n_intervals": len(long_enough),
                "hours": float(long_enough["n_bins"].sum() * bin_hours),
                "usable_rows": usable_rows(long_enough, rolling_window),
            }
        )
    return pd.DataFrame(rows)
