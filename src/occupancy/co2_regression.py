"""Predict CO2 concentration from its own history and building occupancy.

A genuine regression task (continuous CO2 target, ppm) using the MMA3001
Week 5 methods (Linear Regression, Decision Tree Regression, SVR, Neural
Network Regression) -- not the classification methods used for occupancy
prediction. This directly follows up on Keenan Granland's EdStem
suggestion ("irming scope #81") and the negative result found in
occupancy.sensor_matching: since no single occupancy zone could be
statistically linked to sensor 6012002000326 (the one sensor with real,
trustworthy CO2 data -- see occupancy.weather_comparison for its
validation against real BoM weather), this predicts CO2 from *building-wide*
occupancy (how many of the 5 zones are occupied at once) instead of
guessing a single room, and directly tests whether that occupancy signal
improves CO2 prediction at all, or whether CO2 is better explained by its
own recent trend alone.
"""

from __future__ import annotations

import pandas as pd

from occupancy.env_sensors import pivot_variable

#: Numeric feature columns for the CO2 regression task.
NUMERIC_FEATURES_WITH_OCCUPANCY = (
    "co2_now",
    "co2_lag1",
    "co2_rolling_1h",
    "building_occupancy",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
)

#: Same features, minus the occupancy signal -- used to test whether
#: occupancy adds predictive value over CO2's own autoregressive pattern.
NUMERIC_FEATURES_WITHOUT_OCCUPANCY = tuple(
    f for f in NUMERIC_FEATURES_WITH_OCCUPANCY if f != "building_occupancy"
)


def building_occupancy_series(
    occ_resampled: pd.DataFrame,
    *,
    room_col: str = "floorspaceid",
    time_col: str = "collecteddate",
) -> pd.Series:
    """Aggregate per-room occupancy into one building-wide signal.

    Args:
        occ_resampled: Output of
            :func:`occupancy.preprocessing.resample_occupancy`, covering
            all rooms on the same time grid.
        room_col: Column identifying the room/zone.
        time_col: Column holding the time bin.

    Returns:
        A ``pandas.Series`` indexed by time bin, holding the number of
        zones occupied at that instant (0 to 5). Bins where every room is
        simultaneously missing (all ``NaN``) are left ``NaN`` rather than
        counted as zero occupied, since "no data" and "confirmed empty"
        are different claims.
    """
    # pivot_table silently drops any index value whose row is entirely NaN
    # across every column -- exactly the "all rooms missing" bins this
    # function needs to detect and preserve. Reindex against every
    # timestamp actually present in the input to put those bins back.
    wide = occ_resampled.pivot_table(index=time_col, columns=room_col, values="occupied")
    all_timestamps = pd.Index(sorted(occ_resampled[time_col].unique()), name=time_col)
    wide = wide.reindex(all_timestamps)
    all_missing = wide.isna().all(axis=1)
    total = wide.sum(axis=1, min_count=1)
    total[all_missing] = float("nan")
    return total.rename("building_occupancy")


def co2_series(env_long: pd.DataFrame, sensorid: str, bin_size: str) -> pd.Series:
    """Resample one sensor's CO2 readings onto a regular time grid.

    Args:
        env_long: Output of
            :func:`occupancy.env_sensors.load_env_sensor_log`.
        sensorid: Which sensor to extract.
        bin_size: Pandas frequency string, e.g. ``"15min"``.

    Returns:
        A ``pandas.Series`` of mean CO2 (ppm) per bin, indexed by time.
    """
    pivoted = pivot_variable(env_long, "Carbon dioxide")
    sub = pivoted[pivoted["sensorid"] == sensorid].set_index("time")["value"]
    return sub.resample(bin_size).mean().rename("co2")


def _add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical time-of-day/day-of-week features for a DatetimeIndex.

    Sine/cosine pairs keep 23:45 next to 00:00 and Sunday next to Monday,
    which a plain integer hour or day would place at opposite ends.

    Args:
        index: Tz-aware bin timestamps in local time (Australia/Melbourne),
            so the hour follows the building's daylight-saving clock.

    Returns:
        A DataFrame indexed like ``index`` with ``hour_sin``, ``hour_cos``,
        ``dow_sin``, ``dow_cos`` (all -1 to 1) and ``is_weekend`` (0 or 1).
    """
    hour_frac = index.hour + index.minute / 60.0
    import numpy as np

    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour_frac / 24),
            "hour_cos": np.cos(2 * np.pi * hour_frac / 24),
            "dow_sin": np.sin(2 * np.pi * index.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * index.dayofweek / 7),
            "is_weekend": (index.dayofweek >= 5).astype(float),
        },
        index=index,
    )


def build_co2_supervised_dataset(
    co2: pd.Series,
    occupancy: pd.Series,
    *,
    rolling_window: int = 4,
) -> tuple[pd.DataFrame, dict]:
    """Build a supervised (features + continuous CO2 target) table.

    Args:
        co2: Output of :func:`co2_series`.
        occupancy: Output of :func:`building_occupancy_series`.
        rolling_window: Trailing bins (including the current one) averaged
            into ``co2_rolling_1h``.

    Returns:
        A tuple ``(table, info)``. ``table`` has a ``time`` column, every
        name in :data:`NUMERIC_FEATURES_WITH_OCCUPANCY`, and ``target``
        (the *next* bin's CO2, ppm) -- rows with any missing feature or
        target are dropped. ``info`` reports how many.

    Raises:
        ValueError: If the two series share no overlapping timestamps.
    """
    aligned = pd.concat([co2, occupancy], axis=1).sort_index()
    if aligned.dropna().empty:
        raise ValueError("co2 and occupancy series have no overlapping, non-missing timestamps.")

    aligned["co2_now"] = aligned["co2"]
    aligned["co2_lag1"] = aligned["co2"].shift(1)
    aligned["co2_rolling_1h"] = (
        aligned["co2"].rolling(rolling_window, min_periods=rolling_window).mean()
    )
    aligned["target"] = aligned["co2"].shift(-1)

    calendar = _add_calendar_features(aligned.index)
    table = pd.concat([aligned, calendar], axis=1).reset_index(names="time")

    feature_cols = list(NUMERIC_FEATURES_WITH_OCCUPANCY)
    needed = feature_cols + ["target"]

    n_before = len(table)
    table = table.dropna(subset=needed).reset_index(drop=True)
    n_after = len(table)

    info = {
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped": n_before - n_after,
        "pct_dropped": 100.0 * (n_before - n_after) / n_before if n_before else 0.0,
    }

    return table[["time", *feature_cols, "target"]], info
