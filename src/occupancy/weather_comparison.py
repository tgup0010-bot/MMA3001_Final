"""Compare one environmental sensor's readings against real outdoor weather.

Sanity-checks an indoor sensor against BoM's real Moorabbin Airport
observations for the same days: does the sensor's temperature/humidity
plausibly track outdoor conditions (as any real, working indoor sensor
subject to some outdoor influence should), or is it disconnected from
reality (suggesting a faulty sensor)? This is Keenan Granland's third
EdStem suggestion ("irming scope #81") -- pulling in an external dataset
for comparison -- applied to whichever sensor actually has overlapping
dates with BoM's fetchable window (see
:mod:`occupancy.external_weather` for why that window is limited, and the
project report for which sensor qualifies).

This is a plausibility check, not a validation of accuracy: a real indoor
sensor is expected to correlate with outdoor weather only loosely (a
climate-controlled building deliberately dampens the outdoor swing), so a
moderate positive correlation is the expected, healthy result -- not
evidence of anything being "wrong".
"""

from __future__ import annotations

import pandas as pd

from occupancy.env_sensors import pivot_variable


def daily_indoor_series(env_long: pd.DataFrame, sensorid: str, variable: str) -> pd.Series:
    """Aggregate one sensor's readings of one variable to a daily mean.

    Args:
        env_long: Output of :func:`occupancy.env_sensors.load_env_sensor_log`.
        sensorid: Which sensor to extract.
        variable: Which variable, e.g. ``"Temperature"`` or ``"Humidity"``.

    Returns:
        A ``pandas.Series`` of daily mean values, indexed by (naive)
        ``datetime.date``.
    """
    pivoted = pivot_variable(env_long, variable)
    sub = pivoted[pivoted["sensorid"] == sensorid]
    # Daily means, because BoM publishes one observation per day; days are
    # local (Melbourne) calendar days to line up with BoM's dates.
    daily = sub.set_index("time")["value"].resample("1D").mean()
    daily.index = daily.index.date
    return daily


def compare_to_bom(
    indoor_daily: pd.Series,
    bom_df: pd.DataFrame,
    outdoor_col: str,
    min_overlap_days: int = 14,
) -> dict:
    """Correlate a daily indoor series against a BoM outdoor column.

    Args:
        indoor_daily: Output of :func:`daily_indoor_series`.
        bom_df: Output of :func:`occupancy.external_weather.load_bom_weather`.
        outdoor_col: Which BoM column to compare against, e.g.
            ``"temp_9am_c"``, ``"humidity_9am_pct"``.
        min_overlap_days: Minimum overlapping days required to report a
            correlation at all (below this, the result is too thin to
            trust -- see :data:`n_days` in the return value).

    Returns:
        A dict with ``correlation`` (Pearson r, NaN if under
        ``min_overlap_days``), ``n_days`` (overlapping days used), and
        ``date_range`` (``(first, last)`` overlapping dates, or ``None``).
    """
    # Only days present in both series count: a correlation over mismatched
    # dates would be meaningless.
    outdoor = bom_df.set_index("date")[outdoor_col]
    aligned = pd.concat(
        [indoor_daily.rename("indoor"), outdoor.rename("outdoor")], axis=1
    ).dropna()

    if len(aligned) < min_overlap_days:
        return {"correlation": float("nan"), "n_days": len(aligned), "date_range": None}

    # A faulty sensor (stuck value, frozen clock, noise) gives r near zero, so
    # a clearly positive r is evidence the sensor measures real conditions.
    corr = aligned["indoor"].corr(aligned["outdoor"])
    return {
        "correlation": corr,
        "n_days": len(aligned),
        "date_range": (min(aligned.index), max(aligned.index)),
    }
