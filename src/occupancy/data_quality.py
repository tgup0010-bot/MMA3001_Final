"""Outlier screening and a physical check on the CO2 readings.

Two questions are answered here, both on the raw (unbinned) readings of one
sensor:

1. **Are there outliers, and are they errors?** :func:`hampel_flags` marks
   readings far from their local median, using the median absolute
   deviation (MAD) as a robust spread estimate. On this dataset the flagged
   readings are sharp rises followed by a smooth decay over 30-40 minutes:
   genuine CO2 releases being diluted, not sensor glitches (a glitch would
   return to baseline at the next reading).
2. **Do those decays obey the mass balance?** Between releases the
   single-zone balance reduces to :math:`dC/dt = -\\lambda (C - C_b)`, so the
   excess over the background decays exponentially at the ventilation rate
   :math:`\\lambda`. :func:`spike_decay_rates` fits that rate to every
   clean decay event, giving an independent, physical estimate of the
   room's time constant :math:`1/\\lambda`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The scale factor that makes the MAD a consistent estimate of the standard
#: deviation for normally distributed noise.
MAD_TO_SIGMA = 1.4826


def hampel_flags(
    readings: pd.Series,
    *,
    window: int = 7,
    n_sigmas: float = 5.0,
    min_abs_ppm: float = 20.0,
) -> pd.Series:
    """Flag readings far from their local median (Hampel identifier).

    Args:
        readings: One sensor's CO2 readings in ppm, indexed by time, in time
            order, without duplicate timestamps.
        window: Number of readings in the centred rolling window (7 readings
            is about one hour at the sensor's 10-minute interval).
        n_sigmas: How many robust standard deviations from the local median
            a reading must lie to be flagged.
        min_abs_ppm: Minimum absolute departure in ppm. The sensor reports
            whole ppm and is very steady overnight, so the MAD can be 0-1 ppm
            and would otherwise flag ordinary 3-5 ppm wobbles.

    Returns:
        A boolean Series aligned with ``readings``; ``True`` marks an outlier.
    """
    # Median and MAD, not mean and standard deviation: the outliers being
    # looked for would themselves inflate a mean-based threshold.
    median = readings.rolling(window, center=True, min_periods=window // 2 + 1).median()
    deviation = (readings - median).abs()
    mad = deviation.rolling(window, center=True, min_periods=window // 2 + 1).median() * MAD_TO_SIGMA
    # A zero MAD (perfectly flat window) would make every change infinite.
    threshold = (n_sigmas * mad.replace(0.0, np.nan)).fillna(np.inf)
    return (deviation > threshold) & (deviation > min_abs_ppm)


def spike_decay_rates(
    readings: pd.Series,
    *,
    min_rise_ppm: float = 60.0,
    min_excess_ppm: float = 80.0,
    n_after: int = 3,
    interval: pd.Timedelta = pd.Timedelta("10min"),
    baseline_window: int = 6,
) -> np.ndarray:
    """Fit an exponential decay rate to every sharp CO2 release.

    An event is a reading that rises by at least ``min_rise_ppm`` from the
    previous one and sits ``min_excess_ppm`` above the background (median of
    the preceding ``baseline_window`` readings), followed by ``n_after``
    readings at the regular interval that stay above the background. The rate
    is the negative slope of ``log(excess)`` against time.

    Args:
        readings: One sensor's CO2 readings in ppm, indexed by time, in time
            order, without duplicate timestamps.
        min_rise_ppm: Minimum jump from the previous reading.
        min_excess_ppm: Minimum height of the peak above the background.
        n_after: Number of following readings used in the fit.
        interval: The sensor's reporting interval; events with a missing or
            late reading are skipped so the time axis is exact.
        baseline_window: Readings before the event used for the background.

    Returns:
        Decay rates in 1/hour, one per event. Under the single-zone mass
        balance each rate estimates the ventilation rate (air changes per
        hour); local mixing near the sensor makes it an upper estimate.
    """
    values = readings.to_numpy(dtype=float)
    times = readings.index
    background = readings.rolling(baseline_window, min_periods=baseline_window // 2).median().shift(1).to_numpy()
    tolerance = interval / 10
    hours = np.arange(n_after + 1) * (interval / pd.Timedelta("1h"))

    rates = []
    for i in range(baseline_window, len(values) - n_after):
        if np.isnan(background[i]):
            continue
        is_release = values[i] - values[i - 1] >= min_rise_ppm and values[i] - background[i] >= min_excess_ppm
        if not is_release:
            continue
        # Require evenly spaced follow-up readings: a gap would put the
        # wrong time on the decay and bias the fitted rate.
        gaps = np.diff(times[i:i + n_after + 1]) - interval
        if any(abs(g) > tolerance for g in gaps):
            continue
        excess = values[i:i + n_after + 1] - background[i]
        if (excess <= 0).any():
            continue  # fell below background: log undefined, event too short
        slope = np.polyfit(hours, np.log(excess), 1)[0]
        rates.append(-slope)
    return np.asarray(rates)
