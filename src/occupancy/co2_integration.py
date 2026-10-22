"""Newton-Cotes numerical integration of CO2 concentration over time.

Applies MMA3001 Week 6's integration methods to a genuine engineering
question: how much CO2 has *accumulated* over a period, not just what its
instantaneous level was? This mirrors Week 6's own framing exactly --
"engineering instruments commonly record local or instantaneous values
rather than the total quantity required for a decision" -- CO2 sensors
report a concentration at an instant, but the exposure-relevant quantity
(e.g. for ventilation-adequacy assessment) is its time integral.

Formulas match MMA3001 Week 6.1 / 6.2 exactly:

    Q_trap = sum_i (x[i+1] - x[i]) * (f[i] + f[i+1]) / 2
    Q_simpson = (h/3) * [f0 + 4*sum(odd) + 2*sum(even) + fn]   (equal spacing only)

Both are implemented directly (not via scipy) so the calculation is
auditable against the taught formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def trapezoidal_integral(times: pd.Series | np.ndarray, values: pd.Series | np.ndarray) -> float:
    """Newton-Cotes trapezoidal rule, honouring the true (possibly irregular) node spacing.

    Per Week 6.1/7.1: real sensor timestamps are rarely perfectly evenly
    spaced, and averaging the gaps and pretending they are equal invents
    error that has nothing to do with measurement noise. This uses each
    consecutive pair's *actual* time gap.

    Args:
        times: Timestamps (any type ``pandas`` can subtract to get a
            ``Timedelta``, e.g. a tz-aware ``DatetimeIndex`` or column).
        values: The integrand's value at each timestamp (same length as
            ``times``), e.g. CO2 concentration in ppm.

    Returns:
        The accumulated integral, in (value-unit x hours) -- e.g. ppm.h
        for a CO2 series -- computed as
        :math:`\\sum_i (t_{i+1}-t_i)(f_i+f_{i+1})/2`.

    Raises:
        ValueError: If fewer than 2 points are given.
    """
    times = pd.to_datetime(pd.Series(times).reset_index(drop=True))
    values = pd.Series(values).reset_index(drop=True).astype(float)
    if len(times) < 2:
        raise ValueError("Need at least 2 (time, value) points to integrate.")

    # Each panel uses its own width (h_i), not an average step: averaging
    # would add error unrelated to the data. Hours, so the result is ppm.h.
    gaps_hours = times.diff().dt.total_seconds().to_numpy()[1:] / 3600.0
    f = values.to_numpy()
    panel_areas = gaps_hours * (f[:-1] + f[1:]) / 2.0
    return float(panel_areas.sum())


def trapezoidal_integral_gap_aware(
    times: pd.Series | np.ndarray,
    values: pd.Series | np.ndarray,
    max_gap_hours: float,
) -> dict:
    """Trapezoidal integration that excludes panels spanning a sensor outage.

    :func:`trapezoidal_integral` applies the formula literally to whatever
    gap exists between consecutive readings -- correct for genuine
    irregular sampling, but if two readings are separated by a real sensor
    outage (hours or even days with no data), the formula silently implies
    the value varied *linearly* across that whole gap, which is not a
    measurement, it's an unstated assumption. This mirrors
    ``occupancy.preprocessing.resample_occupancy``'s own ``max_gap``
    parameter: panels wider than ``max_gap_hours`` are excluded from the
    total rather than integrated across, and the excluded duration is
    reported so the coverage gap is visible, not silently absorbed.

    Args:
        times: Timestamps, as in :func:`trapezoidal_integral`.
        values: Integrand values, as in :func:`trapezoidal_integral`.
        max_gap_hours: The longest inter-reading gap still integrated
            across.

    Returns:
        A dict with ``total`` (the integral over included panels only,
        value-unit x hours), ``hours_included``, ``hours_excluded``, and
        ``n_panels_excluded``.
    """
    times = pd.to_datetime(pd.Series(times).reset_index(drop=True))
    values = pd.Series(values).reset_index(drop=True).astype(float)
    if len(times) < 2:
        raise ValueError("Need at least 2 (time, value) points to integrate.")

    gaps_hours = times.diff().dt.total_seconds().to_numpy()[1:] / 3600.0
    f = values.to_numpy()
    panel_areas = gaps_hours * (f[:-1] + f[1:]) / 2.0

    # A panel wider than max_gap spans an outage: integrating it assumes CO2
    # varied linearly through days of missing data. On sensor 0326 that one
    # assumption inflated cumulative exposure about 4x (9.54M vs 2.31M ppm.h),
    # so such panels are dropped and their duration reported, not hidden.
    included = gaps_hours <= max_gap_hours
    return {
        "total": float(panel_areas[included].sum()),
        "hours_included": float(gaps_hours[included].sum()),
        "hours_excluded": float(gaps_hours[~included].sum()),
        "n_panels_excluded": int((~included).sum()),
    }


def simpsons_integral(values: pd.Series | np.ndarray, bin_hours: float) -> float:
    """Composite Simpson's 1/3 rule -- requires equally spaced samples.

    Args:
        values: The integrand's value at each equally spaced node, e.g.
            CO2 concentration (ppm) on a regular time grid. Must have an
            odd length (an even number of intervals) -- Simpson's 1/3
            rule's classical requirement (Week 6.1).
        bin_hours: The (constant) spacing between nodes, in hours.

    Returns:
        The accumulated integral, in (value-unit x hours).

    Raises:
        ValueError: If fewer than 3 points are given, or the number of
            intervals is not even.
    """
    f = np.asarray(values, dtype=float)
    n = len(f) - 1  # number of intervals, not points
    if n < 2:
        raise ValueError("Need at least 3 points (2 intervals) for Simpson's rule.")
    # Simpson's pairs intervals into parabolic panels; silently dropping a
    # leftover interval would change the answer, so refuse instead.
    if n % 2 != 0:
        raise ValueError(
            f"Simpson's 1/3 rule needs an even number of intervals, got {n} "
            "(odd number of points required)."
        )

    # Weights 1, 4, 2, 4, ..., 2, 4, 1: interior odd nodes are parabola
    # midpoints (weight 4), interior even nodes are shared panel ends (weight 2).
    odd_sum = f[1:-1:2].sum()
    even_sum = f[2:-1:2].sum()
    return float((bin_hours / 3.0) * (f[0] + 4 * odd_sum + 2 * even_sum + f[-1]))


def richardson_order_check(
    estimate_h: float, estimate_h_half: float, order: int
) -> dict:
    """Check whether halving the step reduced error by the expected factor.

    Per Week 6.2: halving ``h`` should reduce a method of order ``p``'s
    leading error by about ``2**p`` -- 4x for trapezoidal (p=2), 16x for
    Simpson (p=4). Since the true integral is unknown here (real sensor
    data, not a test problem with a known answer), this reports the
    *change* between the two estimates as convergence evidence, exactly
    as Week 6.2 describes -- not a claim of the true error.

    Args:
        estimate_h: The integral estimate at step size ``h``.
        estimate_h_half: The estimate at step size ``h/2``.
        order: The method's assumed order of accuracy (2 for trapezoidal,
            4 for Simpson).

    Returns:
        A dict with ``change`` (the raw difference between estimates),
        ``pct_change`` (relative to the finer estimate), and
        ``expected_reduction_factor`` (``2**order``, for comparison against
        how much the change itself shrank relative to a still-finer pair,
        if available).
    """
    # The true integral of real sensor data is unknown, so only the change
    # between resolutions can be measured -- evidence of convergence, not error.
    change = estimate_h_half - estimate_h
    pct_change = 100.0 * change / estimate_h_half if estimate_h_half else float("nan")
    return {
        "change": change,
        "pct_change": pct_change,
        "expected_reduction_factor": 2**order,
    }
