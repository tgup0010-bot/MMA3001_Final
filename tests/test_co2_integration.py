"""Unit tests for occupancy.co2_integration.

Cases are hand-computable (constant and linear integrands, and the exact
worked unequal-spacing example from MMA3001 Week 7.1's own notes) so the
expected values can be verified by hand, not just re-derived by the code
under test.
"""

import pandas as pd
import pytest

from occupancy.co2_integration import (
    richardson_order_check,
    simpsons_integral,
    trapezoidal_integral,
    trapezoidal_integral_gap_aware,
)


def test_trapezoidal_integral_of_constant_value_is_height_times_duration():
    # A constant 600 ppm held for exactly 2 hours integrates to 1200 ppm.h.
    times = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"], utc=True)
    values = [600.0, 600.0]
    assert trapezoidal_integral(times, values) == pytest.approx(1200.0)


def test_trapezoidal_integral_matches_week7_worked_example():
    """MMA3001 Week 7.1's own worked example: unequal gaps t=0,0.4,1.0
    (hours) with values 0.00, 0.92, 2.60 -- using the actual coordinates
    gives the notes' own stated result (there quoted as 2.5 for the
    *derivative* example, but the same unequal-spacing data integrated
    directly gives a hand-checkable trapezoidal result)."""
    times = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:24", "2024-01-01 01:00"], utc=True
    )  # 0, 0.4, 1.0 hours
    values = [0.00, 0.92, 2.60]
    # Panel 1: (0.4)*(0.00+0.92)/2 = 0.184
    # Panel 2: (0.6)*(0.92+2.60)/2 = 1.056
    expected = 0.4 * (0.00 + 0.92) / 2 + 0.6 * (0.92 + 2.60) / 2
    assert trapezoidal_integral(times, values) == pytest.approx(expected)


def test_trapezoidal_integral_requires_at_least_two_points():
    with pytest.raises(ValueError):
        trapezoidal_integral(pd.to_datetime(["2024-01-01"], utc=True), [1.0])


def test_simpsons_integral_of_constant_value_is_height_times_duration():
    # 5 equally spaced nodes (4 intervals, even -- valid for Simpson),
    # each 0.5h apart, all at 600 ppm -> total duration 2h -> 1200 ppm.h.
    values = [600.0, 600.0, 600.0, 600.0, 600.0]
    assert simpsons_integral(values, bin_hours=0.5) == pytest.approx(1200.0)


def test_simpsons_integral_is_exact_for_a_quadratic():
    # Simpson's rule integrates any quadratic exactly. f(x) = x^2 on
    # [0, 2] with h=0.5: nodes at 0, 0.5, 1.0, 1.5, 2.0.
    # True integral of x^2 from 0 to 2 is 8/3.
    xs = [0.0, 0.5, 1.0, 1.5, 2.0]
    values = [x**2 for x in xs]
    assert simpsons_integral(values, bin_hours=0.5) == pytest.approx(8.0 / 3.0)


def test_simpsons_integral_rejects_odd_number_of_intervals():
    with pytest.raises(ValueError):
        simpsons_integral([1.0, 2.0, 3.0, 4.0], bin_hours=1.0)  # 3 intervals, odd


def test_trapezoidal_integral_gap_aware_excludes_panels_beyond_max_gap():
    # Two readings 1h apart (normal), then a 100h outage, then one more.
    times = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-05 05:00"], utc=True
    )
    values = [600.0, 620.0, 610.0]
    result = trapezoidal_integral_gap_aware(times, values, max_gap_hours=2.0)
    # Only the first (1h) panel is included; the ~100h panel is excluded.
    expected_included_panel = 1.0 * (600.0 + 620.0) / 2
    assert result["total"] == pytest.approx(expected_included_panel)
    assert result["n_panels_excluded"] == 1
    assert result["hours_included"] == pytest.approx(1.0)
    assert result["hours_excluded"] == pytest.approx(100.0, abs=0.1)


def test_trapezoidal_integral_gap_aware_includes_everything_within_limit():
    times = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 01:00"], utc=True)
    values = [600.0, 600.0]
    result = trapezoidal_integral_gap_aware(times, values, max_gap_hours=2.0)
    assert result["total"] == pytest.approx(600.0)
    assert result["n_panels_excluded"] == 0


def test_richardson_order_check_reports_expected_reduction_factor():
    result = richardson_order_check(estimate_h=100.0, estimate_h_half=102.0, order=2)
    assert result["change"] == pytest.approx(2.0)
    assert result["expected_reduction_factor"] == 4  # 2**2, trapezoidal


@pytest.mark.parametrize("rule, expected_ratio", [("trapezoidal", 4.0), ("simpson", 16.0)])
def test_halving_the_step_reduces_error_at_the_theoretical_rate(rule, expected_ratio):
    """Order of accuracy against an exact answer: integral of sin(t) on [0, pi] = 2.

    Halving h should cut the error by 2**p: 4 for trapezoidal (p=2) and 16 for
    Simpson's (p=4). This checks the implementations converge as the theory says,
    not just that the bookkeeping in richardson_order_check is right.
    """
    import numpy as np

    def error(n_intervals):
        hours = np.linspace(0, np.pi, n_intervals + 1)
        values = np.sin(hours)
        if rule == "trapezoidal":
            times = pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h")
            estimate = trapezoidal_integral(times, values)
        else:
            estimate = simpsons_integral(values, bin_hours=np.pi / n_intervals)
        return abs(estimate - 2.0)

    ratio = error(16) / error(32)
    assert ratio == pytest.approx(expected_ratio, rel=0.02)
