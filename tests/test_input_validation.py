"""Unit tests for occupancy.input_validation."""

import math

import pandas as pd
import pytest

from occupancy.input_validation import extrapolation_warnings, training_ranges, validate_query

VALID = dict(co2_now=550.0, co2_lag1=540.0, co2_rolling_1h=530.0, building_occupancy=2, hour=14, day_of_week=2)


def test_valid_query_has_no_problems():
    assert validate_query(**VALID) == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("co2_now", -10.0),
        ("co2_now", 6000.0),
        ("co2_lag1", math.nan),
        ("co2_rolling_1h", math.inf),
        ("building_occupancy", 6),
        ("building_occupancy", 2.5),
        ("building_occupancy", -1),
        ("hour", 24),
        ("hour", 13.5),
        ("day_of_week", 7),
    ],
)
def test_each_invalid_field_is_rejected_with_its_name(field, value):
    problems = validate_query(**{**VALID, field: value})
    assert len(problems) == 1
    assert field in problems[0]


def test_every_problem_is_reported_not_just_the_first():
    problems = validate_query(**{**VALID, "co2_now": -1.0, "hour": 25, "building_occupancy": 9})
    assert len(problems) == 3


def test_range_bounds_are_inclusive():
    assert validate_query(**{**VALID, "co2_now": 350.0, "co2_lag1": 5000.0, "hour": 0, "day_of_week": 6}) == []


def test_extrapolation_flags_only_values_outside_training_range():
    train = pd.DataFrame({"co2_now": [420.0, 900.0], "co2_lag1": [415.0, 880.0], "co2_rolling_1h": [418.0, 870.0]})
    ranges = training_ranges(train)
    assert ranges["co2_now"] == (420.0, 900.0)
    inside = {"co2_now": 600.0, "co2_lag1": 590.0, "co2_rolling_1h": 580.0}
    assert extrapolation_warnings(inside, ranges) == []
    outside = {**inside, "co2_now": 1500.0}
    warnings = extrapolation_warnings(outside, ranges)
    assert len(warnings) == 1 and "co2_now" in warnings[0]
