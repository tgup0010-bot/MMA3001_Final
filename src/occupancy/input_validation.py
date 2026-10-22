"""Validation of the forecaster's inputs: invalid, unsupported and extrapolated.

The report (Section 2.3) distinguishes three kinds of problem input, and
this module implements the two that can be checked on a single query:

- **Invalid** -- physically impossible or outside the encoding's range
  (e.g. negative CO2, hour 25, occupancy 7 of 5 zones). Rejected with a
  reason; the model is never asked to predict from these.
- **Extrapolated** -- valid, but outside the range the model was trained
  on (e.g. CO2 of 1800 ppm when training data stopped near 1500 ppm). A
  prediction is still made, but flagged: Section 4.1 of the report shows
  every model under-predicts at the top of its training range, so
  accuracy there is not established.

The third kind, **unsupported** inputs (another room's sensor, another bin
size), cannot be recognised from one row of numbers and is handled by
documentation instead (report Section 2.3).
"""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd

#: Physically valid range for every CO2 input, in ppm. The lower bound sits
#: below outdoor air (about 420 ppm) to allow for sensor noise, since indoor
#: air cannot hold less CO2 than the outdoor air that ventilates it. The
#: upper bound is the Safe Work Australia 8-hour exposure standard (5000
#: ppm); readings above it indicate a fault or an unsafe space, not a
#: ventilation-control case.
CO2_VALID_PPM = (350.0, 5000.0)

#: Number of monitored occupancy zones; building occupancy is a count of
#: occupied zones, so it must be a whole number from 0 to this value.
N_ZONES = 5

CO2_FEATURES = ("co2_now", "co2_lag1", "co2_rolling_1h")


def validate_query(
    *,
    co2_now: float,
    co2_lag1: float,
    co2_rolling_1h: float,
    building_occupancy: float,
    hour: int,
    day_of_week: int,
) -> list[str]:
    """Check one forecast query for invalid values.

    Args:
        co2_now: Current 15-minute mean CO2, in ppm.
        co2_lag1: Mean CO2 of the previous 15-minute bin, in ppm.
        co2_rolling_1h: Mean CO2 over the last hour (four bins), in ppm.
        building_occupancy: Number of the monitored zones occupied, 0 to
            :data:`N_ZONES`.
        hour: Local hour of day, 0 to 23.
        day_of_week: Day of week, 0 (Monday) to 6 (Sunday).

    Returns:
        A list of human-readable problems; empty if the query is valid.
        Every problem is reported, not just the first, so a user can fix
        them all at once.
    """
    problems = []
    co2_values = {"co2_now": co2_now, "co2_lag1": co2_lag1, "co2_rolling_1h": co2_rolling_1h}
    lo, hi = CO2_VALID_PPM
    for name, value in co2_values.items():
        # NaN compares False with everything, so it must be caught explicitly
        # or it would slip through the range check below.
        if value is None or not math.isfinite(value):
            problems.append(f"{name} is missing or not a number")
        elif not lo <= value <= hi:
            problems.append(f"{name}={value:g} ppm is outside the valid range {lo:g}-{hi:g} ppm")

    if not (math.isfinite(building_occupancy) and float(building_occupancy).is_integer()
            and 0 <= building_occupancy <= N_ZONES):
        problems.append(f"building_occupancy={building_occupancy} must be a whole number from 0 to {N_ZONES}")
    if not (float(hour).is_integer() and 0 <= hour <= 23):
        problems.append(f"hour={hour} must be a whole number from 0 to 23")
    if not (float(day_of_week).is_integer() and 0 <= day_of_week <= 6):
        problems.append(f"day_of_week={day_of_week} must be a whole number from 0 (Mon) to 6 (Sun)")
    return problems


def training_ranges(train: pd.DataFrame, columns: tuple[str, ...] = CO2_FEATURES) -> dict[str, tuple[float, float]]:
    """Record the minimum and maximum of each column seen in training.

    Args:
        train: The training rows, as produced by
            :func:`occupancy.evaluation.chronological_split`.
        columns: Which feature columns to record.

    Returns:
        ``{column: (min, max)}`` -- the trained domain used by
        :func:`extrapolation_warnings`.
    """
    return {c: (float(train[c].min()), float(train[c].max())) for c in columns}


def extrapolation_warnings(
    query: Mapping[str, float], ranges: Mapping[str, tuple[float, float]]
) -> list[str]:
    """Flag valid inputs that fall outside the trained domain.

    Args:
        query: Feature name to value for one forecast query.
        ranges: Output of :func:`training_ranges`.

    Returns:
        A list of warnings; empty if every checked input lies within the
        range seen in training. A warning does not block the prediction:
        the input is physically possible, but the model's accuracy there is
        unverified.
    """
    warnings = []
    for name, (lo, hi) in ranges.items():
        value = query.get(name)
        if value is not None and not lo <= value <= hi:
            warnings.append(
                f"{name}={value:g} ppm is outside the training range {lo:.0f}-{hi:.0f} ppm "
                "(extrapolation: accuracy not validated)"
            )
    return warnings
