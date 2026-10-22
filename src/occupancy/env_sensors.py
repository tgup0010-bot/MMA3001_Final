"""Loading and tidying of the raw environmental sensor log.

The source file stores one row per sensor "batch" report: a ``jsondata``
column holding a JSON array where each element is one measured variable
(e.g. Carbon dioxide, Temperature) with its own Unix-epoch ``time`` and
``value``. This module turns that nested structure into a tidy long-format
table -- one row per (sensor, variable, timestamp, value) -- which is what
:mod:`occupancy.sensor_matching` and ordinary pandas resampling/pivoting
need.

Note this is entirely independent of :mod:`occupancy.data_loading` (the
occupancy event log): the two datasets are joined, if at all, only through
:mod:`occupancy.sensor_matching`'s data-driven correlation analysis -- not
through the room-identifier metadata, which does not reliably link them
(see the project README, Known data limitations).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

#: Columns required to be present in the raw CSV.
REQUIRED_COLUMNS = ("sensorid", "jsondata")

#: The raw ``time`` field inside each JSON variable is Unix-epoch seconds
#: in UTC; readings are converted to this zone for consistency with the
#: occupancy event log (see occupancy.data_loading.LOCAL_TZ).
LOCAL_TZ = "Australia/Melbourne"


def load_env_sensor_log(path: PathLike) -> pd.DataFrame:
    """Load and flatten the raw environmental sensor CSV.

    Args:
        path: Path to the raw CSV (e.g.
            ``data/raw/5EnvSensor_MayToDec2024_180kRows.csv``).

    Returns:
        A tidy long-format DataFrame with one row per (sensor, variable,
        reading) and columns:

        - ``sensorid`` -- the raw sensor identifier string.
        - ``time`` -- reading timestamp, tz-aware (:data:`LOCAL_TZ`).
        - ``variable`` -- the measured quantity's name, e.g.
          ``"Carbon dioxide"``, ``"Temperature"``.
        - ``unit`` -- the variable's unit, e.g. ``"ppm"``, ``"°C"``.
        - ``value`` -- the numeric reading.

        Note this is long-format (many rows per timestamp, one per
        variable) rather than wide -- use :func:`pivot_variable` to get a
        single variable as a per-sensor time series.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If any column in :data:`REQUIRED_COLUMNS` is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Environmental sensor file not found: {path}. See "
            "data/raw/README.md for where to obtain it."
        )

    # sensorid is an opaque identifier, not a quantity -- read it as a
    # string explicitly. Left to type inference, pandas parses the
    # digit-only values as int64, which then fails to match the string
    # keys in occupancy.sensors.SENSOR_LABELS and cannot be JSON-serialised
    # as a dict key.
    df = pd.read_csv(path, dtype={"sensorid": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Environmental sensor CSV is missing expected column(s): "
            f"{missing}. Expected at least: {REQUIRED_COLUMNS}"
        )

    return _flatten(df)


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the ``jsondata`` column into tidy long-format rows.

    Each row of ``jsondata`` is a JSON array of variable readings; in this
    dataset every variable carries exactly one (time, value) pair per row
    (verified against the raw file), so this always emits one output row
    per variable per input row -- there is no need to further explode a
    multi-point ``values`` list.

    A small fraction of readings represent a failed sensor read: their
    ``values`` entry carries only ``time`` (and sometimes an ``errorCode``)
    with no ``value`` at all. These are dropped rather than treated as a
    reading of 0 or NaN-as-data, since they are not measurements.
    """
    records: list[dict] = []
    for sensorid, raw_json in zip(df["sensorid"], df["jsondata"]):
        try:
            variables = json.loads(raw_json)
        except (TypeError, ValueError):
            continue  # malformed/missing payload -- skip rather than crash
        for item in variables:
            values = item.get("values") or []
            if not values or "value" not in values[0]:
                continue  # no values list, or a failed read (see docstring)
            var = item.get("variable", {})
            records.append(
                {
                    "sensorid": sensorid,
                    "time": values[0]["time"],
                    "variable": var.get("name"),
                    "unit": var.get("unit"),
                    "value": values[0]["value"],
                }
            )

    out = pd.DataFrame.from_records(records)
    out["time"] = pd.to_datetime(out["time"], unit="s", utc=True).dt.tz_convert(
        LOCAL_TZ
    )
    return out.sort_values(["sensorid", "variable", "time"]).reset_index(drop=True)


def pivot_variable(long_df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Extract one measured variable as a per-sensor time series.

    Args:
        long_df: Tidy long-format DataFrame as returned by
            :func:`load_env_sensor_log`.
        variable: The ``variable`` value to extract, e.g.
            ``"Carbon dioxide"``.

    Returns:
        A DataFrame with columns ``["sensorid", "time", "value"]`` holding
        only readings for ``variable``, sorted by sensor then time.

    Raises:
        ValueError: If ``variable`` does not appear in ``long_df`` at all
            (almost always a typo -- fail loudly rather than silently
            returning an empty table).
    """
    subset = long_df[long_df["variable"] == variable]
    if subset.empty:
        available = sorted(long_df["variable"].dropna().unique())
        raise ValueError(
            f"Variable {variable!r} not found in the data. "
            f"Available variables: {available}"
        )
    return subset[["sensorid", "time", "value"]].reset_index(drop=True)


def sensor_ids(long_df: pd.DataFrame) -> list[str]:
    """Return the distinct sensor identifiers present.

    Args:
        long_df: Tidy long-format DataFrame as returned by
            :func:`load_env_sensor_log`.

    Returns:
        Sensor ids in ``long_df``'s own row order (which
        :func:`load_env_sensor_log` sorts by sensor id, then variable, then
        time) -- not necessarily the order they first appear in the raw file.
    """
    return list(pd.unique(long_df["sensorid"]))
