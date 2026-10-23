"""Compact processed copies of the raw data, committed so the project runs from a clone.

The two raw course files are too large for GitHub (1.5 GB and 337 MB; the
per-file limit is 100 MB), so they stay out of version control. Every
experiment script, however, only needs three much smaller derived tables:

- the environmental readings for the three variables analysed (CO2,
  temperature, humidity), exactly as :func:`occupancy.env_sensors.load_env_sensor_log`
  returns them;
- the occupancy log already resampled to the 15-minute per-room grid by
  :func:`occupancy.preprocessing.resample_occupancy`;
- each CO2 reading's server receive time (``createdate``), used to detect
  re-transmitted stale packets.

``tools/build_processed_data.py`` writes these tables once from the raw
files. The ``*_or_processed`` loaders below then use the raw files when
they are present and fall back to the committed tables otherwise, so both
paths feed identical data into the analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import pandas as pd

from occupancy.data_loading import load_occupancy_log
from occupancy.env_sensors import LOCAL_TZ, load_env_sensor_log
from occupancy.preprocessing import resample_occupancy

PathLike = Union[str, Path]

PROCESSED_DIR = Path("data/processed")
ENV_PROCESSED_PATH = PROCESSED_DIR / "env_readings.csv.gz"
OCCUPANCY_PROCESSED_PATH = PROCESSED_DIR / "occupancy_15min.csv.gz"
RECEIVE_TIMES_PROCESSED_PATH = PROCESSED_DIR / "co2_receive_times.csv.gz"
BOM_PROCESSED_DIR = PROCESSED_DIR / "bom"

#: The only environmental variables any script analyses. Keeping just these
#: keeps the committed table small.
VARIABLES_KEPT = ("Carbon dioxide", "Temperature", "Humidity")

#: The processed occupancy table is stored on this grid only; other bin sizes
#: need the raw file.
PROCESSED_BIN_SIZE = "15min"

_MISSING_DATA_HELP = (
    "Neither the raw file ({raw}) nor the processed copy ({processed}) was "
    "found. Run the script from the repository root, or see "
    "data/raw/README.md."
)


#: Timestamps are written as UTC ISO-8601 strings with microseconds, so a
#: save/load round trip reproduces every value exactly. The column's
#: resolution (``datetime64[us]`` vs ``[ns]``) can differ from the raw
#: loaders' depending on the pandas version; no calculation depends on it.
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"


def _to_text(times: pd.Series) -> pd.Series:
    """Format tz-aware timestamps as exact UTC ISO-8601 strings."""
    return times.dt.tz_convert("UTC").dt.strftime(_TIME_FORMAT)


def _from_text(text: pd.Series, tz: str = LOCAL_TZ) -> pd.Series:
    """Inverse of :func:`_to_text`, returned in ``tz``."""
    return pd.to_datetime(text, utc=True, format="ISO8601").dt.tz_convert(tz)


# --- environmental readings ------------------------------------------------

def save_env_readings(env_long: pd.DataFrame, path: PathLike = ENV_PROCESSED_PATH) -> pd.DataFrame:
    """Write the analysed variables of a long-format env table to ``path``.

    Args:
        env_long: Output of :func:`occupancy.env_sensors.load_env_sensor_log`.
        path: Destination ``.csv.gz`` file.

    Returns:
        The subset that was written (rows for :data:`VARIABLES_KEPT` only).
    """
    subset = env_long[env_long["variable"].isin(VARIABLES_KEPT)].reset_index(drop=True)
    out = subset.assign(time=_to_text(subset["time"]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return subset


def load_env_readings(path: PathLike = ENV_PROCESSED_PATH) -> pd.DataFrame:
    """Read a table written by :func:`save_env_readings`.

    Args:
        path: The ``.csv.gz`` file to read.

    Returns:
        A DataFrame with the same columns and dtypes as
        :func:`occupancy.env_sensors.load_env_sensor_log`.
    """
    df = pd.read_csv(path, dtype={"sensorid": str})
    df["time"] = _from_text(df["time"])
    return df


def env_readings_raw_or_processed(
    raw_path: PathLike, processed_path: PathLike = ENV_PROCESSED_PATH
) -> pd.DataFrame:
    """Load the env readings from the raw CSV if present, else the processed copy.

    Args:
        raw_path: The raw environmental sensor CSV.
        processed_path: The committed copy written by :func:`save_env_readings`.

    Returns:
        Long-format readings of :data:`VARIABLES_KEPT`, with the columns of
        :func:`occupancy.env_sensors.load_env_sensor_log` (time in local
        time, value in each variable's unit: ppm, degrees C, % RH).

    Raises:
        FileNotFoundError: If neither file exists.
    """
    raw_path, processed_path = Path(raw_path), Path(processed_path)
    if raw_path.exists():
        env_long = load_env_sensor_log(raw_path)
        return env_long[env_long["variable"].isin(VARIABLES_KEPT)].reset_index(drop=True)
    if processed_path.exists():
        print(f"(raw file not found -- using processed copy {processed_path})")
        return load_env_readings(processed_path)
    raise FileNotFoundError(_MISSING_DATA_HELP.format(raw=raw_path, processed=processed_path))


# --- resampled occupancy ---------------------------------------------------

def save_resampled_occupancy(resampled: pd.DataFrame, path: PathLike = OCCUPANCY_PROCESSED_PATH) -> None:
    """Write the 15-minute per-room occupancy grid to ``path``.

    Args:
        resampled: Output of :func:`occupancy.preprocessing.resample_occupancy`
            with ``bin_size=PROCESSED_BIN_SIZE``.
        path: Destination ``.csv.gz`` file.
    """
    out = resampled.assign(collecteddate=_to_text(resampled["collecteddate"]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_resampled_occupancy(path: PathLike = OCCUPANCY_PROCESSED_PATH) -> pd.DataFrame:
    """Read a table written by :func:`save_resampled_occupancy`.

    Args:
        path: The ``.csv.gz`` file to read.

    Returns:
        Columns ``floorspaceid``, ``collecteddate`` (local time) and
        ``occupied`` (1.0, 0.0, or NaN for a gap longer than 2 hours).
    """
    df = pd.read_csv(path, dtype={"floorspaceid": str})
    df["collecteddate"] = _from_text(df["collecteddate"])
    return df


def resampled_occupancy_raw_or_processed(
    raw_path: PathLike,
    bin_size: str = PROCESSED_BIN_SIZE,
    processed_path: PathLike = OCCUPANCY_PROCESSED_PATH,
) -> pd.DataFrame:
    """Resample the raw occupancy log if present, else load the processed grid.

    Args:
        raw_path: The raw occupancy event-log CSV.
        bin_size: Grid spacing; the processed copy only serves
            :data:`PROCESSED_BIN_SIZE`.
        processed_path: The committed copy written by
            :func:`save_resampled_occupancy`.

    Returns:
        The per-room grid, as described in :func:`load_resampled_occupancy`.

    Raises:
        FileNotFoundError: If the raw file is missing and either the
            processed copy is missing or ``bin_size`` is not
            :data:`PROCESSED_BIN_SIZE`.
    """
    raw_path, processed_path = Path(raw_path), Path(processed_path)
    if raw_path.exists():
        return resample_occupancy(load_occupancy_log(raw_path), bin_size=bin_size)
    if processed_path.exists() and pd.Timedelta(bin_size) == pd.Timedelta(PROCESSED_BIN_SIZE):
        print(f"(raw file not found -- using processed copy {processed_path})")
        return load_resampled_occupancy(processed_path)
    raise FileNotFoundError(_MISSING_DATA_HELP.format(raw=raw_path, processed=processed_path))


# --- CO2 receive times -----------------------------------------------------

def co2_receive_times(raw_path: PathLike) -> pd.DataFrame:
    """Each CO2 reading's own timestamp alongside its server receive time.

    A reading received long after its own timestamp is a re-sent stale
    packet, not a new measurement -- see ``scripts/multi_sensor_experiment.py``.

    Args:
        raw_path: The raw environmental sensor CSV (needs ``createdate``).

    Returns:
        Columns ``sensorid``, ``reading_time`` and ``received`` (both UTC).
    """
    raw = pd.read_csv(raw_path, usecols=["sensorid", "jsondata", "createdate"], dtype={"sensorid": str})

    def co2_time(payload: str):
        """Return the CO2 reading's Unix time from one JSON payload, or None."""
        try:
            for item in json.loads(payload):
                if item["variable"]["name"] == "Carbon dioxide":
                    return item["values"][0]["time"]
        except (TypeError, ValueError, KeyError, IndexError):
            pass
        return None

    return pd.DataFrame(
        {
            "sensorid": raw["sensorid"],
            "reading_time": pd.to_datetime(raw["jsondata"].map(co2_time), unit="s", utc=True),
            "received": pd.to_datetime(raw["createdate"], utc=True, format="mixed"),
        }
    ).dropna()


def save_receive_times(received: pd.DataFrame, path: PathLike = RECEIVE_TIMES_PROCESSED_PATH) -> None:
    """Write the output of :func:`co2_receive_times` to ``path``.

    Args:
        received: Output of :func:`co2_receive_times`.
        path: Destination ``.csv.gz`` file.
    """
    out = received.assign(
        reading_time=_to_text(received["reading_time"]), received=_to_text(received["received"])
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def load_receive_times(path: PathLike = RECEIVE_TIMES_PROCESSED_PATH) -> pd.DataFrame:
    """Read a table written by :func:`save_receive_times`.

    Args:
        path: The ``.csv.gz`` file to read.

    Returns:
        Columns ``sensorid``, ``reading_time`` and ``received`` (both UTC).
    """
    df = pd.read_csv(path, dtype={"sensorid": str})
    for col in ("reading_time", "received"):
        df[col] = _from_text(df[col], tz="UTC")
    return df


def receive_times_raw_or_processed(
    raw_path: PathLike, processed_path: PathLike = RECEIVE_TIMES_PROCESSED_PATH
) -> pd.DataFrame:
    """Compute receive times from the raw CSV if present, else load the processed copy.

    Args:
        raw_path: The raw environmental sensor CSV.
        processed_path: The committed copy written by :func:`save_receive_times`.

    Returns:
        As :func:`co2_receive_times`.

    Raises:
        FileNotFoundError: If neither file exists.
    """
    raw_path, processed_path = Path(raw_path), Path(processed_path)
    if raw_path.exists():
        return co2_receive_times(raw_path)
    if processed_path.exists():
        return load_receive_times(processed_path)
    raise FileNotFoundError(_MISSING_DATA_HELP.format(raw=raw_path, processed=processed_path))


# --- Bureau of Meteorology CSVs --------------------------------------------

def bom_dir_raw_or_processed(raw_dir: PathLike, processed_dir: PathLike = BOM_PROCESSED_DIR) -> Path:
    """Choose where to read the Bureau of Meteorology CSVs from.

    Args:
        raw_dir: The fetch script's output directory (``data/raw/bom``).
        processed_dir: The committed copy's directory.

    Returns:
        ``raw_dir`` if it holds ``IDCJDW*.csv`` files, else ``processed_dir``.
    """
    raw_dir = Path(raw_dir)
    if any(raw_dir.glob("IDCJDW*.csv")):
        return raw_dir
    return Path(processed_dir)
