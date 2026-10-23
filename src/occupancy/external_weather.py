"""Loading of real Bureau of Meteorology daily weather observations.

External validation data for the environmental sensor log, per Keenan
Granland's third suggestion on the EdStem forum ("irming scope #81"): does
a sensor's reading make sense against real outdoor weather for the same
days?

Data source and its real limitation
------------------------------------
Files are BoM's public monthly "Daily Weather Observations" CSVs for
Moorabbin Airport (station 086077 -- the nearest official BoM station to
Monash Clayton), fetched by ``tools/fetch_bom_weather.py``. BoM's bulk
historical-download endpoint (Climate Data Online) actively blocks
automated requests as scraping; only a rolling ~15-month window of recent
months is fetchable this way. This means comparisons using this module can
only ever cover recent months (see the project report for exactly which
environmental sensor has usable overlap), not the full multi-year span of
the occupancy/environmental sensor data -- an honest constraint of the
data source, not a shortcut taken in this code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]

#: BoM's CSV encodes the degree sign as Windows-1252/Latin-1, not UTF-8.
BOM_ENCODING = "latin-1"

#: Number of preamble lines (title, prepared-at, copyright, notes, station,
#: blank) before the header row in each monthly file -- verified against
#: the actual downloaded files (see tools/fetch_bom_weather.py).
PREAMBLE_LINES = 6

#: Clean column names, by position, replacing BoM's verbose (and
#: encoding-fragile) header text.
COLUMN_NAMES = [
    "_unnamed",
    "date",
    "min_temp_c",
    "max_temp_c",
    "rainfall_mm",
    "evaporation_mm",
    "sunshine_hours",
    "max_gust_direction",
    "max_gust_speed_kmh",
    "max_gust_time",
    "temp_9am_c",
    "humidity_9am_pct",
    "cloud_9am_oktas",
    "wind_dir_9am",
    "wind_speed_9am_kmh",
    "pressure_9am_hpa",
    "temp_3pm_c",
    "humidity_3pm_pct",
    "cloud_3pm_oktas",
    "wind_dir_3pm",
    "wind_speed_3pm_kmh",
    "pressure_3pm_hpa",
]

#: Timezone the BoM observation times are reported in -- matches
#: occupancy.data_loading.LOCAL_TZ so the two datasets can be compared.
LOCAL_TZ = "Australia/Melbourne"


def load_bom_month(path: PathLike) -> pd.DataFrame:
    """Load one month's BoM Daily Weather Observations CSV.

    Args:
        path: Path to a single ``IDCJDW*.csv`` file, as saved by
            ``tools/fetch_bom_weather.py``.

    Returns:
        A DataFrame with one row per day and the columns in
        :data:`COLUMN_NAMES` (minus ``_unnamed``), with ``date`` parsed to
        a naive date and numeric columns coerced to float (BoM leaves a
        reading blank, not zero, when a station didn't record it that
        day -- these become NaN, not 0).
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        encoding=BOM_ENCODING,
        skiprows=PREAMBLE_LINES,
        header=0,
        names=COLUMN_NAMES,
    )
    df = df.drop(columns=["_unnamed"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    numeric_cols = [c for c in df.columns if c not in ("date", "max_gust_direction", "max_gust_time", "wind_dir_9am", "wind_dir_3pm")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_bom_weather(directory: PathLike) -> pd.DataFrame:
    """Load and concatenate every monthly BoM CSV in a directory.

    Args:
        directory: Directory containing one or more ``IDCJDW*.csv`` files
            (e.g. ``data/raw/bom``).

    Returns:
        A single DataFrame covering every month found, sorted by date, with
        duplicate dates (if the same month was fetched twice) dropped.

    Raises:
        FileNotFoundError: If ``directory`` contains no matching CSV files.
    """
    directory = Path(directory)
    files = sorted(directory.glob("IDCJDW*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No BoM CSV files found in {directory}. Run "
            "tools/fetch_bom_weather.py first."
        )
    months = [load_bom_month(f) for f in files]
    out = pd.concat(months, ignore_index=True)
    return out.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
