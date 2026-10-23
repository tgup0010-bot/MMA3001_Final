"""Build the compact processed data tables committed in ``data/processed/``.

Run once, from the repository root, with the raw course files in
``data/raw/`` (see ``data/raw/README.md``). It writes the tables described
in :mod:`occupancy.processed_data`, which let every experiment script run
from a fresh clone without the raw files, then checks that each table reads
back identical to what the raw files produce.

Usage:
    python tools/build_processed_data.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from occupancy.data_loading import load_occupancy_log
from occupancy.env_sensors import load_env_sensor_log
from occupancy.preprocessing import resample_occupancy
from occupancy.processed_data import (
    BOM_PROCESSED_DIR,
    ENV_PROCESSED_PATH,
    OCCUPANCY_PROCESSED_PATH,
    PROCESSED_BIN_SIZE,
    RECEIVE_TIMES_PROCESSED_PATH,
    co2_receive_times,
    load_env_readings,
    load_receive_times,
    load_resampled_occupancy,
    save_env_readings,
    save_receive_times,
    save_resampled_occupancy,
)

OCC_RAW_PATH = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
BOM_RAW_DIR = Path("data/raw/bom")

#: GitHub rejects files over 100 MB; stay well below it.
MAX_FILE_MB = 50


def check_size(path: Path) -> None:
    """Print the size of a file just written."""
    size_mb = path.stat().st_size / 1e6
    print(f"  wrote {path} ({size_mb:.1f} MB)")
    if size_mb > MAX_FILE_MB:
        raise SystemExit(f"{path} is {size_mb:.1f} MB -- too large to commit.")


def main() -> None:
    """Build every processed data file from the raw files and check each reads back identically."""
    # The read-back checks compare values; check_dtype=False only tolerates a
    # different timestamp resolution (microseconds vs nanoseconds), which
    # does not change any value.
    for path in (OCC_RAW_PATH, ENV_RAW_PATH):
        if not path.exists():
            raise SystemExit(f"Raw file not found: {path}. See data/raw/README.md.")

    print("Environmental readings...")
    written = save_env_readings(load_env_sensor_log(ENV_RAW_PATH))
    check_size(ENV_PROCESSED_PATH)
    pd.testing.assert_frame_equal(load_env_readings(), written, check_dtype=False)

    print("CO2 receive times...")
    received = co2_receive_times(ENV_RAW_PATH).reset_index(drop=True)
    save_receive_times(received)
    check_size(RECEIVE_TIMES_PROCESSED_PATH)
    pd.testing.assert_frame_equal(load_receive_times(), received, check_dtype=False)

    print(f"Occupancy, resampled to {PROCESSED_BIN_SIZE} (the raw file is 1.5 GB, this takes a while)...")
    resampled = resample_occupancy(load_occupancy_log(OCC_RAW_PATH), bin_size=PROCESSED_BIN_SIZE)
    save_resampled_occupancy(resampled)
    check_size(OCCUPANCY_PROCESSED_PATH)
    pd.testing.assert_frame_equal(
        load_resampled_occupancy(), resampled.reset_index(drop=True), check_dtype=False
    )

    bom_files = sorted(BOM_RAW_DIR.glob("IDCJDW*.csv"))
    if bom_files:
        print("Bureau of Meteorology CSVs...")
        BOM_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        for f in bom_files:
            shutil.copy2(f, BOM_PROCESSED_DIR / f.name)
        print(f"  copied {len(bom_files)} files to {BOM_PROCESSED_DIR}")
    else:
        print(f"No BoM CSVs in {BOM_RAW_DIR}; run tools/fetch_bom_weather.py first to include them.")

    print("\nDone. Every table reads back identical to the raw-file version.")


if __name__ == "__main__":
    main()
