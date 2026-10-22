"""Unit tests for occupancy.processed_data.

Each table is written and read back on small hand-built inputs, and must
come back with identical values -- otherwise a fresh clone (processed
copies) and a machine with the raw files would feed different data into the
analysis. ``check_dtype=False`` only allows the timestamp resolution
(microseconds vs nanoseconds) to differ, which changes no value.
"""

import json

import pandas as pd
import pytest

from occupancy.env_sensors import load_env_sensor_log
from occupancy.preprocessing import resample_occupancy
from occupancy.processed_data import (
    bom_dir_raw_or_processed,
    co2_receive_times,
    env_readings_raw_or_processed,
    load_env_readings,
    load_receive_times,
    load_resampled_occupancy,
    resampled_occupancy_raw_or_processed,
    save_env_readings,
    save_receive_times,
    save_resampled_occupancy,
)


def _env_payload(time: float, readings: dict[str, float]) -> str:
    return json.dumps(
        [
            {"values": [{"time": time, "value": value}], "variable": {"name": name, "unit": "u"}}
            for name, value in readings.items()
        ]
    )


@pytest.fixture
def raw_env_csv(tmp_path):
    rows = [
        ("6012002000326", 1_700_000_000, {"Carbon dioxide": 450, "Temperature": 21.5, "Pressure": 1013}),
        ("6012002000326", 1_700_000_900.5, {"Carbon dioxide": 455.25, "Humidity": 40}),
        ("6012002000869", 1_700_000_000, {"Carbon dioxide": 600}),
    ]
    path = tmp_path / "env.csv"
    pd.DataFrame(
        {
            "sensorid": [r[0] for r in rows],
            "jsondata": [_env_payload(r[1], r[2]) for r in rows],
            "createdate": ["2023-11-15 09:13:20.123+11", "2023-11-15 09:28:21+11", "2023-11-20 10:00:00+11"],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def occupancy_log():
    times = pd.to_datetime(
        ["2024-03-01 08:00", "2024-03-01 08:40", "2024-03-01 13:00", "2024-03-01 08:05"]
    ).tz_localize("Australia/Melbourne")
    return pd.DataFrame(
        {
            "floorspaceid": ["room-a", "room-a", "room-a", "room-b"],
            "collecteddate": times,
            "occupancystatus": ["CurrentlyOccupied", "NotOccupied", "CurrentlyOccupied", "RecentlyOccupied"],
        }
    )


def test_env_readings_round_trip_exactly_and_keep_only_analysed_variables(raw_env_csv, tmp_path):
    written = save_env_readings(load_env_sensor_log(raw_env_csv), tmp_path / "env.csv.gz")
    assert set(written["variable"]) == {"Carbon dioxide", "Temperature", "Humidity"}
    pd.testing.assert_frame_equal(load_env_readings(tmp_path / "env.csv.gz"), written, check_dtype=False)


def test_env_fallback_matches_raw_path(raw_env_csv, tmp_path):
    from_raw = env_readings_raw_or_processed(raw_env_csv)
    save_env_readings(load_env_sensor_log(raw_env_csv), tmp_path / "env.csv.gz")
    from_processed = env_readings_raw_or_processed(tmp_path / "missing.csv", tmp_path / "env.csv.gz")
    pd.testing.assert_frame_equal(from_processed, from_raw, check_dtype=False)


def test_env_fallback_raises_when_neither_file_exists(tmp_path):
    with pytest.raises(FileNotFoundError):
        env_readings_raw_or_processed(tmp_path / "missing.csv", tmp_path / "missing.csv.gz")


def test_resampled_occupancy_round_trips_exactly_including_gaps(occupancy_log, tmp_path):
    resampled = resample_occupancy(occupancy_log, bin_size="15min")
    assert resampled["occupied"].isna().any()  # the 08:40 -> 13:00 gap exceeds max_gap
    save_resampled_occupancy(resampled, tmp_path / "occ.csv.gz")
    pd.testing.assert_frame_equal(load_resampled_occupancy(tmp_path / "occ.csv.gz"), resampled, check_dtype=False)


def test_occupancy_fallback_only_serves_the_stored_bin_size(occupancy_log, tmp_path):
    save_resampled_occupancy(resample_occupancy(occupancy_log, bin_size="15min"), tmp_path / "occ.csv.gz")
    missing = tmp_path / "missing.csv"
    assert len(resampled_occupancy_raw_or_processed(missing, "15min", tmp_path / "occ.csv.gz")) > 0
    with pytest.raises(FileNotFoundError):
        resampled_occupancy_raw_or_processed(missing, "30min", tmp_path / "occ.csv.gz")


def test_receive_times_round_trip_exactly_with_subsecond_times(raw_env_csv, tmp_path):
    received = co2_receive_times(raw_env_csv).reset_index(drop=True)
    assert len(received) == 3
    save_receive_times(received, tmp_path / "rt.csv.gz")
    pd.testing.assert_frame_equal(load_receive_times(tmp_path / "rt.csv.gz"), received, check_dtype=False)


def test_bom_dir_prefers_raw_files_when_present(tmp_path):
    raw_dir, processed_dir = tmp_path / "raw", tmp_path / "processed"
    raw_dir.mkdir()
    assert bom_dir_raw_or_processed(raw_dir, processed_dir) == processed_dir
    (raw_dir / "IDCJDW3049.202501.csv").write_text("x")
    assert bom_dir_raw_or_processed(raw_dir, processed_dir) == raw_dir


def test_committed_processed_data_is_present_and_complete():
    """Guard: a fresh clone must contain the data every experiment script needs."""
    from pathlib import Path

    from occupancy.processed_data import (
        BOM_PROCESSED_DIR,
        ENV_PROCESSED_PATH,
        OCCUPANCY_PROCESSED_PATH,
        RECEIVE_TIMES_PROCESSED_PATH,
        VARIABLES_KEPT,
    )

    root = Path(__file__).resolve().parents[1]
    env = load_env_readings(root / ENV_PROCESSED_PATH)
    assert {"6012002000326", "6012002000869"} <= set(env["sensorid"])
    assert set(env["variable"]) == set(VARIABLES_KEPT)
    assert load_resampled_occupancy(root / OCCUPANCY_PROCESSED_PATH)["floorspaceid"].nunique() == 5
    assert len(load_receive_times(root / RECEIVE_TIMES_PROCESSED_PATH)) > 0
    assert any((root / BOM_PROCESSED_DIR).glob("IDCJDW*.csv"))
