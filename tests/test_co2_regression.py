"""Unit tests for occupancy.co2_regression.

Uses small hand-built series/tables (not the real large files) so the
suite runs in well under a second and isolates the aggregation and
feature-building logic.
"""

import pandas as pd
import pytest

from occupancy.co2_regression import (
    NUMERIC_FEATURES_WITH_OCCUPANCY,
    NUMERIC_FEATURES_WITHOUT_OCCUPANCY,
    build_co2_supervised_dataset,
    building_occupancy_series,
)


def test_building_occupancy_series_sums_across_rooms():
    idx = pd.date_range("2024-06-01 09:00", periods=3, freq="15min", tz="Australia/Melbourne")
    occ_resampled = pd.DataFrame(
        {
            "floorspaceid": ["a"] * 3 + ["b"] * 3,
            "collecteddate": list(idx) * 2,
            "occupied": [1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        }
    )
    total = building_occupancy_series(occ_resampled)
    assert total.loc[idx[0]] == 2.0  # both rooms occupied
    assert total.loc[idx[1]] == 1.0  # only room b
    assert total.loc[idx[2]] == 1.0  # only room a


def test_building_occupancy_series_leaves_fully_missing_bin_as_nan():
    idx = pd.date_range("2024-06-01 09:00", periods=2, freq="15min", tz="Australia/Melbourne")
    occ_resampled = pd.DataFrame(
        {
            "floorspaceid": ["a", "a"],
            "collecteddate": list(idx),
            "occupied": [1.0, float("nan")],
        }
    )
    total = building_occupancy_series(occ_resampled)
    assert total.loc[idx[0]] == 1.0
    assert pd.isna(total.loc[idx[1]])


def test_build_co2_supervised_dataset_produces_expected_columns_and_target():
    idx = pd.date_range("2024-06-01 09:00", periods=6, freq="15min", tz="Australia/Melbourne")
    co2 = pd.Series([600.0, 610.0, 620.0, 630.0, 640.0, 650.0], index=idx, name="co2")
    occupancy = pd.Series([1, 1, 2, 2, 1, 0], index=idx, name="building_occupancy")

    table, info = build_co2_supervised_dataset(co2, occupancy, rolling_window=2)

    assert set(NUMERIC_FEATURES_WITH_OCCUPANCY).issubset(table.columns)
    assert set(NUMERIC_FEATURES_WITHOUT_OCCUPANCY).issubset(table.columns)
    assert "building_occupancy" not in NUMERIC_FEATURES_WITHOUT_OCCUPANCY
    # target at row i should equal co2_now at row i+1 (next bin's CO2)
    row0 = table.iloc[0]
    row1 = table.iloc[1]
    assert row0["target"] == pytest.approx(row1["co2_now"])
    assert info["rows_before"] >= info["rows_after"]


def test_build_co2_supervised_dataset_raises_on_no_overlap():
    idx1 = pd.date_range("2024-01-01", periods=3, freq="15min", tz="Australia/Melbourne")
    idx2 = pd.date_range("2025-01-01", periods=3, freq="15min", tz="Australia/Melbourne")
    co2 = pd.Series([1.0, 2.0, 3.0], index=idx1)
    occupancy = pd.Series([1.0, 2.0, 3.0], index=idx2)
    with pytest.raises(ValueError):
        build_co2_supervised_dataset(co2, occupancy)


def test_selected_pipeline_uses_the_tuned_svr_settings():
    from occupancy.co2_models import SELECTED_SVR_PARAMS, build_selected_pipeline

    svr = build_selected_pipeline()[-1]
    assert {k: svr.get_params()[k] for k in SELECTED_SVR_PARAMS} == SELECTED_SVR_PARAMS
