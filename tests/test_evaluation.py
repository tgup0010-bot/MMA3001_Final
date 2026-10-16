"""Unit tests for occupancy.evaluation."""

import pandas as pd
import pytest

from occupancy.evaluation import chronological_split, evaluate_regression_predictions


def test_chronological_split_respects_train_frac_and_time_order():
    times = pd.date_range("2024-01-01", periods=10, freq="D")
    table = pd.DataFrame({"collecteddate": times, "value": range(10)})

    train, test, cutoff = chronological_split(table, time_col="collecteddate", train_frac=0.7)

    assert len(train) == 7
    assert len(test) == 3
    assert train["collecteddate"].max() < test["collecteddate"].min() or train["collecteddate"].max() == cutoff
    # No overlap, and every training timestamp precedes every test timestamp.
    assert train["collecteddate"].max() <= test["collecteddate"].min()


@pytest.mark.parametrize("bad_frac", [0.0, 1.0, -0.1, 1.5])
def test_chronological_split_rejects_invalid_train_frac(bad_frac):
    table = pd.DataFrame({"collecteddate": pd.date_range("2024-01-01", periods=5, freq="D")})
    with pytest.raises(ValueError):
        chronological_split(table, time_col="collecteddate", train_frac=bad_frac)


def test_evaluate_regression_predictions_perfect_predictions():
    metrics = evaluate_regression_predictions([400, 450, 500], [400, 450, 500])
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["n"] == 3


def test_evaluate_regression_predictions_hand_computed_values():
    # Errors are +10, -10, +20 ppm: MAE = 40/3, RMSE = sqrt(600/3) = sqrt(200).
    y_true = [400.0, 450.0, 500.0]
    y_pred = [410.0, 440.0, 520.0]
    metrics = evaluate_regression_predictions(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(40 / 3)
    assert metrics["rmse"] == pytest.approx(200 ** 0.5)
    # R2 = 1 - SS_res / SS_tot = 1 - 600 / 5000.
    assert metrics["r2"] == pytest.approx(1 - 600 / 5000)


def test_evaluate_regression_predictions_mean_forecast_gives_zero_r2():
    y_true = [400.0, 450.0, 500.0]
    metrics = evaluate_regression_predictions(y_true, [450.0, 450.0, 450.0])
    assert metrics["r2"] == pytest.approx(0.0)
