"""Predict CO2 concentration from its own history and building occupancy.

The project's primary computational experiment: compares the four
regression methods taught in MMA3001 Week 5 (Linear Regression, Decision
Tree Regression, SVR, Neural Network Regression) on a genuine regression
task -- predicting sensor 6012002000326's CO2 concentration 15 minutes
ahead -- and tests, for each method, whether adding building-wide
occupancy as a feature actually improves the prediction over CO2's own
recent trend alone.

Usage:
    python scripts/co2_prediction_experiment.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from occupancy.co2_models import (
    build_decision_tree_regression_pipeline,
    build_linear_regression_pipeline,
    build_neural_network_regression_pipeline,
    build_svr_pipeline,
)
from occupancy.co2_regression import (
    NUMERIC_FEATURES_WITH_OCCUPANCY,
    NUMERIC_FEATURES_WITHOUT_OCCUPANCY,
    build_co2_supervised_dataset,
    building_occupancy_series,
    co2_series,
)
from occupancy.evaluation import chronological_split, evaluate_regression_predictions
from occupancy.processed_data import (
    env_readings_raw_or_processed,
    resampled_occupancy_raw_or_processed,
)

OCC_RAW_PATH = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
RESULTS_PATH = Path("reports/co2_prediction_results.json")

SENSOR_ID = "6012002000326"
BIN_SIZE = "15min"

MODEL_BUILDERS = {
    "Linear Regression": build_linear_regression_pipeline,
    "Decision Tree Regression": build_decision_tree_regression_pipeline,
    "SVR": build_svr_pipeline,
    "Neural Network Regression": build_neural_network_regression_pipeline,
}


def timed(label: str, fn, *args, **kwargs):
    """Call ``fn`` and print how long it took; returns ``(result, seconds)``."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[{elapsed:6.1f}s] {label}")
    return result, elapsed


def run_feature_set(table: pd.DataFrame, feature_cols: list[str], label: str) -> dict:
    """Fit all four models on one feature set and score them on the test period.

    Args:
        table: Supervised table from ``build_co2_supervised_dataset``.
        feature_cols: Input columns to give the models.
        label: Name of the feature set, for printing.

    Returns:
        Test metrics (MAE, RMSE, R2) and fit/predict times per model.
    """
    # Chronological, not random: the model may only learn from the past, and a
    # random split would put near-identical neighbouring rows in both sets.
    train, test, cutoff = chronological_split(table, time_col="time", train_frac=0.8)
    print(f"\n--- Feature set: {label} ({len(feature_cols)} features) ---")
    print(f"cutoff={cutoff}  train={len(train):,}  test={len(test):,}")

    results = {}
    for name, builder in MODEL_BUILDERS.items():
        # Original (hand-set) settings; tuning is done separately in
        # optimisation_and_performance.py, on the training period only.
        pipeline = builder()
        _, fit_t = timed(f"fit {name}", pipeline.fit, train[feature_cols], train["target"])
        start = time.perf_counter()
        pred = pipeline.predict(test[feature_cols])
        predict_t = time.perf_counter() - start
        metrics = evaluate_regression_predictions(test["target"], pred)
        metrics["fit_seconds"] = fit_t
        metrics["predict_seconds"] = predict_t
        results[name] = metrics
        print(
            f"  {name:28} MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
            f"R2={metrics['r2']:.4f}"
        )
    return results


def main() -> None:
    """Run the baseline comparison with and without occupancy and save the results."""
    env_long, t = timed("load env readings", env_readings_raw_or_processed, ENV_RAW_PATH)
    occ_resampled, t = timed(
        "load + resample occupancy", resampled_occupancy_raw_or_processed, OCC_RAW_PATH, BIN_SIZE
    )
    # Both series on the same 15-minute grid, so each row lines up in time.
    occupancy = building_occupancy_series(occ_resampled)
    co2 = co2_series(env_long, SENSOR_ID, BIN_SIZE)

    # rolling_window=4 bins = the last hour; rows with any gap are dropped.
    table, info = build_co2_supervised_dataset(co2, occupancy, rolling_window=4)
    print(f"\nsupervised table: {info}")

    # RQ2: the same rows with and without the occupancy feature, so any
    # difference in error is due to occupancy alone.
    with_occ = run_feature_set(table, list(NUMERIC_FEATURES_WITH_OCCUPANCY), "with occupancy")
    without_occ = run_feature_set(
        table, list(NUMERIC_FEATURES_WITHOUT_OCCUPANCY), "without occupancy"
    )

    print("\n=== Does building occupancy improve CO2 prediction? ===")
    for name in MODEL_BUILDERS:
        r2_with = with_occ[name]["r2"]
        r2_without = without_occ[name]["r2"]
        delta = r2_with - r2_without
        # Changes under 0.001 in R2 are treated as no difference (noise level).
        verdict = "helps" if delta > 0.001 else ("hurts" if delta < -0.001 else "no difference")
        print(f"  {name:28} R2 with={r2_with:+.4f}  without={r2_without:+.4f}  delta={delta:+.4f}  ({verdict})")

    # Lowest test MAE among the original settings. This choice used the test
    # set; the final model is selected by cross-validation instead (report 5.4).
    best_model = min(with_occ, key=lambda n: with_occ[n]["mae"])
    print(f"\nBest model (lowest MAE, with occupancy): {best_model}")

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "sensor_id": SENSOR_ID,
                "bin_size": BIN_SIZE,
                "data_info": info,
                "results_with_occupancy": with_occ,
                "results_without_occupancy": without_occ,
                "best_model": best_model,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
