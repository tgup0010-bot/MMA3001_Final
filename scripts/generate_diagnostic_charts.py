"""Generate per-model diagnostic charts for the CO2 regression comparison.

Fills a real gap: the four models compared in the report and in
``reports/co2_prediction_results.json`` only ever had their aggregate
metrics (MAE/RMSE/R2) saved -- never the row-level test-set predictions,
and no script in this repo ever produced a predicted-vs-actual or
residual plot (the diagnostics MMA3001 Week 5.5 teaches). This script
re-runs the exact same pipeline as ``co2_prediction_experiment.py`` (same
feature set, same chronological split, same model configs), captures the
row-level test predictions this time, and produces:

  1. ``chart11_predictions_timeseries.png`` -- ground truth CO2 vs all
     four models' predictions, one panel for the full test period
     (daily-mean, so 3,402 rows are legible) and one zoomed panel at
     native 15-minute resolution for a representative 5-day window.
  2. ``chart12_predicted_vs_actual.png`` -- small-multiples scatter
     (one panel per model) of predicted vs. actual CO2 with a y=x
     reference line, so all four are directly comparable at a glance.
  3. ``chart13_residuals.png`` -- small-multiples residual plot
     (predicted - actual vs. predicted), the standard Week 5.5
     diagnostic for spotting systematic bias.
  4. ``chart14_train_test_split.png`` -- the train/test split over the
     full sensor history.

Also writes ``reports/co2_test_predictions.csv`` (time, actual, and each
model's prediction for every test-set row) so these -- or any future
diagnostic -- can be rebuilt without refitting.

Usage:
    python scripts/generate_diagnostic_charts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from occupancy.co2_models import (
    build_decision_tree_regression_pipeline,
    build_linear_regression_pipeline,
    build_neural_network_regression_pipeline,
    build_svr_pipeline,
)
from occupancy.co2_regression import (
    NUMERIC_FEATURES_WITH_OCCUPANCY,
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
OUT_DIR = Path("doc/report")
PRED_CSV_PATH = Path("reports/co2_test_predictions.csv")
SENSOR_ID = "6012002000326"
BIN_SIZE = "15min"

# Fixed model -> (color, marker) so identity never depends on color alone.
# Colors are the dataviz skill's validated categorical slots 1/2/3/7
# (blue/orange/aqua/violet) rather than the adjacent 1-4 run, since a
# scatter needs all-pairs separation, not just adjacent-pair separation,
# and slot 4 (yellow) sits too close to slot 2 (orange) for that.
MODEL_STYLE = {
    "Linear Regression": {"color": "#2a78d6", "marker": "o"},
    "Decision Tree Regression": {"color": "#eb6834", "marker": "s"},
    "SVR": {"color": "#1baf7a", "marker": "^"},
    "Neural Network Regression": {"color": "#4a3aa7", "marker": "D"},
}
TRUTH_COLOR = "#0b0b0b"
GRID_KW = dict(axis="y", linestyle="--", alpha=0.5, color="#c3c2b7")

MODEL_BUILDERS = {
    "Linear Regression": build_linear_regression_pipeline,
    "Decision Tree Regression": build_decision_tree_regression_pipeline,
    "SVR": build_svr_pipeline,
    "Neural Network Regression": build_neural_network_regression_pipeline,
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#898781",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)


def load_supervised_table() -> pd.DataFrame:
    env_long = env_readings_raw_or_processed(ENV_RAW_PATH)
    occ_resampled = resampled_occupancy_raw_or_processed(OCC_RAW_PATH, BIN_SIZE)
    occupancy = building_occupancy_series(occ_resampled)
    co2 = co2_series(env_long, SENSOR_ID, BIN_SIZE)
    table, info = build_co2_supervised_dataset(co2, occupancy, rolling_window=4)
    print(f"supervised table: {info}")
    return table, co2


def fit_and_predict(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    feature_cols = list(NUMERIC_FEATURES_WITH_OCCUPANCY)
    train, test, cutoff = chronological_split(table, time_col="time", train_frac=0.8)
    print(f"cutoff={cutoff}  train={len(train):,}  test={len(test):,}")

    preds = pd.DataFrame({"time": test["time"].values, "actual": test["target"].values})
    for name, builder in MODEL_BUILDERS.items():
        pipeline = builder()
        pipeline.fit(train[feature_cols], train["target"])
        pred = pipeline.predict(test[feature_cols])
        preds[name] = pred
        metrics = evaluate_regression_predictions(test["target"], pred)
        print(f"  {name:28} MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  R2={metrics['r2']:.4f}")
    return train, preds, cutoff


def chart_predictions_timeseries(preds: pd.DataFrame) -> None:
    preds = preds.set_index(pd.to_datetime(preds["time"]))
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[1, 1.2])

    # Top: full test period, daily mean -- readable across all 3,402 rows.
    daily = preds.resample("1D").mean(numeric_only=True)
    ax_top.plot(daily.index, daily["actual"], color=TRUTH_COLOR, linewidth=2.2, label="Ground truth (CO2)")
    for name, style in MODEL_STYLE.items():
        ax_top.plot(daily.index, daily[name], color=style["color"], linewidth=1.4, alpha=0.9, label=name)
    ax_top.set_title("Full test set (daily mean) — ground truth vs. all four models")
    ax_top.set_ylabel("CO2 (ppm)")
    ax_top.grid(**GRID_KW)
    ax_top.legend(loc="upper right", fontsize=9, ncol=1, framealpha=0.9)
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    # Bottom: a representative 5-day window at native 15-min resolution.
    window_start = preds.index.min() + pd.Timedelta(days=3)
    window_end = window_start + pd.Timedelta(days=5)
    window = preds.loc[window_start:window_end]
    ax_bot.plot(window.index, window["actual"], color=TRUTH_COLOR, linewidth=2.0, label="Ground truth (CO2)")
    for name, style in MODEL_STYLE.items():
        ax_bot.plot(
            window.index, window[name], color=style["color"], linewidth=1.2,
            marker=style["marker"], markersize=3, markevery=8, alpha=0.85, label=name,
        )
    ax_bot.set_title(f"Detail: {window_start.date()} to {window_end.date()} (15-min resolution)")
    ax_bot.set_ylabel("CO2 (ppm)")
    ax_bot.set_xlabel("Date")
    ax_bot.grid(**GRID_KW)
    ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
    # Rotate each panel's labels itself: fig.autofmt_xdate() hides the top
    # panel's dates when panels are stacked.
    for ax in (ax_top, ax_bot):
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Predicted vs. actual CO2 — all four Week 5 regression models", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "chart11_predictions_timeseries.png", dpi=150)
    plt.close(fig)


def chart_predicted_vs_actual(preds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharex=True, sharey=True)
    lo = min(preds["actual"].min(), *(preds[m].min() for m in MODEL_STYLE))
    hi = max(preds["actual"].max(), *(preds[m].max() for m in MODEL_STYLE))
    pad = (hi - lo) * 0.03

    for ax, (name, style) in zip(axes.flat, MODEL_STYLE.items()):
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#898781", linestyle="--", linewidth=1.2, label="y = x (perfect)")
        ax.scatter(
            preds["actual"], preds[name], s=14, alpha=0.35, color=style["color"],
            marker=style["marker"], edgecolors="none",
        )
        metrics = evaluate_regression_predictions(preds["actual"], preds[name])
        ax.set_title(f"{name}\nMAE={metrics['mae']:.2f}  R2={metrics['r2']:.3f}")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.grid(axis="both", linestyle="--", alpha=0.4, color="#c3c2b7")
        ax.set_aspect("equal", adjustable="box")

    for ax in axes:
        ax.set_xlabel("Actual CO2 (ppm)")
    axes[0].set_ylabel("Predicted CO2 (ppm)")

    fig.suptitle("Predicted vs. actual CO2 on the held-out test set (3,402 rows)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "chart12_predicted_vs_actual.png", dpi=150)
    plt.close(fig)


def chart_residuals(preds: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)

    for ax, (name, style) in zip(axes.flat, MODEL_STYLE.items()):
        residual = preds[name] - preds["actual"]
        ax.axhline(0, color="#898781", linestyle="--", linewidth=1.2)
        ax.scatter(
            preds[name], residual, s=14, alpha=0.35, color=style["color"],
            marker=style["marker"], edgecolors="none",
        )
        ax.set_title(f"{name}\nmean={residual.mean():+.2f} ppm, std={residual.std():.2f}")
        ax.grid(axis="both", linestyle="--", alpha=0.4, color="#c3c2b7")

    for ax in axes:
        ax.set_xlabel("Predicted CO2 (ppm)")
    axes[0].set_ylabel("Residual (ppm)")

    fig.suptitle("Residual diagnostics — predicted CO2 vs. prediction error (predicted − actual)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT_DIR / "chart13_residuals.png", dpi=150)
    plt.close(fig)


def chart_train_test_split(table: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    """Rows actually used per week, so the 636-day gap in genuine data is visible."""
    fig, ax = plt.subplots(figsize=(13, 3.2))
    times = pd.to_datetime(table["time"])
    weekly_train = times[times < cutoff].dt.to_period("W").dt.start_time.value_counts().sort_index()
    weekly_test = times[times >= cutoff].dt.to_period("W").dt.start_time.value_counts().sort_index()
    ax.bar(weekly_train.index, weekly_train.values, width=6, align="edge",
           color=MODEL_STYLE["Linear Regression"]["color"], label=f"Training rows ({int(weekly_train.sum()):,})")
    ax.bar(weekly_test.index, weekly_test.values, width=6, align="edge",
           color=MODEL_STYLE["Decision Tree Regression"]["color"], label=f"Test rows ({int(weekly_test.sum()):,})")
    ax.axvline(cutoff.tz_localize(None) if cutoff.tzinfo else cutoff, color="#0b0b0b", linestyle="--", linewidth=1.5)
    ax.set_ylabel("Rows per week")
    ax.set_title("Chronological train / test split, sensor 6012002000326: rows used per week (cutoff 11 March 2026)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart14_train_test_split.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    table, co2 = load_supervised_table()
    train, preds, cutoff = fit_and_predict(table)

    preds.to_csv(PRED_CSV_PATH, index=False)
    print(f"Saved row-level test predictions -> {PRED_CSV_PATH}")

    chart_predictions_timeseries(preds)
    chart_predicted_vs_actual(preds)
    chart_residuals(preds)
    chart_train_test_split(table, cutoff)
    print(f"Saved 4 charts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
