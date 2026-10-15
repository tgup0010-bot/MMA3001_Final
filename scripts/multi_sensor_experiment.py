"""Use every usable interval from every sensor, not just one sensor.

Follows up on teaching-team feedback: instead of accepting or rejecting a
sensor as a whole, cut each sensor's CO2 record into unbroken 15-minute
intervals and use every interval long enough to build a supervised row
(1-hour history + 15-minute-ahead target = 75 minutes).

Steps:

  1. Interval audit of all five sensors (``occupancy.sensor_intervals``),
     plus a check of the log's ``createdate`` (server receive time)
     against each reading's own timestamp, which shows that the
     "unusable" sensors were re-transmitting one stale packet rather than
     producing new readings.
  2. Experiment A -- pooled training: train on sensor 0326's training
     period *plus* every usable 0869 interval, test on 0326's unchanged
     test period. Does more data help?
  3. Experiment B -- cross-sensor transfer: train on 0326 only, test on
     sensor 0869. Does a model learned in one room work in another?

Occupancy is excluded throughout: Section 4 of the report shows it adds
nothing for sensor 0326, and leaving it out keeps the row sets identical
across experiments. A naive persistence forecast (next CO2 = current CO2)
is reported alongside the four Week 5 models as a reference floor.

Writes ``reports/multi_sensor_results.json`` and two charts
(``docs/report/chart15_sensor_intervals.png``,
``docs/report/chart16_multi_sensor_results.png``).

Usage:
    python scripts/multi_sensor_experiment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from occupancy.co2_models import (
    build_decision_tree_regression_pipeline,
    build_linear_regression_pipeline,
    build_neural_network_regression_pipeline,
    build_svr_pipeline,
)
from occupancy.co2_regression import (
    NUMERIC_FEATURES_WITHOUT_OCCUPANCY,
    build_co2_supervised_dataset,
    co2_series,
)
from occupancy.env_sensors import LOCAL_TZ, load_env_sensor_log
from occupancy.evaluation import chronological_split, evaluate_regression_predictions
from occupancy.sensor_intervals import find_unbroken_intervals, interval_audit

ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
RESULTS_PATH = Path("reports/multi_sensor_results.json")
OUT_DIR = Path("docs/report")

MAIN_SENSOR = "6012002000326"
SECOND_SENSOR = "6012002000869"
BIN_SIZE = "15min"
ROLLING_WINDOW = 4
#: A reading received more than this long after its own timestamp is a
#: re-transmitted stale packet, not a new measurement.
STALE_AFTER = pd.Timedelta("1D")
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_BLOCK = 96  # one day of 15-minute bins (errors are autocorrelated)

MODEL_BUILDERS = {
    "Linear Regression": build_linear_regression_pipeline,
    "Decision Tree Regression": build_decision_tree_regression_pipeline,
    "SVR": build_svr_pipeline,
    "Neural Network Regression": build_neural_network_regression_pipeline,
}
FEATURES = list(NUMERIC_FEATURES_WITHOUT_OCCUPANCY)

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
USABLE_COLOR = "#1baf7a"
STALE_COLOR = "#b0aea5"
BASE_COLOR = "#2a78d6"
ALT_COLOR = "#eb6834"


def receive_times(path: Path) -> pd.DataFrame:
    """Each CO2 reading's own timestamp alongside its server receive time."""
    raw = pd.read_csv(path, usecols=["sensorid", "jsondata", "createdate"], dtype={"sensorid": str})

    def co2_time(payload: str):
        try:
            for item in json.loads(payload):
                if item["variable"]["name"] == "Carbon dioxide":
                    return item["values"][0]["time"]
        except (TypeError, ValueError, KeyError, IndexError):
            pass
        return None

    out = pd.DataFrame(
        {
            "sensorid": raw["sensorid"],
            "reading_time": pd.to_datetime(raw["jsondata"].map(co2_time), unit="s", utc=True),
            "received": pd.to_datetime(raw["createdate"], utc=True, format="mixed"),
        }
    ).dropna()
    out["stale"] = (out["received"] - out["reading_time"]) > STALE_AFTER
    return out


def supervised_table(env_long: pd.DataFrame, sensorid: str) -> tuple[pd.DataFrame, dict]:
    co2 = co2_series(env_long, sensorid, BIN_SIZE)
    # Occupancy is not used here; a placeholder keeps the shared builder's signature.
    placeholder = pd.Series(0.0, index=co2.index, name="building_occupancy")
    table, info = build_co2_supervised_dataset(co2, placeholder, rolling_window=ROLLING_WINDOW)
    table["sensor"] = sensorid
    return table, info


def run(label: str, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    results = {"Persistence": evaluate_regression_predictions(test["target"], test["co2_now"])}
    predictions = {}
    for name, builder in MODEL_BUILDERS.items():
        pred = builder().fit(train[FEATURES], train["target"]).predict(test[FEATURES])
        results[name] = evaluate_regression_predictions(test["target"], pred)
        predictions[name] = pred
    print(f"\n### {label}  (train {len(train):,} / test {len(test):,})")
    for name, m in results.items():
        print(f"  {name:28} MAE={m['mae']:6.3f}  RMSE={m['rmse']:6.3f}  R2={m['r2']:+.4f}")
    return {"n_train": len(train), "n_test": len(test), "results": results}, predictions


def block_bootstrap_ci(diff: np.ndarray, seed: int = 0) -> tuple[float, float]:
    """95% moving-block bootstrap CI for the mean of a paired difference."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    starts = np.arange(n - BOOTSTRAP_BLOCK)
    means = [
        diff[np.concatenate([np.arange(s, s + BOOTSTRAP_BLOCK) for s in rng.choice(starts, n // BOOTSTRAP_BLOCK)])].mean()
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def chart_sensor_intervals(env_long: pd.DataFrame, received: pd.DataFrame, sensors: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for row, sid in enumerate(sensors):
        intervals = find_unbroken_intervals(co2_series(env_long, sid, BIN_SIZE))
        intervals = intervals[intervals["n_bins"] > ROLLING_WINDOW]
        ax.broken_barh(
            [(s, e - s + pd.Timedelta(BIN_SIZE)) for s, e in zip(intervals["start"], intervals["end"])],
            (row - 0.35, 0.7), color=USABLE_COLOR,
        )
        stale = received[(received["sensorid"] == sid) & received["stale"]]["received"].dt.tz_convert(LOCAL_TZ)
        ax.scatter(stale, np.full(len(stale), row), s=2, color=STALE_COLOR, marker="|", alpha=0.5)
    ax.set_yticks(range(len(sensors)), [s[-4:] for s in sensors])
    ax.set_ylabel("Sensor (last 4 digits)")
    ax.set_title("Usable unbroken intervals (green) vs. re-transmitted stale packets (grey), by sensor")
    ax.scatter([], [], color=STALE_COLOR, marker="s", label="Stale packet re-sent (no new reading)")
    ax.broken_barh([], (0, 0), color=USABLE_COLOR, label="Usable interval (>= 75 min unbroken)")
    ax.set_ylim(-0.6, len(sensors) - 0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#c3c2b7")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart15_sensor_intervals.png", dpi=150)
    plt.close(fig)


def chart_results(results: dict) -> None:
    models = ["Persistence", *MODEL_BUILDERS]
    short = ["Persistence", "Linear", "Decision Tree", "SVR", "Neural Net"]
    panels = [
        ("Test on sensor 0326 (Mar-Apr 2026)", "baseline_0326", "Train 0326 only", "pooled_0326", "Train 0326 + 0869"),
        ("Test on sensor 0869 (Jun 2024)", "own_0869", "Train 0869 (same room)", "cross_0869", "Train 0326 (other room)"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(models))
    for ax, (title, key_a, lab_a, key_b, lab_b) in zip(axes, panels):
        a = [results[key_a]["results"][m]["mae"] for m in models]
        b = [results[key_b]["results"][m]["mae"] for m in models]
        ax.bar(x - 0.2, a, 0.4, color=BASE_COLOR, label=lab_a)
        ax.bar(x + 0.2, b, 0.4, color=ALT_COLOR, label=lab_b)
        for xi, (va, vb) in enumerate(zip(a, b)):
            ax.text(xi - 0.2, va + 0.15, f"{va:.2f}", ha="center", fontsize=8)
            ax.text(xi + 0.2, vb + 0.15, f"{vb:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x, short)
        ax.set_ylabel("MAE (ppm, lower is better)")
        ax.set_title(title)
        ax.set_ylim(0, max(a + b) * 1.25)
        ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5, color="#c3c2b7")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart16_multi_sensor_results.png", dpi=150)
    plt.close(fig)


def main() -> None:
    env_long = load_env_sensor_log(ENV_RAW_PATH)
    received = receive_times(ENV_RAW_PATH)
    sensors = sorted(env_long["sensorid"].unique())

    audit = interval_audit(env_long, sensors, bin_size=BIN_SIZE, rolling_window=ROLLING_WINDOW)
    stale = received.groupby("sensorid")["stale"].sum().rename("stale_rows")
    audit = audit.merge(stale, left_on="sensorid", right_index=True, how="left")
    print("=== Interval audit ===")
    print(audit.to_string(index=False))

    t326, info326 = supervised_table(env_long, MAIN_SENSOR)
    t869, info869 = supervised_table(env_long, SECOND_SENSOR)
    tr326, te326, cutoff = chronological_split(t326, time_col="time", train_frac=0.8)
    tr869, te869, cutoff869 = chronological_split(t869, time_col="time", train_frac=0.8)
    pooled = pd.concat([tr326, t869], ignore_index=True)  # all of 0869 (2024) predates the 2026 cutoff

    results, preds = {}, {}
    results["baseline_0326"], preds["baseline"] = run("Baseline: train 0326 -> test 0326", tr326, te326)
    results["pooled_0326"], preds["pooled"] = run("A: train 0326 + 0869 -> test 0326", pooled, te326)
    results["own_0869"], _ = run("Reference: train 0869 -> test 0869 (last 20%)", tr869, te869)
    results["cross_0869"], _ = run("B: train 0326 -> test 0869 (last 20%)", tr326, te869)
    results["cross_0869_all"], _ = run("B (all 0869): train 0326 -> test all of 0869", tr326, t869)

    print("\n=== Pooled vs baseline on 0326: MAE improvement, 95% block-bootstrap CI ===")
    significance = {}
    for name in MODEL_BUILDERS:
        diff = np.abs(preds["baseline"][name] - te326["target"].values) - np.abs(
            preds["pooled"][name] - te326["target"].values
        )
        lo, hi = block_bootstrap_ci(diff)
        significance[name] = {"mae_improvement": float(diff.mean()), "ci95": [lo, hi]}
        print(f"  {name:28} {diff.mean():+.3f} ppm  [{lo:+.3f}, {hi:+.3f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_sensor_intervals(env_long, received, sensors)
    chart_results(results)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "bin_size": BIN_SIZE,
                "features": FEATURES,
                "interval_audit": audit.to_dict(orient="records"),
                "supervised_rows": {MAIN_SENSOR: info326, SECOND_SENSOR: info869},
                "cutoffs": {MAIN_SENSOR: cutoff, SECOND_SENSOR: cutoff869},
                "experiments": results,
                "pooled_vs_baseline_bootstrap": significance,
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nSaved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
