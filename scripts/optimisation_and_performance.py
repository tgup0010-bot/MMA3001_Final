"""Optimisation and performance analysis (report Section 6).

Answers the questions the original four-model comparison left open:

1. **Model selection without the test set.** Each method's settings are
   tuned by 5-fold time-series cross-validation on the training period
   only, and the final model is chosen by cross-validated MAE. The test
   period is used once, at the end, to report the selected design.
2. **Baseline vs improved.** Original settings vs tuned settings for every
   method, against the persistence forecast (next CO2 = current CO2), with
   moving-block bootstrap intervals on the paired error differences.
3. **Cost.** Repeated timing, peak memory, model size, a cProfile run of
   the whole workflow, estimated FLOPs per prediction and the SVR's
   arithmetic intensity.
4. **Sensitivity and robustness.** Input resolution (bin size), training
   data volume (learning curve), MLP seed repeatability, per-fold scores
   over time, the integration gap threshold, and outlier removal.
5. **Physical check.** Ventilation rate estimated from the decay of CO2
   releases (``occupancy.data_quality``).

Writes ``reports/optimisation_results.json`` and
``doc/report/chart17_optimisation.png``.

Usage:
    python scripts/optimisation_and_performance.py               # about 8 minutes
    python scripts/optimisation_and_performance.py --chart-only  # redraw the chart
"""

from __future__ import annotations

import cProfile
import json
import pstats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from occupancy.co2_integration import trapezoidal_integral_gap_aware
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
from occupancy.data_quality import hampel_flags, spike_decay_rates
from occupancy.env_sensors import pivot_variable
from occupancy.evaluation import block_bootstrap_ci, chronological_split, evaluate_regression_predictions
from occupancy.performance import (
    median_time,
    model_size_kb,
    peak_memory_mb,
    prediction_flops,
    svr_arithmetic_intensity,
)
from occupancy.processed_data import env_readings_raw_or_processed, resampled_occupancy_raw_or_processed

OCC_RAW_PATH = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
RESULTS_PATH = Path("reports/optimisation_results.json")
CHART_PATH = Path("doc/report/chart17_optimisation.png")

SENSOR_ID = "6012002000326"
BIN_SIZE = "15min"
FEATURES = list(NUMERIC_FEATURES_WITH_OCCUPANCY)
CV_FOLDS = 5
#: First reading of the genuine record (Section 2.4); earlier rows are stale packets.
GENUINE_START = "2025-09-07"

BUILDERS = {
    "Linear Regression": build_linear_regression_pipeline,
    "Decision Tree": build_decision_tree_regression_pipeline,
    "SVR": build_svr_pipeline,
    "Neural Network": build_neural_network_regression_pipeline,
}

#: Search grids, kept small enough to run in minutes. Each includes the
#: original setting, so tuning can only match or improve the CV score.
GRIDS = {
    "Linear Regression": {},
    "Decision Tree": {"regress__max_depth": [4, 6, 8, 10, None], "regress__min_samples_leaf": [1, 10, 50]},
    "SVR": {"regress__C": [1.0, 10.0, 100.0], "regress__epsilon": [0.5, 2.0, 5.0], "regress__gamma": ["scale", 0.03, 0.3]},
    "Neural Network": {"regress__hidden_layer_sizes": [(16,), (32, 16), (64, 32)], "regress__alpha": [1e-4, 1e-2, 1.0]},
}


def load_table(env_long: pd.DataFrame, occupancy: pd.Series, bin_size: str = BIN_SIZE, rolling_window: int = 4):
    """Supervised table for the main sensor at a given bin size and history length."""
    co2 = co2_series(env_long, SENSOR_ID, bin_size)
    table, _ = build_co2_supervised_dataset(co2, occupancy, rolling_window=rolling_window)
    return table


def tune(train: pd.DataFrame) -> dict:
    """Grid-search every method by time-series CV on the training period."""
    cv = TimeSeriesSplit(n_splits=CV_FOLDS)
    out = {}
    for name, builder in BUILDERS.items():
        grid = GridSearchCV(
            builder(), GRIDS[name] or {"regress__fit_intercept": [True]},
            cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1, refit=False,
        )
        grid.fit(train[FEATURES], train["target"])
        res = pd.DataFrame(grid.cv_results_)
        best = res.loc[res["rank_test_score"].idxmin()]
        fold_cols = [f"split{i}_test_score" for i in range(CV_FOLDS)]
        # The original setting is in every grid: find its row for the baseline score.
        defaults = builder()[-1].get_params()
        is_default = np.ones(len(res), dtype=bool)
        for key in GRIDS[name]:
            is_default &= res[f"param_{key}"].astype(str) == str(defaults[key.split("__")[1]])
        base = res[is_default].iloc[0] if GRIDS[name] else best
        out[name] = {
            "best_params": {k.split("__")[1]: v for k, v in best["params"].items() if k != "regress__fit_intercept"},
            "cv_mae_tuned": -float(best["mean_test_score"]),
            "cv_mae_tuned_std": float(best["std_test_score"]),
            "cv_mae_baseline": -float(base["mean_test_score"]),
            "cv_mae_per_fold_tuned": [-float(best[c]) for c in fold_cols],
            "n_settings_tried": int(len(res)),
        }
        print(f"  {name}: CV MAE {out[name]['cv_mae_baseline']:.3f} -> {out[name]['cv_mae_tuned']:.3f} with {out[name]['best_params']}")
    return out


def cost(model, X_train, y_train, X_test) -> dict:
    """Timing, memory and operation counts for one configuration."""
    fit_s = median_time(lambda: model.fit(X_train, y_train), repeats=3)
    predict_s = median_time(lambda: model.predict(X_test), repeats=5)
    flops_row = prediction_flops(model, X_test.shape[1])
    return {
        "fit_s": fit_s,
        "predict_s": predict_s,
        "predict_ms_per_row": 1e3 * predict_s / len(X_test),
        "fit_peak_mb": peak_memory_mb(lambda: model.fit(X_train, y_train)),
        "model_size_kb": model_size_kb(model),
        "flops_per_row": flops_row,
        "achieved_gflops": flops_row * len(X_test) / predict_s / 1e9,
        "n_support_vectors": int(len(model[-1].support_)) if hasattr(model[-1], "support_") else None,
    }


def bootstrap_vs(y, pred_a, pred_b) -> dict:
    """Paired MAE difference (A minus B, positive = B better) with a 95% CI."""
    diff = np.abs(pred_a - y) - np.abs(pred_b - y)
    lo, hi = block_bootstrap_ci(diff)
    return {"mae_difference": float(diff.mean()), "ci95": [lo, hi]}


def profile_workflow(env_path: Path, occ_path: Path) -> list[dict]:
    """cProfile the end-to-end baseline workflow; return the costliest calls."""
    def workflow():
        """The whole pipeline from files to a fitted SVR, as one call for the profiler."""
        env_long = env_readings_raw_or_processed(env_path)
        occ = building_occupancy_series(resampled_occupancy_raw_or_processed(occ_path))
        table = load_table(env_long, occ)
        train, test, _ = chronological_split(table, time_col="time", train_frac=0.8)
        build_svr_pipeline().fit(train[FEATURES], train["target"]).predict(test[FEATURES])

    profiler = cProfile.Profile()
    profiler.runcall(workflow)
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    total = stats.total_tt
    rows = []
    for (file, line, func), (_, ncalls, tottime, cumtime, _) in stats.stats.items():
        rows.append({"function": f"{Path(file).name}:{func}", "calls": ncalls, "own_s": tottime, "cumulative_s": cumtime})
    rows.sort(key=lambda r: r["own_s"], reverse=True)
    print(f"  profiled workflow: {total:.1f} s total")
    return [{**r, "share_of_total": r["own_s"] / total} for r in rows[:8]]


def main() -> None:
    """Run tuning, selection, significance, cost, profiling and sensitivity; save everything."""
    env_long = env_readings_raw_or_processed(ENV_RAW_PATH)
    occupancy = building_occupancy_series(resampled_occupancy_raw_or_processed(OCC_RAW_PATH, BIN_SIZE))
    table = load_table(env_long, occupancy)
    train, test, cutoff = chronological_split(table, time_col="time", train_frac=0.8)
    X_tr, y_tr, X_te, y = train[FEATURES], train["target"], test[FEATURES], test["target"].to_numpy()
    results: dict = {"cutoff": str(cutoff), "n_train": len(train), "n_test": len(test), "features": FEATURES}

    print("1. Tuning by time-series cross-validation on the training period")
    tuning = tune(train)
    results["tuning"] = tuning
    selected = min(tuning, key=lambda n: tuning[n]["cv_mae_tuned"])
    results["selected_by_cv"] = selected
    print(f"  selected by CV: {selected}")

    print("2. Baseline vs tuned on the test period, with costs")
    persistence = test["co2_now"].to_numpy()
    results["persistence"] = evaluate_regression_predictions(y, persistence)
    preds, comparison = {}, {}
    for name, builder in BUILDERS.items():
        for variant, params in (("baseline", {}), ("tuned", tuning[name]["best_params"])):
            model = builder(**params).fit(X_tr, y_tr)
            pred = model.predict(X_te)
            preds[(name, variant)] = pred
            comparison[f"{name} | {variant}"] = {
                **evaluate_regression_predictions(y, pred),
                "train_mae": evaluate_regression_predictions(y_tr, model.predict(X_tr))["mae"],
                **cost(model, X_tr, y_tr, X_te),
            }
            c = comparison[f"{name} | {variant}"]
            print(f"  {name:18s} {variant:8s} MAE {c['mae']:.3f}  R2 {c['r2']:.3f}  predict {c['predict_s']*1e3:8.1f} ms  {c['flops_per_row']:.3g} FLOP/row")
    results["comparison"] = comparison

    sel = preds[(selected, "tuned")]
    results["significance"] = {
        "persistence_vs_selected": bootstrap_vs(y, persistence, sel),
        **{f"{n}_tuned_vs_selected": bootstrap_vs(y, preds[(n, 'tuned')], sel) for n in BUILDERS if n != selected},
        "selected_baseline_vs_tuned": bootstrap_vs(y, preds[(selected, "baseline")], sel),
    }
    # Failure modes: averages hide what happens at sudden rises, the moments
    # an early-warning forecast exists for.
    rising, high = (y - persistence) > 30, y > 550
    failure = {"rows_rising_over_30ppm": int(rising.sum()), "rows_above_550ppm": int(high.sum())}
    candidates = [("Persistence", persistence), (f"{selected} | baseline", preds[(selected, "baseline")])]
    candidates += [(f"{n} | tuned", preds[(n, "tuned")]) for n in BUILDERS]
    for label, p in candidates:
        err = p - y
        failure[label] = {
            "max_prediction": float(p.max()),
            "bias": float(err.mean()),
            "mae_rising": float(np.abs(err[rising]).mean()),
            "mae_above_550": float(np.abs(err[high]).mean()),
            # 0 = predicted none of the coming rise (same as persistence), 1 = all of it.
            "median_share_of_rise_predicted": float(np.median((p[rising] - persistence[rising]) / (y[rising] - persistence[rising]))),
        }
    results["failure_modes"] = failure
    results["svr_arithmetic_intensity_flop_per_byte"] = svr_arithmetic_intensity(len(FEATURES))
    results["test_rows_above_600ppm"] = int((y > 600).sum())

    print("3. Profiling the end-to-end workflow")
    results["profile_top_functions"] = profile_workflow(ENV_RAW_PATH, OCC_RAW_PATH)

    print("4. Sensitivity and robustness")
    sel_params = tuning[selected]["best_params"]
    sel_builder = BUILDERS[selected]

    # Input resolution: one-bin-ahead forecasts at three bin sizes; occupancy
    # exists only on the 15-minute grid and adds nothing (Section 4), so it
    # is left out here and skill is measured against persistence.
    resolution = {}
    no_occ = list(NUMERIC_FEATURES_WITHOUT_OCCUPANCY)
    for bin_size, window in (("10min", 6), ("15min", 4), ("30min", 2)):
        co2 = co2_series(env_long, SENSOR_ID, bin_size)
        placeholder = pd.Series(0.0, index=co2.index, name="building_occupancy")
        t, _ = build_co2_supervised_dataset(co2, placeholder, rolling_window=window)
        tr, te, _ = chronological_split(t, time_col="time", train_frac=0.8)
        pers = evaluate_regression_predictions(te["target"], te["co2_now"])["mae"]
        row = {"rows": len(t), "persistence_mae": pers}
        for label, b, p in (("Linear Regression", build_linear_regression_pipeline, {}), (selected, sel_builder, sel_params)):
            mae = evaluate_regression_predictions(te["target"], b(**p).fit(tr[no_occ], tr["target"]).predict(te[no_occ]))["mae"]
            row[label] = {"mae": mae, "skill_vs_persistence": 1 - mae / pers}
        resolution[bin_size] = row
        print(f"  bin {bin_size}: rows {len(t):,}, persistence {pers:.2f}, {selected} {row[selected]['mae']:.2f}")
    results["input_resolution"] = resolution

    # Data requirement: train on the most recent fraction of the training period.
    curve = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        part = train.iloc[int(len(train) * (1 - frac)):]
        curve[str(frac)] = {
            "rows": len(part),
            selected: evaluate_regression_predictions(y, sel_builder(**sel_params).fit(part[FEATURES], part["target"]).predict(X_te))["mae"],
            "Linear Regression": evaluate_regression_predictions(y, build_linear_regression_pipeline().fit(part[FEATURES], part["target"]).predict(X_te))["mae"],
        }
    results["learning_curve"] = curve

    # Repeatability: the MLP's loss is non-convex, so its result depends on the seed.
    mlp_params = tuning["Neural Network"]["best_params"]
    seeds = [evaluate_regression_predictions(y, build_neural_network_regression_pipeline(**{**mlp_params, "random_state": s}).fit(X_tr, y_tr).predict(X_te))["mae"] for s in range(5)]
    svr_repeat = [evaluate_regression_predictions(y, build_svr_pipeline(**tuning["SVR"]["best_params"]).fit(X_tr, y_tr).predict(X_te))["mae"] for _ in range(2)]
    results["repeatability"] = {"mlp_mae_over_5_seeds": seeds, "svr_mae_two_runs": svr_repeat}

    # Integration gap threshold: how much cumulative exposure depends on it.
    readings = pivot_variable(env_long, "Carbon dioxide")
    readings = readings[readings["sensorid"] == SENSOR_ID].sort_values("time")
    gap = {}
    for hours in (0.5, 1, 2, 6, 24, 1e9):
        r = trapezoidal_integral_gap_aware(readings["time"], readings["value"], hours)
        gap["naive" if hours == 1e9 else str(hours)] = {"total_ppmh": r["total"], "hours_included": r["hours_included"]}
    results["integration_gap_threshold"] = gap

    # Outliers: robust screening of the genuine readings, and whether removing
    # them from the training data changes the test error.
    raw = readings.drop_duplicates("time").set_index("time")["value"]
    genuine = raw[raw.index >= pd.Timestamp(GENUINE_START, tz=raw.index.tz)]
    flags = hampel_flags(genuine)
    local_median = genuine.rolling(7, center=True, min_periods=4).median()
    cleaned = raw.copy()
    cleaned.loc[flags[flags].index] = local_median[flags]
    clean_env = pd.DataFrame({"sensorid": SENSOR_ID, "time": cleaned.index, "variable": "Carbon dioxide", "value": cleaned.values})
    clean_train = load_table(clean_env, occupancy)
    clean_train = clean_train[clean_train["time"] < cutoff]
    results["outliers"] = {
        "genuine_readings": int(len(genuine)),
        "flagged": int(flags.sum()),
        "share_at_sensor_floor_400ppm": float((genuine == 400).mean()),
        "test_mae_trained_raw": comparison[f"{selected} | tuned"]["mae"],
        "test_mae_trained_cleaned": evaluate_regression_predictions(
            y, sel_builder(**sel_params).fit(clean_train[FEATURES], clean_train["target"]).predict(X_te))["mae"],
    }
    rates = spike_decay_rates(genuine)
    results["ventilation_from_decays"] = {
        "events": int(len(rates)),
        "median_rate_per_h": float(np.median(rates)),
        "iqr_rate_per_h": [float(np.percentile(rates, 25)), float(np.percentile(rates, 75))],
        "median_time_constant_min": float(60 / np.median(rates)),
    }
    print(f"  outliers flagged {results['outliers']['flagged']}; decay events {len(rates)}, "
          f"median rate {np.median(rates):.2f}/h")

    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {RESULTS_PATH}")
    chart(results, selected)


COLOURS = {"Linear Regression": "#2a78d6", "Decision Tree": "#eb6834", "SVR": "#1baf7a", "Neural Network": "#7a4fd6"}


def chart(results: dict, selected: str) -> None:
    """Chart 17: tuning (a), accuracy against cost (b) and data requirement (c)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    names = list(BUILDERS)
    tuning = results["tuning"]

    # (a) Dot plot, not bars: the differences are a few tenths of a ppm, and
    # bars would either hide them (axis from zero) or exaggerate them (cut axis).
    ax = axes[0]
    for i, n in enumerate(names):
        before, after = tuning[n]["cv_mae_baseline"], tuning[n]["cv_mae_tuned"]
        ax.plot([before, after], [i, i], color="#b0aea5", lw=2, zorder=1)
        ax.scatter(before, i, color="#b0aea5", s=60, zorder=2)
        ax.scatter(after, i, color=COLOURS[n], s=60, zorder=3)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated MAE on training period (ppm)")
    ax.set_title("(a) Tuning on training data only")
    ax.legend(handles=[plt.Line2D([], [], color="#b0aea5", marker="o", ls="", label="original settings"),
                       plt.Line2D([], [], color="k", marker="o", ls="", label="tuned (method colour)")],
              frameon=False, loc="upper right")

    # (b) Colour = method, marker = settings; a legend avoids overlapping labels.
    ax = axes[1]
    for key, c in results["comparison"].items():
        name, variant = key.split(" | ")
        ax.scatter(c["predict_ms_per_row"], c["mae"], marker="o" if variant == "tuned" else "x", s=55, color=COLOURS[name])
    ax.axhline(results["persistence"]["mae"], color="grey", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Prediction time per row (ms, log scale)")
    ax.set_ylabel("Test MAE (ppm)")
    ax.set_title("(b) Accuracy vs cost on the test period")
    handles = [plt.Line2D([], [], color=COLOURS[n], marker="o", ls="", label=n) for n in names]
    handles += [plt.Line2D([], [], color="k", marker="x", ls="", label="original settings"),
                plt.Line2D([], [], color="k", marker="o", ls="", label="tuned"),
                plt.Line2D([], [], color="grey", ls="--", label="persistence")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.94))

    ax = axes[2]
    curve = results["learning_curve"]
    rows = [curve[k]["rows"] for k in curve]
    ax.plot(rows, [curve[k][selected] for k in curve], "o-", color=COLOURS[selected], label=f"{selected} (tuned)")
    ax.plot(rows, [curve[k]["Linear Regression"] for k in curve], "s-", color=COLOURS["Linear Regression"], label="Linear Regression")
    ax.axhline(results["persistence"]["mae"], color="grey", ls="--", lw=1, label="persistence")
    ax.set_xlabel("Training rows used (most recent)")
    ax.set_ylabel("Test MAE (ppm)")
    ax.set_title("(c) Data requirement")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {CHART_PATH}")


if __name__ == "__main__":
    import sys

    if "--chart-only" in sys.argv:  # redraw from saved results without rerunning
        saved = json.loads(RESULTS_PATH.read_text())
        chart(saved, saved["selected_by_cv"])
    else:
        main()
