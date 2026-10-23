"""Live demo: replay real CO2 history, then predict CO2 for scenarios you type in.

Trains the selected model (tuned SVR, chosen by time-series
cross-validation in scripts/optimisation_and_performance.py; report
Section 5.4) fresh each run -- fast enough (a few seconds) on this
dataset's size that a saved model file isn't needed. Invalid inputs are
rejected, and inputs outside the training range are flagged.

Usage:
    python scripts/co2_demo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from occupancy.co2_models import build_selected_pipeline
from occupancy.co2_regression import (
    NUMERIC_FEATURES_WITH_OCCUPANCY,
    build_co2_supervised_dataset,
    building_occupancy_series,
    co2_series,
)
from occupancy.evaluation import chronological_split
from occupancy.input_validation import extrapolation_warnings, training_ranges, validate_query
from occupancy.processed_data import (
    env_readings_raw_or_processed,
    resampled_occupancy_raw_or_processed,
)

OCC_RAW_PATH = Path("data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
ENV_RAW_PATH = Path("data/raw/5EnvSensor_MayToDec2024_180kRows.csv")
SENSOR_ID = "6012002000326"
BIN_SIZE = "15min"

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_and_train():
    """Build the supervised table, train the selected SVR on the training period.

    Returns:
        ``(pipeline, feature columns, test rows, training ranges)``; the ranges
        are used to flag extrapolation in the interactive part.
    """
    print("Loading data and training the selected model (tuned SVR, report Section 5.4)...")
    env_long = env_readings_raw_or_processed(ENV_RAW_PATH)
    occ_resampled = resampled_occupancy_raw_or_processed(OCC_RAW_PATH, BIN_SIZE)
    occupancy = building_occupancy_series(occ_resampled)
    co2 = co2_series(env_long, SENSOR_ID, BIN_SIZE)
    table, _ = build_co2_supervised_dataset(co2, occupancy, rolling_window=4)

    x_cols = list(NUMERIC_FEATURES_WITH_OCCUPANCY)
    # Same chronological 80/20 split as the report, so the replayed test rows
    # were never seen during training.
    train, test, cutoff = chronological_split(table, time_col="time", train_frac=0.8)

    pipeline = build_selected_pipeline()
    pipeline.fit(train[x_cols], train["target"])
    print(f"Trained on {len(train):,} rows (up to {cutoff}), {len(test):,} held-out test rows.\n")
    return pipeline, x_cols, test, training_ranges(train)


def replay_real_examples(pipeline, x_cols, test: pd.DataFrame, n: int = 12) -> None:
    """Predict ``n`` random test rows and print each prediction beside the truth."""
    print("=" * 76)
    print("PART 1 -- Replaying real examples from the test set")
    print("(these 15-minute intervals were never seen during training)")
    print("=" * 76)

    # Fixed seed so the demo shows the same rows every run.
    sample = test.sample(n=n, random_state=7).sort_values("time")
    pred = pipeline.predict(sample[x_cols])

    for (_, row), p in zip(sample.iterrows(), pred):
        actual = row["target"]
        error = p - actual  # positive = over-prediction
        print(
            f"  {row['time']:%a %d-%b %H:%M}  "
            f"CO2 now={row['co2_now']:6.1f}ppm  occupancy={row['building_occupancy']:.0f}/5  "
            f"-> predicted next={p:6.1f}ppm  actual next={actual:6.1f}ppm  "
            f"(error {error:+.1f})"
        )

    mae = (sample.assign(pred=pred)["target"] - pred).abs().mean()
    print(f"\n  Mean absolute error on this sample: {mae:.2f} ppm "
          "(full test-set MAE is in reports/co2_prediction_results.json).\n")


def _build_query_row(x_cols: list[str], *, co2_now: float, co2_lag1: float,
                      rolling_mean: float, occupancy: float, hour: int, dow: int) -> pd.DataFrame:
    """Turn a typed-in scenario into one row with the model's nine features."""
    # Hour and day are encoded on a circle exactly as in training
    # (occupancy.co2_regression), so 23:00 sits next to 00:00.
    row = {
        "co2_now": co2_now,
        "co2_lag1": co2_lag1,
        "co2_rolling_1h": rolling_mean,
        "building_occupancy": occupancy,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dow / 7),
        "dow_cos": math.cos(2 * math.pi * dow / 7),
        "is_weekend": float(dow >= 5),
    }
    return pd.DataFrame([row])[x_cols]


def _ask_number(prompt: str, default: float | None = None) -> float:
    """Prompt until the user types a number (or accepts ``default`` with blank)."""
    while True:
        raw = input(prompt).strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"  '{raw}' is not a number, try again.")


def interactive_loop(pipeline, x_cols: list[str], ranges: dict[str, tuple[float, float]]) -> None:
    """Ask for scenarios until the user quits, validating each before predicting."""
    print("=" * 76)
    print("PART 2 -- Try your own scenario")
    print("=" * 76)
    print("(leave the first answer blank at any time to quit)\n")

    while True:
        co2_raw = input("Current CO2 level in ppm, e.g. 550 (blank to quit): ").strip()
        if not co2_raw:
            print("Bye!")
            return
        try:
            co2_now = float(co2_raw)
        except ValueError:
            print(f"  '{co2_raw}' is not a number, try again.\n")
            continue

        co2_lag1 = _ask_number("CO2 15 minutes ago, ppm (blank = same as now): ", co2_now)
        rolling_mean = _ask_number("Average CO2 over the last hour, ppm (blank = same as now): ", co2_now)
        occupancy = _ask_number("How many of the 5 occupancy zones are occupied right now, 0-5: ", 0.0)
        hour = _ask_number("Hour of day right now, 0-23 (e.g. 14 for 2pm): ")
        print("Days: " + ", ".join(f"{i + 1}={d}" for i, d in enumerate(DAYS)))
        dow = _ask_number("Day of week, 1-7: ") - 1  # model uses Monday = 0

        # Invalid inputs are refused rather than predicted from: the model
        # would happily return a number for negative CO2 or hour 25.
        problems = validate_query(
            co2_now=co2_now, co2_lag1=co2_lag1, co2_rolling_1h=rolling_mean,
            building_occupancy=occupancy, hour=hour, day_of_week=dow,
        )
        if problems:
            print("\n  Input rejected, no prediction made:")
            for problem in problems:
                print(f"   - {problem}")
            print()
            continue

        query = _build_query_row(
            x_cols, co2_now=co2_now, co2_lag1=co2_lag1, rolling_mean=rolling_mean,
            occupancy=occupancy, hour=hour, dow=dow,
        )
        predicted = pipeline.predict(query)[0]
        # Valid but outside the training data: still predicted, but flagged,
        # because the model has not learned how CO2 behaves there.
        warnings = extrapolation_warnings(
            {"co2_now": co2_now, "co2_lag1": co2_lag1, "co2_rolling_1h": rolling_mean}, ranges
        )
        direction = "rising" if predicted > co2_now else ("falling" if predicted < co2_now else "flat")

        print(
            f"\n  --> Model predicts {predicted:.1f} ppm in 15 minutes "
            f"(currently {co2_now:.1f} ppm, {direction})."
        )
        for warning in warnings:
            print(f"  WARNING: {warning}")
        print()


def main() -> None:
    """Train the model, replay real test rows, then take typed-in scenarios."""
    pipeline, x_cols, test, ranges = load_and_train()
    replay_real_examples(pipeline, x_cols, test)
    try:
        interactive_loop(pipeline, x_cols, ranges)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
