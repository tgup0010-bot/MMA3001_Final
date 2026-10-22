"""Chronological train/test splitting and prediction-quality metrics.

Splitting must be done by time, never at random: a randomly shuffled split
would place test rows beside near-identical training rows (CO2 15 minutes
apart is strongly correlated), so the score would measure interpolation
rather than forecasting an unseen future, silently inflating every metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def chronological_split(
    table: pd.DataFrame,
    *,
    time_col: str = "collecteddate",
    train_frac: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split a table into train/test sets by time, not at random.

    All rows before a single cutoff timestamp (the ``train_frac`` quantile
    of the sorted timestamps) go to training; everything from the cutoff on
    goes to test. This matches deployment, where a model only ever sees the
    past.

    Args:
        table: Supervised table (e.g. from
            :func:`occupancy.co2_regression.build_co2_supervised_dataset`).
        time_col: Column holding each row's timestamp.
        train_frac: Fraction of rows (by time, not room-balanced count)
            assigned to training.

    Returns:
        ``(train, test, cutoff_time)``: two DataFrames with the same
        columns as ``table``, and the first test timestamp.

    Raises:
        ValueError: If ``train_frac`` is not strictly between 0 and 1, or
            the split would leave either set empty.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    # Sort first so "the first 80% of rows" means "the earliest 80%".
    sorted_table = table.sort_values(time_col).reset_index(drop=True)
    cutoff_idx = int(len(sorted_table) * train_frac)
    if cutoff_idx == 0 or cutoff_idx >= len(sorted_table):
        raise ValueError(
            f"train_frac={train_frac} produced an empty train or test set "
            f"for {len(sorted_table)} rows."
        )

    cutoff_time = sorted_table.loc[cutoff_idx, time_col]
    train = sorted_table.iloc[:cutoff_idx].reset_index(drop=True)
    test = sorted_table.iloc[cutoff_idx:].reset_index(drop=True)
    return train, test, cutoff_time


def evaluate_regression_predictions(y_true, y_pred) -> dict:
    """Compute the standard MMA3001 Week 5.5 regression metrics.

    Args:
        y_true: Ground-truth continuous target values.
        y_pred: Predicted values.

    Returns:
        A dict with ``mae`` (mean absolute error, in target units),
        ``rmse`` (root mean squared error, in target units, more
        sensitive to large errors than MAE), ``r2`` (fraction of target
        variance explained relative to a mean-prediction baseline), and
        ``n`` (rows evaluated).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Report all three: MAE is the typical error, RMSE weights the rare large
    # misses (CO2 spikes) more, and R2 compares against predicting the mean.
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "n": int(len(y_true)),
    }
