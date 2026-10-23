"""Timing, memory and operation-count estimates for the fitted models.

Wall-clock time alone says how long something took on one machine;
operation counts say *why*. This module provides both:

- :func:`median_time` and :func:`peak_memory_mb` measure;
- :func:`prediction_flops` estimates the floating-point operations needed
  to predict one row, from the structure of each fitted model, and
  :func:`svr_arithmetic_intensity` relates those operations to the data
  each prediction must read.
"""

from __future__ import annotations

import pickle
import time
import tracemalloc
from typing import Callable

import numpy as np
from sklearn.pipeline import Pipeline

#: Rough cost of one exponential, in floating-point operations. Library
#: implementations take a few tens of operations; the exact figure changes
#: the SVR estimate by a few percent, not its order of magnitude.
EXP_FLOPS = 20

#: Bytes per float64 value.
FLOAT_BYTES = 8


def median_time(fn: Callable[[], object], repeats: int = 5) -> float:
    """Median wall-clock time of ``fn()`` over several runs.

    Args:
        fn: A zero-argument callable, e.g. ``lambda: model.predict(X)``.
        repeats: Number of timed runs.

    Returns:
        The median duration in seconds. The median, not the mean, so one
        run slowed by the operating system does not distort the result.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def peak_memory_mb(fn: Callable[[], object]) -> float:
    """Peak Python heap memory allocated while running ``fn()``.

    Args:
        fn: A zero-argument callable.

    Returns:
        The peak traced allocation in megabytes (``tracemalloc``; memory
        allocated by compiled libraries outside Python's allocator is only
        partly visible, so treat this as a lower bound).
    """
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1e6


def model_size_kb(model: object) -> float:
    """Size of a fitted model when serialised, in kilobytes.

    Args:
        model: Any picklable fitted model.

    Returns:
        The pickled size, a proxy for the memory needed to deploy it.
    """
    return len(pickle.dumps(model)) / 1e3


def prediction_flops(pipeline: Pipeline, n_features: int) -> float:
    """Estimated floating-point operations to predict one row.

    Args:
        pipeline: A fitted pipeline from :mod:`occupancy.co2_models`
            (``StandardScaler`` then one regressor).
        n_features: Number of input features ``d``.

    Returns:
        Operations per predicted row, including the scaling step (one
        subtraction and one division per feature):

        - linear: ``2d`` (multiply-add per feature);
        - decision tree: ``depth`` (one comparison per level visited);
        - SVR (RBF): ``n_sv * (3d + EXP_FLOPS + 2)`` -- for every support
          vector, ``d`` subtractions, squares and additions for the
          distance, one exponential, and a multiply-add into the sum;
        - MLP: ``sum(2 * n_in * n_out + n_out)`` over layers (weights,
          bias), plus one comparison per hidden unit for ReLU.
    """
    d = n_features
    scaling = 2 * d
    model = pipeline[-1]
    name = type(model).__name__
    if name == "LinearRegression":
        core = 2 * d
    elif name == "DecisionTreeRegressor":
        core = model.get_depth()
    elif name == "SVR":
        core = len(model.support_) * (3 * d + EXP_FLOPS + 2)
    elif name == "MLPRegressor":
        sizes = [d, *model.hidden_layer_sizes, 1]
        core = sum(2 * a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
        core += sum(model.hidden_layer_sizes)  # ReLU comparisons
    else:
        raise ValueError(f"No FLOP model for {name}")
    return float(scaling + core)


def svr_arithmetic_intensity(n_features: int) -> float:
    """Operations per byte read for an RBF-SVR prediction.

    Args:
        n_features: Number of input features ``d``.

    Returns:
        FLOPs per byte: each support vector contributes ``3d + EXP_FLOPS + 2``
        operations and must be read from memory (``d`` float64 values plus
        one coefficient). A low value (around 1 or below) means the
        computation is limited by memory bandwidth rather than arithmetic.
    """
    d = n_features
    return (3 * d + EXP_FLOPS + 2) / ((d + 1) * FLOAT_BYTES)
