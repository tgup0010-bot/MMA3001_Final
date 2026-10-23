"""Unit tests for occupancy.performance."""

import numpy as np
import pytest

from occupancy.co2_models import (
    build_decision_tree_regression_pipeline,
    build_linear_regression_pipeline,
    build_neural_network_regression_pipeline,
    build_svr_pipeline,
)
from occupancy.performance import (
    EXP_FLOPS,
    median_time,
    model_size_kb,
    peak_memory_mb,
    prediction_flops,
    svr_arithmetic_intensity,
)

RNG = np.random.default_rng(0)
X = RNG.normal(size=(200, 3))
Y = X @ np.array([1.0, -2.0, 0.5]) + RNG.normal(0, 0.1, 200)


def test_linear_flops_are_scaling_plus_one_multiply_add_per_feature():
    model = build_linear_regression_pipeline().fit(X, Y)
    assert prediction_flops(model, 3) == 2 * 3 + 2 * 3


def test_svr_flops_scale_with_support_vectors():
    model = build_svr_pipeline().fit(X, Y)
    n_sv = len(model[-1].support_)
    assert prediction_flops(model, 3) == 6 + n_sv * (9 + EXP_FLOPS + 2)


def test_mlp_flops_count_every_weight_and_bias():
    model = build_neural_network_regression_pipeline(hidden_layer_sizes=(4,), max_iter=50).fit(X, Y)
    # scaling 6; layer 3->4: 2*3*4+4 = 28; layer 4->1: 2*4+1 = 9; ReLU 4
    assert prediction_flops(model, 3) == 6 + 28 + 9 + 4


def test_tree_flops_equal_depth_plus_scaling():
    model = build_decision_tree_regression_pipeline(max_depth=3).fit(X, Y)
    assert prediction_flops(model, 3) == 6 + model[-1].get_depth()


def test_svr_arithmetic_intensity_formula():
    assert svr_arithmetic_intensity(9) == pytest.approx((27 + EXP_FLOPS + 2) / 80)


def test_measurement_helpers_return_positive_numbers():
    assert median_time(lambda: sum(range(1000)), repeats=3) > 0
    assert peak_memory_mb(lambda: [0.0] * 100_000) > 0.5
    assert model_size_kb(build_linear_regression_pipeline().fit(X, Y)) > 0
