"""The four regression methods taught in MMA3001 Week 5, applied to CO2 prediction.

Week 5's notebooks (5.1-5.4) each cover one regression technique --
Linear Regression, Decision Tree Regression, Support Vector Regression and
Neural Network Regression -- and 5.5 teaches comparing them with MAE/RMSE/R².
This module builds all four as directly comparable pipelines for the CO2
regression task in :mod:`occupancy.co2_regression`, so the "which method is
most suitable, and why" comparison (occupancy.evaluation, the "Alternative
solutions" report requirement) uses methods actually taught in the unit --
unlike the occupancy classifier, which uses classification methods (Logistic
Regression, Random Forest Classifier) that Week 5 never covers.

All four share one preprocessing step (`StandardScaler`). Scaling is
essential for SVR and the neural network (both sensitive to feature scale)
and harmless for the other two: ordinary least squares is scale-invariant
in its predictions, and a decision tree's splits are invariant to any
monotonic per-feature transform. Using one shared preprocessing step
keeps the four models genuinely comparable -- differences in performance
come from the model, not from inconsistent inputs.
"""

from __future__ import annotations

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def build_linear_regression_pipeline(**kwargs) -> Pipeline:
    """Week 5.1: ordinary least-squares linear regression.

    Args:
        **kwargs: Passed to ``sklearn.linear_model.LinearRegression``.

    Returns:
        An unfitted ``Pipeline`` of ``StandardScaler`` then the regressor.
        Call ``.fit(X, y)`` with the feature columns in
        :data:`occupancy.co2_regression.NUMERIC_FEATURES_WITH_OCCUPANCY`
        (or the without-occupancy set) and the next-bin CO2 target, in ppm.
    """
    # Scaling does not change OLS predictions, but it is kept so all four
    # models see identical preprocessing and the comparison stays fair.
    return Pipeline(
        [("scale", StandardScaler()), ("regress", LinearRegression(**kwargs))]
    )


def build_decision_tree_regression_pipeline(**kwargs) -> Pipeline:
    """Week 5.2: a single decision tree regressor.

    Args:
        **kwargs: Passed to ``sklearn.tree.DecisionTreeRegressor``,
            overriding the defaults ``max_depth=8`` and ``random_state=42``.

    Returns:
        An unfitted ``Pipeline`` of ``StandardScaler`` then the tree.
    """
    # Depth 8 allows at most 256 leaves for ~13,600 training rows: enough
    # for the daily cycle, too few to memorise individual days. An unlimited
    # tree fits training data almost exactly and extrapolates worst.
    defaults = dict(max_depth=8, random_state=42)
    defaults.update(kwargs)
    return Pipeline(
        [("scale", StandardScaler()), ("regress", DecisionTreeRegressor(**defaults))]
    )


def build_svr_pipeline(**kwargs) -> Pipeline:
    """Week 5.3: Support Vector Regression with the RBF kernel.

    Args:
        **kwargs: Passed to ``sklearn.svm.SVR``, overriding the defaults
            ``C=10.0`` and ``epsilon=0.5``.

    Returns:
        An unfitted ``Pipeline`` of ``StandardScaler`` then the SVR.
    """
    # Scaling is essential here: the RBF kernel uses Euclidean distance, so
    # unscaled CO2 (hundreds of ppm) would swamp the 0/1 weekend flag.
    # These are the original settings (report Section 4): errors within
    # epsilon=0.5 ppm cost nothing, which is below the sensor's 1 ppm
    # resolution, so 94% of training rows become support vectors. Tuning
    # (Section 6) selected SELECTED_SVR_PARAMS instead.
    defaults = dict(C=10.0, epsilon=0.5)
    defaults.update(kwargs)
    return Pipeline([("scale", StandardScaler()), ("regress", SVR(**defaults))])


#: Settings chosen by 5-fold time-series cross-validation on the training
#: period only (``scripts/optimisation_and_performance.py``, report Section 6).
#: epsilon=2 ppm ignores errors up to twice the sensor's resolution; C=100
#: and gamma=0.03 give a smoother, less regularised fit than the originals.
SELECTED_SVR_PARAMS = {"C": 100.0, "epsilon": 2.0, "gamma": 0.03}


def build_selected_pipeline() -> Pipeline:
    """The final design: SVR with :data:`SELECTED_SVR_PARAMS`.

    Returns:
        An unfitted ``Pipeline``; selected over the other three methods and
        the original SVR settings by cross-validated MAE, then confirmed on
        the held-out test period (report Section 6).
    """
    return build_svr_pipeline(**SELECTED_SVR_PARAMS)


def build_neural_network_regression_pipeline(**kwargs) -> Pipeline:
    """Week 5.4: a small multi-layer-perceptron regressor.

    Args:
        **kwargs: Passed to ``sklearn.neural_network.MLPRegressor``,
            overriding the defaults below.

    Returns:
        An unfitted ``Pipeline`` of ``StandardScaler`` then the MLP.
    """
    # The loss is non-convex, so the result depends on the random starting
    # weights: random_state fixes them for repeatability. early_stopping
    # holds out 10% of training rows and stops after 10 epochs without
    # improvement -- the convergence criterion, and a guard against
    # overfitting a small dataset. 32 -> 16 units is ample for 9 inputs.
    defaults = dict(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        random_state=42,
        early_stopping=True,
    )
    defaults.update(kwargs)
    return Pipeline(
        [("scale", StandardScaler()), ("regress", MLPRegressor(**defaults))]
    )
