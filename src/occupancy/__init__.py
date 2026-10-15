"""co2_prediction: indoor CO2 concentration forecasting for MMA3001.

Predicts CO2 concentration 15 minutes ahead from recent sensor history
and building occupancy, using all four Week 5 regression methods
(Linear Regression, Decision Tree, SVR, Neural Network).

Also includes Week 6 numerical integration of accumulated CO2 exposure,
sensor selection (BoM external-weather validation), sensor-matching
analysis and an unbroken-interval audit of every sensor.

See the project README and ``docs/report`` for the full engineering
context, validation methodology and results.
"""

from occupancy.co2_models import (
    build_decision_tree_regression_pipeline,
    build_linear_regression_pipeline,
    build_neural_network_regression_pipeline,
    build_svr_pipeline,
)
from occupancy.co2_regression import (
    build_co2_supervised_dataset,
    building_occupancy_series,
    co2_series,
)
from occupancy.co2_integration import (
    trapezoidal_integral,
    trapezoidal_integral_gap_aware,
    simpsons_integral,
    richardson_order_check,
)
from occupancy.env_sensors import load_env_sensor_log, pivot_variable
from occupancy.evaluation import (
    chronological_split,
    evaluate_regression_predictions,
)
from occupancy.external_weather import load_bom_weather
from occupancy.sensor_intervals import find_unbroken_intervals, interval_audit, usable_rows
from occupancy.sensor_matching import build_correlation_matrix, lagged_correlation
from occupancy.sensors import sensor_label
from occupancy.weather_comparison import compare_to_bom, daily_indoor_series

__all__ = [
    # CO2 regression (Week 5)
    "build_co2_supervised_dataset",
    "building_occupancy_series",
    "co2_series",
    "build_linear_regression_pipeline",
    "build_decision_tree_regression_pipeline",
    "build_svr_pipeline",
    "build_neural_network_regression_pipeline",
    # Numerical integration (Week 6)
    "trapezoidal_integral",
    "trapezoidal_integral_gap_aware",
    "simpsons_integral",
    "richardson_order_check",
    # Evaluation
    "chronological_split",
    "evaluate_regression_predictions",
    # Environmental sensors
    "load_env_sensor_log",
    "pivot_variable",
    "sensor_label",
    "find_unbroken_intervals",
    "interval_audit",
    "usable_rows",
    # Sensor validation
    "build_correlation_matrix",
    "lagged_correlation",
    "load_bom_weather",
    "compare_to_bom",
    "daily_indoor_series",
]

__version__ = "0.2.0"
