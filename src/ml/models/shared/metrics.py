# src/ml/models/shared/metrics.py

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score

# -------------------------------------------------------------------------------------------
# Forecasting metrics
#
# https://gist.github.com/bshishov/5dc237f59f019b26145648e2124ca1c9
# -------------------------------------------------------------------------------------------

def _naive_forecasting(actual: np.ndarray, seasonality: int = 1):
    # Naive forecasting method which just repeats previous samples
    return actual[:-seasonality]

def root_mean_squared_scaled_error(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    # RMSSE - Root Mean Squared Scaled Error
    q = mean_squared_error(actual, predicted) / mean_squared_error(actual[seasonality:], _naive_forecasting(actual, seasonality))
    return np.sqrt(q)

def mean_absolute_scaled_error(actual: np.ndarray, predicted: np.ndarray, seasonality: int = 1):
    # MASE - Mean Absolute Scaled Error
    return mean_absolute_error(actual, predicted) / mean_absolute_error(actual[seasonality:], _naive_forecasting(actual, seasonality))

def _is_darts_timeseries(value):
    try:
        from darts import TimeSeries
    except ModuleNotFoundError:
        return False

    return isinstance(value, TimeSeries)

def calculate_metrics(actual, prediction, model_name, data_type):
    # Convert Darts TimeSeries to NumPy arrays if necessary
    if _is_darts_timeseries(actual):
        actual = actual.values().flatten()
    if _is_darts_timeseries(prediction):
        prediction = prediction.values().flatten()

    # Calculate metrics using scikit-learn functions
    metrics = {
        "MAE": mean_absolute_error(actual, prediction),
        "MSE": mean_squared_error(actual, prediction),
        "RMSE": root_mean_squared_error(actual, prediction),
        "RMSSE": root_mean_squared_scaled_error(actual, prediction),
        "MAPE": mean_absolute_percentage_error(actual, prediction) * 100,
        "MASE": mean_absolute_scaled_error(actual, prediction),
        "R2": r2_score(actual, prediction)
    }

    for metric_name, metric_value in metrics.items():
        value_str = f"{metric_value:.2f}"
        if metric_name == "MAPE":
            value_str += "%"
        print(f"{model_name} ({data_type}) | {metric_name}: {value_str}")

    return metrics
