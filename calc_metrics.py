import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(actual_values, predicted_values):
    if isinstance(actual_values, pd.core.series.Series):
        actual_values = actual_values.to_list()
    return mean_absolute_error(actual_values, predicted_values)


def calculate_mse(actual_values, predicted_values):
    if isinstance(actual_values, pd.core.series.Series):
        actual_values = actual_values.to_list()
    return mean_squared_error(actual_values, predicted_values)


def calculate_rmse(actual_values, predicted_values):
    if isinstance(actual_values, pd.core.series.Series):
        actual_values = actual_values.to_list()
    return np.sqrt(calculate_mse(actual_values, predicted_values))


def calculate_r2(actual_values, predicted_values):
    if isinstance(actual_values, pd.core.series.Series):
        actual_values = actual_values.to_list()
    return r2_score(actual_values, predicted_values)


def calculate_mape_0(actual_values, predicted_values):
    if isinstance(actual_values, pd.core.series.Series):
        actual_values = actual_values.to_list()

    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    # 过滤掉实际值为零的数据点
    mask = actual_values != 0
    actual_values = actual_values[mask]
    predicted_values = predicted_values[mask]

    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    return mape


def calculate_mape_1(actual_values, predicted_values):
    if isinstance(actual_values, pd.core.series.Series):
        actual_values = actual_values.to_list()

    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)

    actual_values += 1
    predicted_values += 1

    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    return mape


def calculate_metrics(label, pred):
    # 计算评估指标
    mae = calculate_mae(label, pred)
    mse = calculate_mse(label, pred)
    rmse = calculate_rmse(label, pred)
    r2 = calculate_r2(label, pred)
    mape_0 = calculate_mape_0(label, pred)
    mape_1 = calculate_mape_1(label, pred)

    # 返回评估指标的字典
    evaluation_results = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_0": mape_0,
        "MAPE_1": mape_1,
    }
    return evaluation_results
