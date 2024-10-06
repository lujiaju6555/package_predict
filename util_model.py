import warnings

import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
import pandas as pd

from nbeats_forecast import NBeats
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from torch import optim

from calc_metrics import *
from set_feature import get_feature_columns

warnings.filterwarnings("ignore")


# arima模型
def arima(data, split_time, args):
    label, pred = [], []
    train = data[data.index < split_time]
    test = data[data.index >= split_time]
    model = auto_arima(train, seasonal=True, m=args.m_ARIMA, trace=True)
    model.fit(train)
    forecast = model.predict(n_periods=len(test))
    for i in range(len(test)):
        label.append(test[i])
        pred.append(forecast[i])

    return label, pred


# N-BEATS模型
def nbeats(data, args):
    label, pred = [], []

    # 将数据转换为numpy数组
    train = data[data.index < args.split_time_2].values
    test = data[data.index >= args.split_time_4].values

    train = train.reshape(-1, 1)
    test = test.reshape(-1, 1)

    # 初始化N-BEATS模型
    model = NBeats(data=train, period_to_forecast=1, backcast_length=args.input_size)

    # 训练模型
    model.fit(epoch=args.epochs,
              optimiser=optim.AdamW(model.parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                                    amsgrad=False))

    # 预测
    for i in range(len(test) - args.input_size):
        input_seq = test[i:i + args.input_size].flatten()
        forecast = model.predict(input_seq)  # 使用模型进行预测
        label.append(test[i + args.input_size])  # 真实值
        pred.append(forecast)  # 预测值

    return pd.Series(label), pd.Series(pred)


def create_inout_sequences(data, input_size, output_size):
    inout_seq = []
    for i in range(len(data) - input_size - output_size + 1):
        inout_seq.append((data[i:i + input_size], data[i + input_size:i + input_size + output_size]))
    return zip(*inout_seq)


# lightgbm模型
class TimeSeriesLightGBM:
    def __init__(self, categorical_features):
        self.model = None
        self.categorical_features = categorical_features

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, train_data=None, test_data=None, params=None,
            optuna_trials=100):
        if params is None:
            params = self._optimize_params(X_train, y_train, X_valid, y_valid, train_data, test_data,
                                           trials=optuna_trials)

        self.model = lgb.train(params, train_data, 100, valid_sets=[test_data],
                               categorical_feature=self.categorical_features)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        return self.model.predict(X_test)

    def _objective(self, trial, X_train, y_train, X_valid, y_valid, train_data, test_data):
        params = {
            'objective': 'regression',
            'metric': trial.suggest_categorical('metric', ['huber', 'mae', 'mse']),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'max_depth': trial.suggest_int('max_depth', 5, 20),  # 3-6  5-20 10-40
            'num_leaves': trial.suggest_int('num_leaves', 64, 256),  # 8-16  64-256 64-1024
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1.0),
            'verbose': -1
        }

        model = lgb.train(params, train_data, 100, valid_sets=[test_data],
                          categorical_feature=self.categorical_features)

        predictions = model.predict(X_valid)

        mse = calculate_mse(y_valid, predictions)
        mae = calculate_mae(y_valid, predictions)

        return mae

    def _optimize_params(self, X_train, y_train, X_valid, y_valid, train_data, test_data, trials=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self._objective(trial, X_train, y_train, X_valid, y_valid, train_data, test_data),
                       n_trials=trials)

        best_params = study.best_params
        print(f"Best params: {best_params}")
        return best_params


# catboost模型
class TimeSeriesCatBoost:
    def __init__(self, categorical_features):
        self.model = None
        self.categorical_features = categorical_features

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, train_data=None, test_data=None, params=None,
            optuna_trials=100):
        if params is None:
            params = self._optimize_params(X_train, y_train, X_valid, y_valid, train_data, test_data,
                                           trials=optuna_trials)

        self.model = cb.CatBoostRegressor(**params)
        self.model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=self.categorical_features, verbose=0)

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        return self.model.predict(X_test)

    def _objective(self, trial, X_train, y_train, X_valid, y_valid, train_data, test_data):
        params = {
            'objective': 'RMSE',
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'loss_function': trial.suggest_categorical('loss_function', ['RMSE', 'MAE']),
            'verbose': 0
        }

        model = cb.CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=self.categorical_features, verbose=0)

        predictions = model.predict(X_valid)

        mse = calculate_mse(y_valid, predictions)
        mae = calculate_mae(y_valid, predictions)

        return mae

    def _optimize_params(self, X_train, y_train, X_valid, y_valid, train_data, test_data, trials=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self._objective(trial, X_train, y_train, X_valid, y_valid, train_data, test_data),
                       n_trials=trials)

        best_params = study.best_params
        print(f"Best params: {best_params}")
        return best_params


def get_data(df, features, args):
    feature_and_label_columns = features['numeric_columns'].copy() + features['categorical_columns'].copy()

    if args.label_col not in feature_and_label_columns:
        feature_and_label_columns.append(args.label_col)

    df[args.label_col] = df[args.label_col].astype(int)

    train_data = df.loc[(df.day.dt.date <= args.split_time_3.date()) & (df.day.dt.date > args.split_time_1.date())][
        feature_and_label_columns]
    test_data = df.loc[df.day.dt.date >= args.split_time_2.date()][feature_and_label_columns]

    for ind, col in enumerate(features['categorical_columns']):
        if args.model == 'lightgbm':
            train_data[col] = train_data[col].astype('category')
            test_data[col] = test_data[col].astype('category')
        else:
            train_data[col] = train_data[col].astype(str)
            test_data[col] = test_data[col].astype(str)

    X_train, y_train, X_test, y_test = (train_data.drop(columns=args.label_col), train_data[args.label_col],
                                        test_data.drop(columns=args.label_col), test_data[args.label_col])

    return X_train, y_train, X_test, y_test


# 生成移动窗口特征（前几天的各个统计指标）
def get_window_feature(df, H3_grid, col_list, time_step_list, features):
    for len_of_window in time_step_list:
        for col in col_list:
            shifted_dfs = []
            for ind, h3_grid in enumerate(H3_grid):
                df_i = df.loc[df['H3_grid'] == h3_grid].copy()
                df_i = df_i.sort_values(by='hour_from_start', ascending=True)
                hour_from_start_index = df_i.hour_from_start.tolist()
                shifted = df_i[col].rolling(window=len_of_window, min_periods=0).agg(
                    ['mean', 'std', 'min', 'max']).reset_index(drop=True)
                shifted.drop(index=len(shifted) - 1, inplace=True)
                shifted = pd.DataFrame(np.insert(shifted.values, 0, values=[0, 0, 0, 0], axis=0))
                shifted.columns = [f'{col}_{len_of_window}_mean', f'{col}_{len_of_window}_std',
                                   f'{col}_{len_of_window}_min', f'{col}_{len_of_window}_max']
                shifted['hour_from_start'] = hour_from_start_index
                shifted['H3_grid'] = h3_grid
                shifted_dfs.append(shifted)

            # 加入数值型变量列
            for ind, val in enumerate(shifted.columns):
                if val == 'H3_grid' or val == 'hour_from_start':
                    continue
                features['numeric_columns'].append(val)

            result_df = pd.concat(shifted_dfs, ignore_index=True)
            df = pd.merge(df, result_df, on=['H3_grid', 'hour_from_start'], how='left')

    return df, features


# 生成滞后指特征
def get_lag_feature(df, H3_grid, col_list, lag_steps, features):
    for col in col_list:
        for lag_step in lag_steps:
            lagged_dfs = pd.DataFrame(columns=['H3_grid', 'hour_from_start', f'{col}_lag_{lag_step}'])
            for h3_grid in H3_grid:
                df_i = df.loc[df['H3_grid'] == h3_grid].copy()
                df_i = df_i.sort_values(by='hour_from_start', ascending=True)
                lagged = df_i[col].shift(lag_step)
                lagged = lagged.rename(f'{col}_lag_{lag_step}')
                tmp = df_i[['H3_grid', 'hour_from_start']].copy()
                tmp = tmp.join(lagged)
                lagged_dfs = lagged_dfs._append(tmp)
            df = pd.merge(df, lagged_dfs, on=['H3_grid', 'hour_from_start'], how='left')
            features['numeric_columns'].append(f'{col}_lag_{lag_step}')

    return df, features


# 生成网格整体统计指标（训练集数据split_time_3）
def get_grid_feature(df, H3_grid, col_list, features, args):
    for col in col_list:
        data_all = pd.DataFrame(
            columns=['H3_grid', col + '_H3_mean', col + '_H3_std', col + '_H3_min', col + '_H3_max'])
        for ind, h3_grid in enumerate(H3_grid):
            df_i = df.loc[df['H3_grid'] == h3_grid].copy()
            data = df_i.drop(df_i.loc[df_i.timestamp < args.split_time_3].index)[col].agg(['mean', 'std', 'min', 'max'])
            data_all = data_all._append(
                {'H3_grid': h3_grid, col + '_H3_mean': data['mean'], col + '_H3_std': data['std'],
                 col + '_H3_min': data['min'], col + '_H3_max': data['max']}, ignore_index=True)

        for ind, val in enumerate(data_all.columns):
            if val == 'H3_grid':
                continue
            features['numeric_columns'].append(val)

        df = df.merge(data_all, on=['H3_grid'], how='left')

    return df, features


def calc_trend(a, l1, r1, l2, r2):
    b = a.copy()
    for i in range(r2):
        b[i] = None
    for i in range(r2, len(b)):
        mean_a = a[i - r2: i - l2 + 1].mean()
        mean_b = a[i - r1: i - l1 + 1].mean()
        b[i] = mean_b / mean_a if mean_a != 0 else None
    return b


def get_trend_feature(df, H3_grid, col_list, features, trend_window_len_list):
    for col in col_list:
        for a, b, c, d in trend_window_len_list:
            col_name = f'{col}_trend_{a}_{b}_{c}_{d}'
            df_trend = pd.DataFrame(columns=['H3_grid', 'hour_from_start', col_name])
            for ind, h3_grid in enumerate(H3_grid):
                df_i = df.loc[df['H3_grid'] == h3_grid].copy()
                df_i = df_i.sort_values(by='hour_from_start', ascending=True)
                trend = calc_trend(df_i[col], a, b, c, d)
                trend = trend.rename(col_name)
                tmp = df_i[['H3_grid', 'hour_from_start']].copy()
                tmp = tmp.join(trend)
                df_trend = df_trend._append(tmp)
            df = pd.merge(df, df_trend, on=['H3_grid', 'hour_from_start'], how='left')
            features['numeric_columns'].append(col_name)

    return df, features


def get_feature(df, features, args):
    h3_grid = df['H3_grid'].unique().tolist()
    # 数值型天气特征的窗口特征生成（由于所有网格天气序列相同，所以不生成网格整体特征）
    df, features = get_window_feature(df, h3_grid, features['numeric_weather_columns'], args.window_len_list, features)
    df, features = get_lag_feature(df, h3_grid, features['numeric_weather_columns'], args.lag_list, features)

    # 删去夜晚数据
    if args.if_delete_night:
        df = df.drop(df.loc[(df.hour > 19) | (df.hour < 8)].index)

    # 快递量的窗口特征生成
    df, features = get_window_feature(df, h3_grid, [args.label_col], args.window_len_list, features)
    df, features = get_lag_feature(df, h3_grid, [args.label_col], args.lag_list, features)

    # 快递量的网格整体特征生成
    df, features = get_grid_feature(df, h3_grid, [args.label_col], features, args)

    # 生成趋势特征
    df, features = get_trend_feature(df, h3_grid, [args.label_col], features, args.trend_len_list)

    return df, features


def train_and_evaluate_lightgbm(X_train, y_train, X_test, y_test, categorical_features, optuna_trials, random_state):
    # 划分验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1 / 9, random_state=random_state)

    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
    test_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=categorical_features)

    # 创建模型实例
    lgb_model = TimeSeriesLightGBM(categorical_features=categorical_features)

    # 用训练数据拟合模型（包含超参数优化）
    lgb_model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, train_data=train_data, test_data=test_data,
                  optuna_trials=optuna_trials)

    # 查看训练集效果
    pred = lgb_model.predict(X_train)
    metrics_train = calculate_metrics(y_train, pred)

    # 查看测试集效果
    pred = lgb_model.predict(X_test)
    pred = np.clip(pred, 0, None)
    metrics_test = calculate_metrics(y_test, pred)

    return metrics_test, lgb_model, y_test, pred


def train_and_evaluate_catboost(X_train, y_train, X_test, y_test, categorical_features, optuna_trials, random_state):
    # 划分验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1 / 9, random_state=random_state)

    # 创建模型实例
    catboost_model = TimeSeriesCatBoost(categorical_features=categorical_features)

    # 用训练数据拟合模型（包含超参数优化）
    catboost_model.fit(X_train, y_train, X_valid=X_valid, y_valid=y_valid, optuna_trials=optuna_trials)

    # 查看训练集效果
    pred_train = catboost_model.predict(X_train)
    metrics_train = calculate_metrics(y_train, pred_train)

    # 查看测试集效果
    pred_test = catboost_model.predict(X_test)
    pred_test = np.clip(pred_test, 0, None)
    metrics_test = calculate_metrics(y_test, pred_test)

    return metrics_test, catboost_model, y_test, pred_test


def feature_importance_select(Model, args):
    # 获取特征重要性
    if args.model == 'lightgbm':
        feature_importance = Model.model.feature_importance()
        feature_names = Model.model.feature_name()
    else:
        feature_importance = Model.model.feature_importances_
        feature_names = Model.model.feature_names_

    feature_importance_dict = dict(zip(feature_names, feature_importance))
    # 对特征重要性进行排序
    sorted_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    # 提取排序后的特征名称和重要性
    sorted_feature_names = list(sorted_importance_dict.keys())
    sorted_feature_importance = list(sorted_importance_dict.values())
    df_feature_importance = pd.DataFrame({'Feature_Name': sorted_feature_names,
                                          'Feature_Importance': sorted_feature_importance})
    # 计算特征重要性总和
    total_importance = np.sum(df_feature_importance.Feature_Importance)
    # 计算每个特征的相对重要性
    relative_importance = df_feature_importance.Feature_Importance / total_importance
    # 对相对重要性进行排序
    sorted_indices = np.argsort(relative_importance)[::-1]
    # 计算累积相对重要性
    cumulative_importance = np.cumsum(relative_importance[sorted_indices])
    threshold_index = np.argmax(cumulative_importance >= args.feature_importance_threshold)
    # 使用阈值选择特征
    selected_feature_indices = sorted_indices[:threshold_index + 1]
    selected_features_columns = [df_feature_importance.iloc[i].Feature_Name for i in selected_feature_indices]
    # 输出选择的特征
    print("选择的特征为:", selected_features_columns)
    print('选择的特征数量：', len(selected_features_columns))
    print('删去的特征数量：', len(df_feature_importance) - len(selected_features_columns))

    return selected_features_columns, sorted_feature_names, sorted_feature_importance


def figure_importance_new(Model, args, grid):
    if args.model == 'lightgbm':
        feature_importance = Model.model.feature_importance()
        feature_names = Model.model.feature_name()
    else:
        feature_importance = Model.model.feature_importances_
        feature_names = Model.model.feature_names_

    feature_importance_dict = dict(zip(feature_names, feature_importance))
    # 对特征重要性进行排序
    sorted_importance_dict = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))
    # 提取排序后的特征名称和重要性
    sorted_feature_names = list(sorted_importance_dict.keys())
    sorted_feature_importance = list(sorted_importance_dict.values())
    # 绘制柱状图
    plt.figure(figsize=(50, 50))
    plt.barh(sorted_feature_names, sorted_feature_importance)
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.savefig(args.feature_importance_figure_path + args.model + '/' + grid + '_feature_importance.png')
    plt.close()
    # 创建DataFrame
    df_feature_importance = pd.DataFrame({'Feature_Name': sorted_feature_names,
                                          'Feature_Importance': sorted_feature_importance})
    # 导出DataFrame为CSV文件
    df_feature_importance.to_csv(args.feature_importance_csv_path + args.model + '/' + grid + '_feature_importance.csv',
                                 index=False, encoding='utf_8_sig')


def get_feature_columns_from_set_feature():
    # 获取特征
    feature_cols, categorical_cols, numeric_cols, weather_cols = get_feature_columns()
    numeric_weather_cols = [col for col in weather_cols if col in numeric_cols]
    features = {'feature_columns': feature_cols,
                'categorical_columns': categorical_cols,
                'numeric_columns': numeric_cols,
                'weather_columns': weather_cols,
                'numeric_weather_columns': numeric_weather_cols}

    return features


def select_features(features):
    tmp = {'feature_columns': [], 'categorical_columns': [], 'numeric_columns': [],
           'weather_columns': [], 'numeric_weather_columns': []}
    for fea in features:
        for col in features[fea]:
            if 'count' not in col and 'H3' not in col:
                tmp[fea].append(col)

    return tmp


def tree_model(df, args):
    grid = df['H3_grid'].unique()[0]
    df['day'] = pd.to_datetime(df['day'])

    # 处理特征
    features = get_feature_columns_from_set_feature()
    df, features = get_feature(df, features, args)
    features = select_features(features)

    # 划分数据集
    X_train, y_train, X_test, y_test = get_data(df, features, args)

    # 训练评估并输出
    if args.model == 'lightgbm':
        func = train_and_evaluate_lightgbm
    else:
        func = train_and_evaluate_catboost

    metrics, model, label, pred = func(X_train, y_train, X_test, y_test, features['categorical_columns'],
                                       args.optuna_trials, args.random_state)

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df.to_csv(args.metrics_path + args.model + '/all_features.csv', index=True, encoding='utf_8_sig')

    # 根据特征重要性进行特征选择
    important_features, sorted_feature_names, sorted_feature_importance = feature_importance_select(model, args)
    categorical_columns_new = [col for col in features['categorical_columns'] if col in important_features]
    X_train_new = X_train[important_features]
    X_test_new = X_test[important_features]

    # 使用新的数据集重新训练一个LightGBM模型
    metrics_new, model_new, label_new, pred_new = func(X_train_new, y_train, X_test_new,
                                                       y_test,
                                                       categorical_columns_new,
                                                       args.optuna_trials,
                                                       args.random_state)
    metrics_df = pd.DataFrame.from_dict(metrics_new, orient='index', columns=['Value'])
    metrics_df.to_csv(args.metrics_path + args.model + '/select_features.csv', index=True, encoding='utf_8_sig')

    # 画特征重要性图
    figure_importance_new(model_new, args, grid)

    return label_new, pred_new
