import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def get_grid_time_series(df):
    # 按照H3_grid进行分组，并对每个分组按照hour_from_start排序
    grids = {}
    tmp = df.groupby('H3_grid')
    for grid, group in tmp:
        sorted_group = group.sort_values(by='hour_from_start')
        time_series = sorted_group.set_index('hour_from_start')['parcel']
        grids[grid] = time_series
    return grids


def plot_method_outlier(data, label_col, outliers, grid, path):
    # 重新创建折线图，并在其中标注离群点
    plt.figure(figsize=(12, 6))
    # 绘制forecast值的折线图
    plt.plot(data, label=label_col, color='blue', alpha=0.7)
    # 标注离群点
    plt.scatter(outliers.index, outliers, color='red', label='Outliers')
    # 设置标题和标签
    plt.title(grid, fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(label_col, fontsize=12)
    plt.legend()
    plt.savefig(path)
    plt.grid(True)
    plt.close()


def std_method(data, grid, label_col, path):
    mean = data.mean()
    std_dev = data.std()

    # Define the number of standard deviations a data point has to be away from the mean to be considered an outlier
    num_std_dev = 2

    # Find outliers
    outlier_bool = (data - mean > num_std_dev * std_dev) | (mean - data > num_std_dev * std_dev)
    outliers = data[outlier_bool]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=label_col, color='blue', alpha=0.7)
    plt.scatter(outliers.index, outliers, color='red', label='Outliers')
    plt.axhline(mean, color='green', linestyle='--', label='Mean')
    plt.axhline(mean + num_std_dev * std_dev, color='orange', linestyle='--', label='Upper Bound')
    if mean - num_std_dev * std_dev > 0:
        plt.axhline(mean - num_std_dev * std_dev, color='orange', linestyle='--', label='Lower Bound')

    # Enhancing the plot with a tech-style theme
    plt.style.use('ggplot')
    # 设置标题和标签
    plt.title(grid, fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(label_col, fontsize=12)
    plt.legend()
    plt.savefig(path)
    plt.grid(True)
    plt.close()

    return outlier_bool, len(outliers)


def quantile_method(data, grid, label_col, path):
    # 计算异常值
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 标识异常值
    outlier_bool = (data > upper_bound)

    # 创建折线图，并在其中标注异常值
    plt.figure(figsize=(12, 6))

    # 绘制OT值的折线图
    plt.plot(data, label=label_col, color='blue', alpha=0.7)
    plt.axhline(upper_bound, color='orange', linestyle='--', label='Upper Bound')

    # 标注异常值
    outliers = data[outlier_bool]
    plt.scatter(outliers.index, outliers, color='red', label='outlier')

    plt.style.use('ggplot')

    # 设置标题和标签
    plt.title(grid, fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(label_col, fontsize=12)
    plt.legend()
    plt.savefig(path)
    plt.close()

    # 返回异常值的界限和异常值的数量
    return outlier_bool, len(outliers)


def knn_method(data, grid, label_col, path, n_neighbors_smaller):
    # 使用K-NN模型
    knn_smaller = NearestNeighbors(n_neighbors=n_neighbors_smaller)
    knn_smaller.fit(pd.DataFrame(data.values.reshape(-1, 1), columns=['parcel']))

    # 计算每个点到其邻居的距离
    distances_smaller, _ = knn_smaller.kneighbors(pd.DataFrame(data))

    # 计算平均距离
    mean_distance_smaller = np.mean(distances_smaller, axis=1)

    # 标记离群点
    outlier_bool = np.where(mean_distance_smaller > n_neighbors_smaller)[0]
    outliers = data[outlier_bool]

    # 画图
    plot_method_outlier(data, label_col, outliers, grid, path)

    # 返回离群点的数量
    a = pd.Series([False for _ in range(len(data))])
    for i in range(len(outlier_bool)):
        a[outlier_bool[i]] = True
    outlier_bool = a

    return outlier_bool, len(outliers)


def lof_method(data, grid, label_col, path, n_neighbors):
    # 设置LoF参数
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)  # 可以调整n_neighbors的值，来调整对数据的敏感度

    # 使用LoF模型
    lof_labels = lof.fit_predict(pd.DataFrame(data.values.reshape(-1, 1), columns=[label_col]))
    lof_scores = -lof.negative_outlier_factor_  # LoF分数（负数，越小越异常）

    # 标记离群点
    outlier_bool = (lof_labels == -1)
    outliers = data[outlier_bool]

    # 画图
    plot_method_outlier(data, label_col, outliers, grid, path)

    # 返回离群点的数量
    return pd.Series(outlier_bool), len(outliers)


def dbscan_method(data, grid, label_col, path, eps, min_samples):
    # 设置DBSCAN参数
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    # 使用DBSCAN模型
    dbscan_labels = dbscan.fit_predict(pd.DataFrame(data.values.reshape(-1, 1), columns=[label_col]))

    # 标记离群点（在DBSCAN中，-1标签表示离群点）
    outlier_bool = (dbscan_labels == -1)
    outliers = data[outlier_bool]

    # 画图
    plot_method_outlier(data, label_col, outliers, grid, path)

    return pd.Series(outlier_bool), len(outliers)


def plot_origin_parcel(data, grid, label_col, path):
    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(data, label=label_col, color='blue', alpha=0.7)
    plt.title(grid, fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(label_col, fontsize=12)
    plt.legend()
    plt.savefig(path)
    plt.grid(True)
    plt.close()


def get_detect_result(data, grid, plot_path, args):
    if args.detect_method == 'std':
        return std_method(data, grid, args.label_col, plot_path)
    elif args.detect_method == 'quantile':
        return quantile_method(data, grid, args.label_col, plot_path)
    elif args.detect_method == 'knn':
        return knn_method(data, grid, args.label_col, plot_path, args.n_neighbors_smaller)
    elif args.detect_method == 'lof':
        return lof_method(data, grid, args.label_col, plot_path, args.n_neighbors_lof)
    elif args.detect_method == 'dbscan':
        return dbscan_method(data, grid, args.label_col, plot_path, args.eps, args.min_samples)


def detect_outlier(data, grid, args):
    if args.if_plot_origin_parcel:
        plot_origin_parcel(data, grid, args.label_col, args.origin_figure_path + grid + '.png')

    plot_path = args.outlier_figure_path + args.detect_method + '/' + grid

    if args.if_hourly_outlier is False:
        return get_detect_result(data, grid, plot_path + '.png', args)
    else:
        hour_outliers, outlier_bool, outlier_cnt = [], [], 0
        hours = 24 // args.step
        for hour in range(hours):
            data_hour = []
            for ind, val in enumerate(data):
                if ind % hours == hour:
                    data_hour.append(val)
            data_hour = pd.Series(data_hour)
            hour_outlier, hour_outlier_cnt = get_detect_result(data_hour, grid, plot_path + '_' + str(hour) + '.png',
                                                               args)
            outlier_cnt += hour_outlier_cnt
            hour_outliers.append(hour_outlier)
        for i in range(len(hour_outliers[0])):
            for j in range(hours):
                outlier_bool.append(hour_outliers[j][i])
        outlier_bool = pd.Series(outlier_bool)

        plot_method_outlier(data, args.label_col, data[outlier_bool], grid, plot_path + '.png')

        return outlier_bool, outlier_cnt


def get_handle_result(data, outliers, args):
    if args.handle_method == 'mean':  # 序列均值
        data[outliers] = data.mean()  # 将异常值替换为整个序列的均值
    elif args.handle_method == 'linear':  # 线性插值
        data[outliers] = float('nan')
        data = data.interpolate(method='linear')
    elif args.handle_method == 'spline':  # 样条插值
        data[outliers] = float('nan')
        data = data.interpolate(method='spline', order=args.interpolate_order)

    for i in range(len(data)):
        if data[i] < 0:
            data[i] = 0
        if np.isnan(data[i]):
            data[i] = data.mean()

    return data


def plot_handled_data(grid, data_origin, data, outliers, path, args):
    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(data_origin, label=args.label_col, color='blue', alpha=0.7)
    plt.scatter(outliers[outliers].index, data_origin[outliers], color='red', label='Outliers')
    plt.scatter(outliers[outliers].index, data[outliers], color='green', label='Handled_Outliers')
    plt.title(grid, fontsize=15)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel(args.label_col, fontsize=12)
    plt.legend()
    plt.savefig(path)
    plt.grid(True)
    plt.close()


def handle_outlier(grid, data, outliers, args):
    data_origin = data.copy()
    plot_path = args.outlier_figure_path + args.detect_method + '_' + args.handle_method + '/' + grid

    if args.if_hourly_outlier is False:
        data = get_handle_result(data, outliers, args)
        plot_handled_data(grid, data_origin, data, outliers, plot_path + '.png', args)
    else:
        hour_data, data_new = [], []
        hours = 24 // args.step
        for hour in range(hours):
            data_hour, outlier_hour, data_origin_hour = [], [], []
            for ind, val in enumerate(data):
                if ind % hours == hour:
                    data_hour.append(val)
                    outlier_hour.append(outliers[ind])
                    data_origin_hour.append(data_origin[ind])

            data_hour, outlier_hour, data_origin_hour = pd.Series(data_hour), pd.Series(outlier_hour), pd.Series(
                data_origin_hour)
            handled_data = get_handle_result(data_hour, outlier_hour, args)
            hour_data.append(handled_data)
            plot_handled_data(grid, data_origin_hour, pd.Series(handled_data), outlier_hour,
                              plot_path + str(hour) + '.png', args)

        for i in range(len(hour_data[0])):
            for j in range(hours):
                data_new.append(hour_data[j][i])
        data = pd.Series(data_new)
        plot_handled_data(grid, data_origin, data, outliers, plot_path + '.png', args)

    return data


def update_original_df(df, processed_data):
    # 处理好的数据是以H3_grid为键，时间序列为值的字典
    for grid, series in processed_data.items():
        df.loc[(df['H3_grid'] == grid) & (df['hour_from_start'].isin(series.index)), 'parcel'] = series.values
    return df
