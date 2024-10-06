import warnings
from datetime import datetime

import matplotlib.pyplot as plt

from calc_metrics import *
from util_model import arima, tree_model, nbeats

warnings.filterwarnings("ignore")


def parse_datetime(value):
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')


def group_night_parcel(df, args):
    for i in range(8, args.num_of_hours, 24):
        l = i - 12
        if (l < 0):
            l = 0
        night_parcel_value = df.loc[(df['hour_from_start'] >= l) & (df['hour_from_start'] < i), args.label_col].sum()
        df.loc[df['hour_from_start'] == i, args.label_col] = df.loc[df[
                                                                        'hour_from_start'] == i, args.label_col] + night_parcel_value
    return df


# 单独各个网格画图
def grid_step_figure(df, args):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    step = args.figure_step

    for ind, val in enumerate(df['H3_grid'].unique().tolist()):
        df_i = df.loc[df.H3_grid == val].copy()
        if args.if_delete_night:
            df_i = df_i.drop(df_i.loc[(df_i.hour < 8) | (df_i.hour > 19)].index)
        df_i = df_i.sort_values(by='hour_from_start', ascending=True)
        hours = 24 // args.step if args.if_delete_night is False else 12 // args.step
        for i in range(0, len(df_i), hours * step):
            fig, ax = plt.subplots(figsize=(14, 4))
            r = i + hours * step
            if (r > len(df_i)):
                r = len(df_i) - 1
            plt.plot(range(i, r), df_i.iloc[i:r][args.label_col])
            ax.vlines(range(i, r, hours), 0, max(df_i.iloc[i:r][args.label_col]), linestyles='dashed', colors='red')
            plt.xticks(range(i, r), df_i.iloc[i:r]['hour'])
            plt.title('网格 ' + val + ' 第' + str(int(i / hours) + 1) + '-' + str(int(i / hours) + step) + '天')
            plt.savefig(args.pattern_figure_path + val + '第%d天' % (int(i / hours) + 1) + '-第%d天.png' % (
                        int(i / hours) + step))
            plt.close()


# 设置手动删除的网格：时间分布明显不均匀
def get_delete_grid():
    delete_grid = ['88309a55d5fffff',
                   '88309a56adfffff',
                   '88309a428bfffff',
                   '88309a0941fffff',
                   '88309a5591fffff',
                   # 以下为效果不好的网格
                   '88309a096dfffff',
                   '88309a5483fffff',
                   '88309a54cdfffff',
                   '88309a5585fffff',
                   '88309a5491fffff',

                   ]

    return delete_grid


def plot_pred(grid, label, pred, args):
    if isinstance(label, pd.Series):
        label = np.array(label)

    if args.if_delete_night:
        line_step, start_time, end_time = 12, 8, 20
    else:
        line_step, start_time, end_time = 24, 0, 24

    fig, ax = plt.subplots(figsize=(16, 4))
    plt.title(grid)
    plt.plot(label, color='blue', label='label')
    plt.plot(pred, color='red', label='pred')
    ax.vlines(range(line_step % 24, len(pred), line_step // args.step), 0, max(max(label), max(pred)),
              linestyles='dashed', colors='green')

    # 生成横坐标
    x = []
    t = start_time
    for i in range(len(label)):
        x.append(t)
        t += args.step
        if t == end_time:
            t = start_time

    plt.xticks(range(len(x)), x)

    plt.legend()
    plt.savefig(args.plot_pred_path + args.model + '/' + grid + '.png')
    plt.close()

    res = calculate_metrics(label, pred)
    return res


def add_list(x, y, a, b):
    a = a.tolist()
    b = b.tolist()
    for i in range(len(a)):
        x.append(a[i])
        y.append(b[i])
    return x, y


def agg_hour(df, args):
    # 合并 hour_from_start
    df['hour_group'] = df['hour_from_start'] // args.step

    # 聚合数据
    df = df.groupby(['H3_grid', 'hour_group']).agg(
        parcel=('parcel', 'sum'),
        **{col: (col, lambda x: x.iloc[0]) for col in df.columns if col != 'parcel'}
    ).reset_index(drop=True)

    df['hour_from_start'] = df['hour_group']

    return df


def model(df, args):
    label_all, pred_all = [], []
    metrics = pd.DataFrame(columns=['grid', 'mse', 'rmse', 'mape0', 'mape1', 'R2', 'mean', 'std'])

    # 处理时间特征
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['half'] = df['hour'].apply(lambda x: 1 if x < 12 else 0)
    day_names = df['timestamp'].dt.day_name()
    df['is_weekend'] = day_names.apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

    count = 0
    for grid in df['H3_grid'].unique().tolist():
        # grid = '88309a4299fffff'
        # grid = '88309a0b25fffff'
        count += 1
        if count == 333:
            break

        grid_data = df[df['H3_grid'] == grid].sort_values(by='hour_from_start')

        # 模型
        if args.model == 'arima':
            data = pd.Series(grid_data['parcel'].values, index=grid_data['timestamp'])
            label, pred = arima(data, args.split_time_2, args)
        elif args.model in ('lightgbm', 'catboost'):
            label, pred = tree_model(grid_data, args)
        elif args.model == 'nbeats':
            data = pd.Series(grid_data['parcel'].values, index=grid_data['timestamp'])
            label, pred = nbeats(data, args)

        label_all, pred_all = add_list(label_all, pred_all, label, pred)

        # 画图+计算指标
        res = plot_pred(grid, label, pred, args)
        pd.DataFrame(data={'label': label, 'pred': pred}).to_csv('./metrics/lightgbm/' + grid + '.csv', header=True,
                                                                 index=False, encoding='utf_8_sig')
        metrics = metrics._append({'grid': grid, 'mse': res['MSE'], 'rmse': res['RMSE'], 'mape0': res['MAPE_0'],
                                   'mape1': res['MAPE_1'], 'R2': res['R2'], 'mean': grid_data[args.label_col].mean(),
                                   'std': grid_data[args.label_col].std()}, ignore_index=True)

    metrics_all = calculate_metrics(label_all, pred_all)
    metrics_all = pd.DataFrame.from_dict(metrics_all, orient='index', columns=['Value'])

    return label_all, pred_all, metrics, metrics_all
