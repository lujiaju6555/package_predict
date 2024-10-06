import argparse
import time

import util
from util_data import get_input_data
from util_outlier import get_grid_time_series, plot_origin_parcel

parser = argparse.ArgumentParser()

# 数据路径参数
parser.add_argument('--all_data_path', type=str, default='./data/data.csv', help='未处理异常值前的数据路径')
parser.add_argument('--outlier_data_path', type=str, default='./data/outlier/', help='处理异常值后的数据路径')
parser.add_argument('--pattern_figure_path', type=str, default='./figure/pattern/', help='绘制数据模式的存放图片路径')
parser.add_argument('--metrics_path', type=str, default='./metrics/', help='保存预测结果评估指标的路径')
parser.add_argument('--plot_pred_path', type=str, default='./figure/pred/', help='绘制预测示意图的路径')
parser.add_argument('--feature_importance_figure_path', type=str, default='./figure/pred/', help='绘制特征重要性图像的路径')
parser.add_argument('--feature_importance_csv_path', type=str, default='./metrics/', help='保存特征重要性数值表格的路径')
# 字符串类型
parser.add_argument('--label_col', type=str, default='parcel', help='预测变量')
parser.add_argument('--detect_method', type=str, default='std', help='异常值检测方法')
parser.add_argument('--handle_method', type=str, default='linear', help='异常值处理方法')
parser.add_argument('--model', type=str, default='lightgbm', help='预测模型')
# 时间日期类型
parser.add_argument('--split_time_1', type=util.parse_datetime, default='2023-08-07 00:00:00', help='')
parser.add_argument('--split_time_2', type=util.parse_datetime, default='2023-09-26 00:00:00', help='')
parser.add_argument('--split_time_3', type=util.parse_datetime, default='2023-09-25 00:00:00', help='')
parser.add_argument('--split_time_4', type=util.parse_datetime, default='2023-09-24 00:00:00', help='')
# 整数类型
parser.add_argument('--num_of_hours', type=int, default=1464, help='时间序列数据小时数量')
parser.add_argument('--figure_step', type=int, default=7, help='绘制数据模式时以几天为步长')
parser.add_argument('--m_ARIMA', type=int, default=12, help='ARIMA模型的时间步长')
parser.add_argument('--forecast_start_step', type=int, default=(31 + 25) * 12, help='')
parser.add_argument('--optuna_trials', type=int, default=25, help='optuna调参实验次数')
parser.add_argument('--random_state', type=int, default=42, help='随机种子')
parser.add_argument('--step', type=int, default=1, help='数据以几小时为步长')
parser.add_argument('--input_size', type=int, default=24, help='深度学习数据处理输入步长')
parser.add_argument('--epochs', type=int, default=100, help='深度学习训练轮数')
parser.add_argument('--batch_size', type=int, default=32, help='深度学习批大小')
# 浮点数类型
parser.add_argument('--feature_importance_threshold', type=float, default=0.8, help='选择特征重要性累计占比百分之多少的特征')
# store_true类型
parser.add_argument('--if_handle_outlier', action='store_true', default=True, help='是否使用处理了异常值的数据')
parser.add_argument('--if_delete_night', action='store_true', default=False, help='是否删去夜晚数据并加至早上8点')
parser.add_argument('--if_figure_step', action='store_true', default=False, help='是否绘制快递量pattern图像')
parser.add_argument('--if_delete_grid', action='store_true', default=True, help='是否删去指定网格')
parser.add_argument('--if_hourly_outlier', action='store_true', default=False,
                    help='是否分别对每个小时的序列进行异常值处理')
# list类型
parser.add_argument('--window_len_list', type=int, nargs='+', default=[24, 168], help='统计窗口特征')
parser.add_argument('--lag_list', type=int, nargs='+',
                    default=[1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 47, 48, 49, 71, 72, 73, 95, 96, 97, 119, 120, 121, 143,
                             144, 145, 167, 168, 169], help='滞后值特征')
parser.add_argument('--trend_len_list', type=int, nargs='+',
                    default=[(1, 1, 2, 2), (1, 1, 24, 24), (1, 24, 25, 48), (1, 24, 145, 168), (1, 24, 25, 168)],
                    help='趋势特征：四元组(a,b,c,d)表示用(t-b, t-a)的均值除以(t-d, t-c)的均值')

args = parser.parse_args()


def main():
    # 读入初始数据
    if args.if_handle_outlier:
        data_path = args.outlier_data_path + args.detect_method + '_' + args.handle_method
        if args.if_hourly_outlier:
            data_path += '_hourly'
        data_path += '.csv'
    else:
        data_path = args.all_data_path
    df = get_input_data(data_path)

    # 删除手动选择的时间分布不均匀的网格
    if args.if_delete_grid:
        delete_grid = util.get_delete_grid()
        df = df[~df['H3_grid'].isin(delete_grid)]
        print('筛选后网格数量：', len(df.H3_grid.unique().tolist()))

    # 删除夜晚数据
    if args.if_delete_night:
        df = df.groupby('H3_grid').apply(lambda x: util.group_night_parcel(x, args)).reset_index(drop=True)

    # 绘制快递量图像
    if args.if_figure_step:
        util.grid_step_figure(df, args)
        grid_time_series = get_grid_time_series(df)
        for grid, data in grid_time_series.items():
            plot_origin_parcel(data, grid, args.label_col, args.pattern_figure_path + grid + '.png')

    # 对各个网格分别使用模型进行预测
    label, pred, metrics, metrics_all = util.model(df, args)
    print(df[args.label_col].describe())
    print(metrics)
    print(metrics_all)

    # 导出指标
    metrics.to_csv(args.metrics_path + args.model + '.csv', header=True, index=False, encoding='utf_8_sig')


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f} s".format(t2 - t1))
