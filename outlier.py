import argparse
import time

import pandas as pd

import util
import util_data
import util_outlier

parser = argparse.ArgumentParser()

# 数据路径参数
parser.add_argument('--all_data_path', type=str, default='./data/data.csv', help='')
parser.add_argument('--origin_figure_path', type=str, default='./figure/origin/', help='')
parser.add_argument('--outlier_figure_path', type=str, default='./figure/outlier/', help='')
parser.add_argument('--outlier_handled_data_path', type=str, default='./data/outlier/', help='')
# 字符串类型
parser.add_argument('--label_col', type=str, default='parcel', help='')
parser.add_argument('--detect_method', type=str, default='std', help='')
parser.add_argument('--handle_method', type=str, default='linear', help='')
# 整数类型
parser.add_argument('--num_of_hours', type=int, default=1464, help='')
parser.add_argument('--n_neighbors_smaller', type=int, default=3, help='')
parser.add_argument('--interpolate_order', type=int, default=2, help='')
parser.add_argument('--n_neighbors_lof', type=int, default=10, help='')
parser.add_argument('--min_samples', type=int, default=5, help='')
parser.add_argument('--step', type=int, default=1, help='是否将1小时单位的数据聚合成2小时')
# 浮点数类型
parser.add_argument('--eps', type=float, default=0.05, help='')
# store_true类型
parser.add_argument('--if_hourly_outlier', action='store_true', default=False,
                    help='是否分别对每个小时的序列进行异常值处理')
parser.add_argument('--if_plot_origin_parcel', action='store_true', default=False, help='是否绘制初始图像')

args = parser.parse_args()


def main():
    # 获取数据并生成每个网格的时间序列数据
    df = util_data.get_input_data(args.all_data_path)
    if args.step > 1:
        df = util.agg_hour(df, args)
    grid_time_series = util_outlier.get_grid_time_series(df)
    df_outlier = pd.DataFrame(columns=['grid', 'outlier_cnt'])

    # 对每个网格的时间序列应用异常检测
    outliers = {}
    for grid, series in grid_time_series.items():
        outliers[grid], outlier_cnt = util_outlier.detect_outlier(series, grid, args)
        if grid == '88309a5483fffff':
            df_tmp = pd.DataFrame(outliers[grid])
            df_tmp.to_csv('tmp.csv')
        df_outlier = df_outlier._append({'grid': grid, 'outlier_cnt': outlier_cnt}, ignore_index=True)

    # 将异常值处理后的时间序列替换到原数据中
    processed_data = {}
    for grid, series in grid_time_series.items():
        processed_data[grid] = util_outlier.handle_outlier(grid, series, outliers[grid], args)
        if grid == '88309a5483fffff':
            df_tmp = pd.DataFrame(processed_data[grid])
            df_tmp.to_csv('tmp2.csv')

    # 这句话是速度的瓶颈：
    df = util_outlier.update_original_df(df, processed_data)

    # 导出处理好的数据
    csv_path = args.outlier_handled_data_path + args.detect_method + '_' + args.handle_method
    if args.if_hourly_outlier:
        csv_path += '_hourly'
    df.to_csv(csv_path + '.csv', header=True, index=False, encoding='utf_8_sig')
    df_outlier.to_csv(csv_path + '_cnt.csv', header=True, index=False, encoding='utf_8_sig')


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f} s".format(t2 - t1))
