import argparse
import time

import pandas as pd

import set_feature
import util_data

parser = argparse.ArgumentParser()

# 数据路径参数
parser.add_argument("--path_POI", type=str, default="./data/POI_2023.csv", help="")
parser.add_argument("--path_outline", type=str, default="./data/lat_and_lng_of_outline.xlsx", help="")
parser.add_argument("--path_weather", type=str, default="./data/weather.csv", help="")
parser.add_argument("--path_parcel_202308", type=str, default="./data/202308_parcel_demand.xlsx", help="",)
parser.add_argument("--path_parcel_202309", type=str, default="./data/202309_parcel_demand.xlsx", help="",)
parser.add_argument("--visualize_h3_grid_path", type=str, default="./figure/H3/筛选后的H3网格.html", help="",)
parser.add_argument("--all_data_path", type=str, default="./data/data.csv", help="")
# 字符串类型
parser.add_argument("--start_date_str", type=str, default="2023-08-01 00:00:00", help="")
parser.add_argument("--end_date_str", type=str, default="2023-09-30 23:00:00", help="")
parser.add_argument("--label_col", type=str, default="parcel", help="")
# 整数类型
parser.add_argument("--num_of_hours", type=int, default=1464, help="")
parser.add_argument("--resolution", type=int, default=8, help="")
parser.add_argument("--parcel_lower_bound", type=int, default=1464, help="")
parser.add_argument("--parcel_upper_bound", type=int, default=146400, help="")
# store_true类型
parser.add_argument("--if_visualize_h3_grid", action="store_true", default=True, help="whether to visualize H3 grid",)
# 小数类型
# parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')

args = parser.parse_args()


def main():

    feature_columns, categorical_columns, numeric_columns, weather_columns = (set_feature.get_feature_columns())

    # 读入POI数据
    pivot_table_result = util_data.get_POI(args.path_POI, args.resolution)

    # 读入区域轮廓经纬度数据
    df_outline = pd.read_excel(args.path_outline)
    coordinates = df_outline.values.tolist()
    h3_grid = util_data.create_h3_grid_and_delete_river(coordinates, args.resolution)

    # 时间和天气数据
    df_time = util_data.generate_date_dataframe(args.start_date_str, args.end_date_str)
    df_weather = util_data.get_input_data(args.path_weather)
    df_time = util_data.merge_weather_data(df_time, df_weather, weather_columns)  # 合并时间、天气
    df_time_and_h3 = util_data.generate_hourly_h3_data(df_time, h3_grid)  # 合并时间、天气、H3网格

    # 合并数据集
    df = pd.merge(df_time_and_h3, pivot_table_result, on="H3_grid", how="left")
    df["hour_from_start"] = df.apply(util_data.calc_hour_from_start, axis=1)
    df["hour_from_start"] = df["hour_from_start"].astype(int)

    # 读入快递量数据
    df_parcel = util_data.get_parcel_data(args)

    # 合并快递量数据和时间、天气、网格
    df = util_data.merge_parcel_data(df, df_parcel, args.label_col)
    h3_grid_now = df.H3_grid.unique().tolist()
    print("筛选后网格数量为：", len(h3_grid_now))

    # 可视化筛选的网格
    if args.if_visualize_h3_grid:
        util_data.visualize_h3_grid(
            args.label_col,
            h3_grid_now,
            df_parcel,
            coordinates,
            args.visualize_h3_grid_path,
        )

    # 导出处理好的数据
    df.to_csv(args.all_data_path, header=True, index=False, encoding="utf_8_sig")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f} s".format(t2 - t1))
