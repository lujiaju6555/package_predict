from datetime import datetime

import pandas as pd
from h3 import h3
from shapely.geometry import mapping, shape

import plot_figure


# 读入数据，并查看各列缺失情况
def get_input_data(path):
    df = pd.read_csv(path)
    col = df.columns.tolist()
    for i in range(len(col)):
        print('\t', 'column_name:', col[i], '\t', '\t', 'sum of nan values:', sum(df[col[i]].isna()))
    print('\n')
    return df


# 输出POI数据大类和中类有哪些,以及数据量
def print_category(df):
    """
    参数:
    - df: dataframe POI数据集
    返回:无
    """
    print('\tprint first category and second category and their numbers...\n')
    first_cate = df.first_category.unique().tolist()
    for i in range(len(first_cate)):
        second_cate = df.loc[df.first_category == first_cate[i]].second_category.unique().tolist()
        print('\t', first_cate[i])
        for j in range(len(second_cate)):
            print('\t\t', second_cate[j], '\t',
                  len(df.loc[(df.first_category == first_cate[i]) & (df.second_category == second_cate[j])]))


def create_h3_grid(coordinates, resolution):
    """
    生成H3网格
    参数:
    - coordinates: list 区域边界坐标
    - resolution: int H3网格分辨率
    返回:
    - h3_indexes: list H3网格的编码
    """
    polygon = {"type": "Polygon", "coordinates": [coordinates]}
    shapely_polygon = shape(polygon)
    h3_indexes = list(h3.polyfill(mapping(shapely_polygon), resolution, True))

    return h3_indexes


def generate_date_dataframe(start_date_str, end_date_str):
    """
    生成时间数据，从起始时间到终止时间每个小时作为一条数据
    参数:
    - start_date_str: str 起始时间
    - end_date_str: str 终止时间
    返回:
    - dataframe（5个字段：日期、小时、月份、星期几、月的第几天）
    """
    # 将输入的日期字符串转换为datetime对象
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d %H:%M:%S')

    # 生成日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')

    # 提取相关字段
    data = {
        'timestamp': date_range,
        'day': date_range.date,
        'hour': date_range.hour,
        'month': date_range.month,
        'day_of_week': date_range.dayofweek,
        'day_of_month': date_range.day,
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    return df


def transform_poi_data(df_poi, resolution):
    """
    转换POI数据（高德地图API返回的形式）
    参数:
    - df_poi: dataframe POI数据集
    返回:
    - dataframe（5个字段：经度、纬度、大类、中类、小类）
    """
    new_data = []

    for index, row in df_poi.iterrows():
        location = row['Location']
        longitude, latitude = map(float, location.split(','))

        types = row['Type'].split('|')

        for poi_type in types:
            categories = poi_type.split(';')

            if len(categories) == 3:
                category, subcategory1, subcategory2 = categories
            elif len(categories) == 2:
                category, subcategory1 = categories
                subcategory2 = None
            else:
                category = categories[0]
                subcategory1 = subcategory2 = None

            new_data.append({
                'lng': longitude,
                'lat': latitude,
                'first_category': category,
                'second_category': subcategory1,
                'third_category': subcategory2,
            })

    new_df = pd.DataFrame(new_data)

    new_df['H3_grid'] = new_df.apply(lambda row: h3.geo_to_h3(row['lat'], row['lng'], resolution=resolution), axis=1)

    return new_df


def generate_hourly_h3_data(result_df, h3_grid):
    """
    对于每个时间，都生成各个H3网格对应的数据，即一个小时对应若干条H3网格
    参数:
    - result_df: dataframe 时间数据集
    - h3_grid: list H3网格编码
    返回:
    - dataframe（日期、小时、月份、星期几、月的第几天、H3网格编码、天气特征）
    """
    # 存储生成的数据
    new_data = []

    '''
    for _, row in result_df.iterrows():
        for h3_code in h3_grid:
            new_data.append({
                'day': row['day'],
                'hour': row['hour'],
                'month': row['month'],
                'day_of_week': row['day_of_week'],
                'day_of_month': row['day_of_month'],
                'H3_grid': h3_code,
            })
    '''
    for _, row in result_df.iterrows():
        row_dict = row.to_dict()
        for h3_code in h3_grid:
            row_dict['H3_grid'] = h3_code
            new_data.append(row_dict.copy())

    # 创建DataFrame
    new_df = pd.DataFrame(new_data)
    return new_df


# 合并天气数据和时间数据
def merge_weather_data(df_time, df_weather, weather_columns):
    # 将upTime列转换为datetime类型
    df_weather['upTime'] = pd.to_datetime(df_weather['upTime'])

    # 初始化一个空的DataFrame来存储合并后的数据
    merged_df = pd.DataFrame()

    # 遍历df_time中的每一行
    for index, row in df_time.iterrows():

        # 获取df_time中的日期、小时和月份
        day = row['day_of_month']
        hour = row['hour']
        month = row['month']

        # 检查是否和上一行相同，如果相同则跳过
        # 在df_weather中找到与df_time中日期最接近的记录
        filtered_weather = df_weather[df_weather['upTime'].dt.day == int(day)]

        # 找到最接近的时间
        hour_str = str(hour).zfill(2)  # 补零以匹配时间格式
        closest_time = filtered_weather.loc[
            (filtered_weather['upTime'] - pd.Timestamp(f'{2023}-{month}-{day}-{hour_str}')).abs().idxmin()]

        # 将找到的最接近的天气特征合并到df_time中
        merged_row = row.copy()
        for i in range(len(weather_columns)):
            merged_row[weather_columns[i]] = closest_time[weather_columns[i]]

        # 将合并后的行添加到merged_df中
        merged_df = merged_df._append(merged_row, ignore_index=True)

    return merged_df


# 获取钱塘江内网格编码
def get_river_h3_grid():
    river_h3_grid = ['88309a4007fffff',
                     '88309a4003fffff',
                     '88309a4015fffff',
                     '88309a4011fffff',
                     '88309a401bfffff',
                     '88309a40cdfffff',
                     '88309a40c9fffff',
                     '88309a40cbfffff',
                     '88309a42b5fffff',
                     '88309a42b7fffff',
                     '88309a42b1fffff',
                     '88309a42b3fffff',
                     '88309a476dfffff',
                     '88309a4769fffff',
                     '88309a4745fffff',
                     '88309a4741fffff',
                     '88309a474bfffff',
                     '88309a55b7fffff',
                     '88309a55b1fffff',
                     '88309a55b3fffff',
                     '88309a55bbfffff',
                     '88309a5597fffff',
                     '88309a4669fffff',
                     '88309a5593fffff',
                     '88309a464dfffff',
                     '88309a4641fffff',
                     '88309a4649fffff',
                     '88309a464bfffff',
                     '88309a54b5fffff',
                     '88309a0965fffff',
                     '88309a0961fffff',
                     '88309a0963fffff'
                     ]

    return river_h3_grid


# 计算第几个小时
def calc_hour_from_start(row):
    day_of_month = row['day_of_month']
    hour = row['hour']
    month = row['month']

    return (month - 8) * 744 + (day_of_month - 1) * 24 + hour


def get_POI(path_POI, resolution):
    df_POI = get_input_data(path_POI)
    df_POI = transform_poi_data(df_POI, resolution)
    pivot_table_result = pd.pivot_table(df_POI, values='lat', index=['H3_grid'], columns=['first_category'],
                                        aggfunc='count', fill_value=0) # 统计每个 H3 网格中各个 POI 大类的数量
    pivot_table_result.columns = [f'{col}_count' for col in pivot_table_result.columns] # 重命名列名，可以根据需要修改

    return pivot_table_result


def create_h3_grid_and_delete_river(coordinates, resolution):

    h3_grid = create_h3_grid(coordinates, resolution)
    # 删除江内网格
    river_h3_grid = get_river_h3_grid()
    for i in range(len(river_h3_grid)):
        if river_h3_grid[i] in h3_grid:
            h3_grid.remove(river_h3_grid[i])

    return h3_grid

def merge_parcel_data(df, df_parcel, label_col):
    for index, row in df.iterrows():
        h3_grid_i = row['H3_grid']
        hour_from_start = row['hour_from_start']
        # 检查是否有符合条件的行
        matching_rows = df_parcel.loc[df_parcel['H3_grid'] == h3_grid_i, hour_from_start]
        if not matching_rows.empty:
            parcel_value = matching_rows.values[0]
            df.at[index, label_col] = parcel_value
        else:
            df.at[index, label_col] = None  # 或者使用其他值来表示没有匹配到任何值

    # 删除Nan值
    df = df.dropna()
    # df[label_col] = df[label_col].astype('int')

    return df


def visualize_h3_grid(label_col, h3_grid_now, df_parcel, coordinates, visualize_h3_grid_path):
    parcel_list = []
    swapped_coordinates = [[coord[1], coord[0]] for coord in coordinates]
    for i in range(len(h3_grid_now)):
        h3_grid_i = h3_grid_now[i]
        matching_rows = df_parcel.loc[df_parcel['H3_grid'] == h3_grid_i, label_col]
        parcel_value = matching_rows.values[0]
        parcel_list.append(parcel_value)

    plot_figure.figure_H3_grid(h3_grid_now, parcel_list, swapped_coordinates, visualize_h3_grid_path)


def get_parcel_data(args):
    df_parcel_202308 = pd.read_excel(args.path_parcel_202308)
    df_parcel_202309 = pd.read_excel(args.path_parcel_202309)
    df_parcel = pd.merge(df_parcel_202308, df_parcel_202309, how='inner', on='H3_grid')
    df_parcel[args.label_col] = df_parcel.iloc[:, 1:args.num_of_hours].sum(axis=1)
    df_parcel.drop(
        df_parcel.loc[(df_parcel[args.label_col] < args.parcel_lower_bound) | (
                    df_parcel[args.label_col] > args.parcel_upper_bound)].index,
        inplace=True)

    return df_parcel