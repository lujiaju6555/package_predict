import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap
from h3 import h3
from sklearn.metrics import r2_score


# 绘制预测值和真实值比较曲线图
def plot_pred_curve(label, pred, title, a, b, path):
    plt.title(title)
    plt.plot(np.arange(b - a), label[a:b], color='blue', label='label')
    plt.plot(np.arange(b - a), pred[a:b], color='red', label='pred')
    plt.legend()
    plt.savefig(path)
    plt.show()


# H3网格地理位置图
def figure_H3_grid(h3_grid, values, coordinates, name):
    # 创建 folium 地图对象
    mymap = folium.Map(location=[30.2590, 120.1443], zoom_start=13)
    # 将区域边界可视化在地图上
    folium.Polygon(locations=coordinates, color='red', fill=False, fill_opacity=0.2).add_to(mymap)
    # 循环处理每个 H3 网格
    for i in range(len(h3_grid)):
        h3_cell = h3_grid[i]
        geometry = h3.h3_to_geo_boundary(h3_cell)
        # 计算网格中心点
        center_lat = sum(coord[0] for coord in geometry) / len(geometry)
        center_lng = sum(coord[1] for coord in geometry) / len(geometry)

        if len(values) > 0:
            value = values[i]
        # 绘制网格
        polygon = folium.Polygon(locations=geometry, color='blue', fill=True, fill_color='blue', fill_opacity=0.2)
        mymap.add_child(polygon)

        # 添加网格标签
        html = f'<div style="font-size: 18px; font-weight: bold;">{h3_cell}'
        if len(values) > 0:
            html += f'<div style="font-size: 18px;">{value}</div></div>'
        label = folium.Marker(location=(center_lat, center_lng),
                              icon=folium.DivIcon(html=html))
        mymap.add_child(label)
    # 保存地图为 HTML 文件并在浏览器中打开
    mymap.save(name)


# 变量聚合后统计图
def figure_grouped(grouped_df, metric, group_column, figsize):
    # 对聚合后的DataFrame按照均值从小到大进行排序
    sorted_df = grouped_df.sort_values(by=metric)
    # 计算每个H3网格的数量
    sorted_df['count'] = grouped_df.groupby(group_column).size()
    # 计算累计频率
    sorted_df['cumulative_frequency'] = sorted_df['count'].cumsum() / sorted_df['count'].sum()

    # 绘制散点图并连接线
    plt.figure(figsize=figsize)
    plt.plot(sorted_df[metric], sorted_df['cumulative_frequency'], marker='o', linestyle='-')
    plt.title('Scatter Plot of Mean Parcel with Connected Lines')
    plt.xlabel(metric + ' Parcel')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True)

    # 设置x轴坐标范围和步长
    x_ticks_1 = np.arange(0, 5, 1)
    x_ticks_2 = np.arange(5, 31, 5)
    x_ticks_3 = np.arange(100, sorted_df[metric].max() + 1, 50)
    x_ticks = np.concatenate([x_ticks_1, x_ticks_2, x_ticks_3])

    plt.xticks(x_ticks)
    plt.tight_layout()
    plt.savefig('./figure/group/' + group_column + '_plot.png')
    plt.show()

    # 绘制直方图
    plt.figure(figsize=figsize)
    plt.hist(sorted_df[metric], bins=x_ticks, color='skyblue', edgecolor='black')
    plt.title('Histogram of ' + metric + ' Parcel with Customized Bins')
    plt.xlabel(metric + ' Parcel')
    plt.ylabel('Frequency')
    plt.grid(True)
    # 添加刻度标签
    plt.xticks(x_ticks)
    plt.tight_layout()
    plt.savefig('./figure/group/' + group_column + '_hist.png')
    plt.show()

    # 计算每个段的个数和占比
    segment_counts = []
    segment_ratios = []
    for i in range(len(x_ticks) - 1):
        lower_bound = x_ticks[i]
        upper_bound = x_ticks[i + 1]
        segment_data = sorted_df[(sorted_df[metric] >= lower_bound) & (sorted_df[metric] < upper_bound)]
        segment_count = len(segment_data)
        segment_ratio = segment_count / len(sorted_df)
        segment_counts.append(segment_count)
        segment_ratios.append(segment_ratio)
    # 输出统计结果表
    segment_df = pd.DataFrame({
        'Lower Bound': x_ticks[:-1],
        'Upper Bound': x_ticks[1:],
        'Count': segment_counts,
        'Ratio': segment_ratios
    })

    return segment_df


# 聚合后统计图2
def figure_grouped_2(grouped_df, group_column, figsize):
    # 绘制散点图并连接线
    plt.figure(figsize=figsize)
    plt.plot(grouped_df.index, grouped_df['mean'], marker='o', linestyle='-', color='blue', label='快递需求均值')
    plt.title('快递需求均值_聚合' + group_column)
    plt.xlabel(group_column)
    plt.ylabel('快递需求量')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./figure/group/' + group_column + '_plot.png')
    plt.show()

    # 绘制直方图
    plt.figure(figsize=figsize)
    plt.bar(grouped_df.index, grouped_df['sum'], color='skyblue', label='Sum')
    plt.title('Histogram of Sum Parcel per ' + group_column)
    plt.xlabel(group_column)
    plt.ylabel('Parcel')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./figure/group/' + group_column + '_bar.png')
    plt.show()


# H3网格快递量热力图
def figure_H3_grid_heatmap(h3_grid, values, coordinates, name):
    # 创建 folium 地图对象
    mymap = folium.Map(location=[30.2590, 120.1443], zoom_start=13)
    # 将区域边界可视化在地图上
    folium.Polygon(locations=coordinates, color='red', fill=False, fill_opacity=0.2).add_to(mymap)
    # 创建热力图数据列表
    heatmap_data = []
    # 循环处理每个 H3 网格
    for i in range(len(h3_grid)):
        h3_cell = h3_grid[i]
        geometry = h3.h3_to_geo_boundary(h3_cell)
        # 计算网格中心点
        center_lat = sum(coord[0] for coord in geometry) / len(geometry)
        center_lng = sum(coord[1] for coord in geometry) / len(geometry)
        # 添加热力图数据
        value = values[i]
        heatmap_data.append([center_lat, center_lng, value])
    # 添加热力图到地图上
    # HeatMap(heatmap_data, radius=15, blur=10, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(mymap)
    HeatMap(heatmap_data, radius=33, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(mymap)
    # 保存地图为 HTML 文件并在浏览器中打开
    mymap.save(name)


# 以各个小时为统计，预测与真实情况图
def plot_hour_pred(label_i, pred_i, model_name, i):
    R_2 = r2_score(label_i, pred_i)
    print('第%d个小时' % i, 'R方为:%f' % R_2)
    # 实际值与预测值对比图
    f, ax = plt.subplots()
    plt.plot(label_i, 'b-', label='测试集真实值')
    plt.plot(pred_i, 'r-', label=model_name + '预测值')
    plt.legend(loc='upper right')
    plt.title(model_name + '_第%d小时' % i)
    plt.text(0.12, 0.9, "R2 = %f" % R_2,
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax.transAxes)
    plt.savefig('./figure/result/' + model_name + '/hour_%d.png' % i)
    plt.show()
    return R_2


# 计算数值型变量相关性并绘图
def plot_corr(df, title, figsize):
    # 计算相关性
    correlation_matrix = df.corr()

    # 将相关性矩阵转换为DataFrame
    correlation_df = pd.DataFrame(correlation_matrix)

    # 将DataFrame保存为Excel文件
    correlation_df.to_csv("./excel/" + title + ".csv", index=True, encoding='utf_8_sig')

    # 可视化相关性
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('相关性矩阵')
    plt.savefig('./figure/corr/' + title + '.png')
    plt.show()

    return correlation_matrix


# 类别变量相关性图
def plot_categorical_corr(df, var1, var2, path):
    # 设置画布大小
    plt.figure(figsize=(10, 6))

    # 使用 countplot 绘制柱状图
    sns.countplot(x=var1, hue=var2, data=df[[var1, var2]])

    # 添加标题和标签
    plt.title('Countplot of ' + var1 + ' vs. ' + var2)
    plt.xlabel(var1)
    plt.ylabel('Count')
    plt.savefig(path)

    # 显示图形
    plt.show()