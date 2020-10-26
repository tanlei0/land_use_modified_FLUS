# -*- coding: utf-8 -*-
# @Time : 2020/8/9 上午11:26
# @Author : cmk
# @File : map_plot.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Raster import Raster
from tqdm import tqdm

plt.rcParams[u'font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False


# 16色转rgb
def hex2rgb(hex):
    """
    将16进制颜色转为rgb三原色
    #e50000 [229, 0, 0]
    :param hex: 16进制颜色字符串
    :return:
    """
    r = int(hex[1:3], 16)
    g = int(hex[3:5], 16)
    b = int(hex[5:7], 16)
    rgb = [r, g, b]
    return rgb


# 设置图片颜色
def set_land_color(data, lc=None, hexc=None, NoDataValue=None, nodata_c=None, filterType=None, fc=None):
    """
    设置图片特定值的颜色，单通道图像
    :param data: 土地使用数据
    :param lc: 每种cuing的土地类型颜色
    :param hexc: 给出每种土地的16进制颜色
    :param NoDataValue: 无效值
    :param nodata_c: 无效值的颜色，默认为白色
    :param filterType: 不显示某种土地类型
    :param fc: 不需要显示的土地类型的颜色 默认为黑色
    :return: 设置颜色后的图像矩阵
    """
    if filterType is None:
        filterType = []
    if fc is None:
        fc = [255, 255, 255]
    if nodata_c is None:
        nodata_c = [255, 255, 255]
    img = np.zeros(shape=(data.shape[0], data.shape[1], 3), dtype="int32")
    # 转rgb颜色
    rgblc = [hex2rgb(c) for c in hexc]
    rows = data.shape[0]
    cols = data.shape[1]
    for i in tqdm(range(rows)):
        for j in range(cols):
            if data[i][j] == NoDataValue:
                img[i][j] = nodata_c
            elif data[i][j] in filterType:
                img[i][j] = fc
            else:
                img[i][j] = rgblc[data[i][j] - 1]
    return img


def get_defalut_colors():
    """
    获取默认的seaborn颜色
    :return:
    """
    colors = sns.xkcd_rgb
    return colors


def show_prob_map(raster_path, land_type=None, save_path=None, show=True, title="适应度", figure_size=None):
    """
    显示概率图
    :param raster_path: 概率.tif文件路径
    :param land_type: 需要显示的三种土地类型，当长度是1时表示灰度图
    :param save_path: 保存路径
    :param show: 是否显示
    :param title: 标题
    :param figure_size: 图像大小
    :return: img 图像矩阵
    """
    prob_raster = Raster(raster_path)
    print(prob_raster.data[0].shape)
    if land_type is None:
        land_type = [1, 2, 3]
    gray = True if len(land_type) == 1 else False
    type_count = len(land_type)
    if type_count != 1:
        img = np.zeros(shape=(prob_raster.rows, prob_raster.cols, type_count), dtype="int16")
    else:
        img = np.zeros(shape=(prob_raster.rows, prob_raster.cols), dtype="int16")
    print("设置颜色...")
    for i in tqdm(range(prob_raster.rows)):
        for j in range(prob_raster.cols):
            if prob_raster.data[land_type[0]][i][j] == prob_raster.NoDataValue[0]:
                img[i][j] = 255 if type_count == 1 else [255, 255, 255]
            else:
                img[i][j] = int(255 * prob_raster.data[land_type[0] - 1][i][j]) if type_count == 1 else [
                    int(255 * prob_raster.data[t - 1][i][j]) for t in land_type]
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    colors = sns.xkcd_rgb
    hexc = [colors['red'], colors['green'], colors['blue']]
    # 绘制标签
    for i in range(len(land_type)):
        plt.text(x=-20, y=50 * (i + 1) + 5, s=str(land_type[i]),
                 bbox=dict(facecolor=hexc[i], alpha=1), size=10)
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.title(title)
    if type_count == 1:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return img


# 显示彩色图片
def show_color_map(raster_path=None, lc=None, hexc=None, title='None', labels=None, save_path=None, show=True,
                   figure_size=None):
    """
    显示土地使用数据的彩色图
    :param figure_size: 图片大小
    :param raster_path: 土地使用数据的tif文件
    :param lc: 每种土地对应的颜色名称，如red，blue。可用get_defalut_colors查看默认颜色
    :param hexc: 每种土地对应的16进制颜色
    :param title: 图片标题
    :param labels: 每种土地对应的名称
    :param save_path: 保存路径
    :param show: 是否显示
    :return: img彩色图像
    """
    if raster_path is None:
        raise ValueError("raster_path is None, 需要tif数据的路径")
    land = Raster(raster_path)
    colors = sns.xkcd_rgb
    hexc = hexc if hexc is not None else [colors[c] for c in lc]
    img = set_land_color(land.data, hexc=hexc, NoDataValue=land.NoDataValue, filterType=[], fc=[255, 255, 255])
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    else:
        plt.figure(figsize=(19.2, 10.8))
    # 绘制标签
    for i in range(len(hexc)):
        plt.text(x=-20, y=50 * (i + 1) + 5, s=str(i + 1) if labels is None else labels[i],
                 bbox=dict(facecolor=hexc[i], alpha=1), size=10)
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.title(title)
    plt.imshow(img)
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    return img


def show_img(img=None, title="", figure_size=None, save_path=None):
    if img is None:
        print("没有可用于显示的数据")
        return
    if figure_size is not None:
        plt.figure(figsize=figure_size)
    else:
        plt.figure(figsize=(19.2, 10.8))
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.title(title)
    plt.imshow(img, cmap='gray')
    if save_path is not None:
        plt.savefig(save_path)


def get_diff_from2images(image_path1=None, image_path2=None):
    raster1 = Raster(image_path1)
    raster2 = Raster(image_path2)
    data1 = raster1.data
    data2 = raster2.data
    out_data = np.ones_like(data1)
    noval1 = raster1.NoDataValue
    noval2 = raster2.NoDataValue
    rows = data1.shape[0]
    cols = data1.shape[1]
    count = 0
    for i in tqdm(range(rows)):
        for j in range(cols):
            if data1[i][j] != noval1 and data2[i][j] != noval2:
                if data1[i][j] != data2[i][j]:
                    count += 1
                    out_data[i][j] = 0
    return out_data, count


if __name__ == "__main__":
    lc = ['red', 'green', 'blue', 'pink', 'yellow']
    labels = ["土地1", "土地1", "土地1", "土地1", "土地1"]
    # show_color_map(raster_path="../my_sim/my_sim.tif", lc=lc, title="模拟", labels=labels, show=True,
    #                save_path="./test.png")
    # show_prob_map(raster_path="../my_sim/my_ann_probs.tif", land_type=[1, 2, 3])
    # show_prob_map(raster_path="../test_data/result/Probability-of-occurrence.tif", land_type=[4, 2, 3])