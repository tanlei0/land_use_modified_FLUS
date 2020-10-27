# -*- coding: utf-8 -*-
# @Time : 2020/7/28 下午12:07
# @Author : cmk
# @File : mutils.py
import json
import numpy as np
from tqdm import tqdm
from Raster import Raster


# ############ 将ANN的预测结果保存到tif文件中 ########################
def write_probs(probs, y_raster, save_path='./probs.tif'):
    """
    将ANN计算的适应性概率保存到文件
    :param probs: ANN的输出
    :param y_raster: 土地类型的raster
    :param save_path: 概率文件保存位置
    :return:
    """
    y_raster.bandsCount = len(np.unique(y_raster.data)) - 1
    y_raster.data = np.zeros(shape=(y_raster.bandsCount, y_raster.rows, y_raster.cols))
    y_reshape = y_raster.maskedData.reshape(-1)
    probs[np.isnan(probs)] = -1
    for i in range(5):
        temp = np.ma.masked_array(probs[:, i], y_reshape.mask).reshape(y_raster.rows, y_raster.cols)
        temp = np.ma.masked_values(temp, -1)
        y_raster.data[i] = temp
    y_raster.NoDataValue = [-1] * y_raster.bandsCount
    y_raster.write(save_path)
    print("概率文件保存在" + save_path)


# def write_probs1(probs, baseRaster, y_raster, save_path='./probs.tif'):
#     """
#     将ANN计算的适应性概率保存到文件
#     :param probs: ANN的输出
#     :param baseRaster: 以baseRaster中的参数作为基本参数
#     :param y_raster: 土地类型的raster
#     :param save_path: 概率文件保存位置
#     :return:
#     """
#     baseRaster.bandsCount = int(y_raster.vmax - y_raster.vmin + 1)
#     baseRaster.data = np.zeros(shape=(baseRaster.bandsCount, y_raster.rows, y_raster.cols))
#     y_reshape = y_raster.maskedData.reshape(-1)
#     probs[np.isnan(probs)] = -1
#     for i in range(5):
#         temp = np.ma.masked_array(probs[:, i], y_reshape.mask).reshape(y_raster.rows, y_raster.cols)
#         temp = np.ma.masked_values(temp, -1)
#         baseRaster.data[i] = temp
#     baseRaster.NoDataValue = [-1] * baseRaster.bandsCount
#     baseRaster.write(save_path)
#     print("概率文件保存在" + save_path)


# ############ 计算kappa系数 ########################
def calc_kappa(sim_path, true_path):
    # 读取数据
    sim_raster = Raster(sim_path)
    true_raster = Raster(true_path)
    sim_data = sim_raster.data
    sim_noValue = sim_raster.NoDataValue
    true_data = true_raster.data
    true_noValue = true_raster.NoDataValue

    type_num = true_raster.vmax
    # 计算混淆矩阵
    count = 0
    confusion_matrix = np.zeros(shape=(type_num, type_num), dtype=np.int64)
    rows = true_raster.rows
    cols = true_raster.cols
    for i in tqdm(range(rows)):
        for j in range(cols):
            if type_num >= true_data[i][j] >= 1 and type_num >= sim_data[i][j] >= 1:
                confusion_matrix[true_data[i][j] - 1][sim_data[i][j] - 1] += 1
                count += 1

    # 计算整体准确率overall accuracy（p0）
    oa = sum([confusion_matrix[i][i] for i in range(type_num)]) / count
    # 计算pe
    pe = sum([confusion_matrix[:, i].sum() * confusion_matrix[i, :].sum() for i in range(type_num)]) / (count * count)
    kappa = (oa - pe) / (1 - pe)

    return {"overall accuracy": oa, "kappa": kappa, 'confusion_matrix': confusion_matrix}


# ############ 计算fom指数 ########################
def calc_fom(start_path, true_path, sim_path):
    # 读取数据
    sim_raster = Raster(sim_path)
    true_raster = Raster(true_path)
    start_raster = Raster(start_path)

    sim_data = sim_raster.data
    sim_noValue = sim_raster.NoDataValue

    true_data = true_raster.data
    true_noValue = true_raster.NoDataValue

    start_data = start_raster.data
    start_noValue = start_raster.NoDataValue
    fom = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'FoM': 0}
    # 获取A、B、C、D
    for i in tqdm(range(true_raster.rows)):
        for j in range(true_raster.cols):
            if true_data[i][j] == true_noValue or sim_data[i][j] == sim_noValue or start_data[i][j] == start_noValue:
                continue
            # 实际没变，但模拟变化了====D
            if start_data[i][j] == true_data[i][j] and start_data[i][j] != true_data[i][j]:
                fom['D'] += 1
            # 实际变化了
            if start_data[i][j] != true_data[i][j]:
                # 模拟没有变化===A
                if start_data[i][j] == sim_data[i][j]:
                    fom['A'] += 1
                # 模拟变化和实际变化相符===B
                if sim_data[i][j] == true_data[i][j]:
                    fom['B'] += 1
                # 模拟变化和实际变化不相符合===C
                elif sim_data[i][j] != true_data[i][j]:
                    fom['C'] += 1
    fom['FoM'] = fom['B'] / sum([fom[key] for key in fom])
    return fom


def eval_sim(config=None, config_path=None):
    if config is None:
        with open(config_path, 'r', encoding="utf8") as fp:
            config = json.load(fp)
    sim_path_ = config["sim_path"]
    true_path_ = config["true_path"]
    start_path_ = config["start_path"]
    print("=================================")
    print("计算Kappa系数")
    print(calc_kappa(sim_path=sim_path_, true_path=true_path_))
    print("=================================")
    print("计算FoM")
    print(calc_fom(sim_path=sim_path_, true_path=true_path_, start_path=start_path_))


if __name__ == "__main__":
    # sim_path = "../test_data/my_sim_m.tif"
    # true_path = "../test_data/landtype/dg2006true.tif"
    # start_path = "../test_data/landtype/dg2001coor.tif"
    start_path = "../datas/黄瓜山模拟FLUS/2012_89cleaned.tif"
    true_path = "../datas/黄瓜山模拟FLUS/2017_89cleaned.tif"
    sim_path = "../my_sim/黄瓜山my_sim.tif"
    # print(calc_kappa(sim_path=sim_path, true_path=true_path))
    print(calc_fom(sim_path=sim_path, true_path=true_path, start_path=start_path))
    # print(2147483647.0 == 2147483647)
