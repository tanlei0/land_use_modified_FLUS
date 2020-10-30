#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/10/28 9:19
# @Author  : cmk
# @Email   : litaoyo@163.com
# @File    : simulation_ca.py
# @Software: PyCharm
import multiprocessing
import time

import numpy as np
from numpy.ma import MaskedArray
from typing import List
import random
from tqdm import tqdm

from Raster import Raster

np.seterr(divide='ignore', invalid='ignore')


# ############# Tips ######################
# all values which represent the type of the land start from 0, while start from 1 in the land matrix
# ###########################################

class LandCell(object):

    def __init__(self, row: int, col: int, land_type: int, nei_offset: np.ndarray):
        # cell's coordinate
        self.row = row
        self.col = col
        self.land_type = land_type

        # the coordinate of cell's neighbor
        self.nei_coores = nei_offset + [self.row, self.col]
        # neighbor's effect
        self.nei_effect = None
        # combine probability
        self.combine_prob = None

    def update_neis(self, nTypes: int, nei_window: float, land_mask_data: MaskedArray, wnb: np.ndarray) -> np.ndarray:
        """
        update the neighbor's effect of the cell
        :param nTypes: The total number of land types
        :param nei_window: the window of neighbor
        :param land_mask_data: landUse data
        :param wnb: the weight of each neighbor
        :return:
        """
        nei_count = np.zeros((nTypes,))
        nrows, ncols = land_mask_data.shape
        for nei_coor in self.nei_coores:
            if 0 <= nei_coor[0] < nrows and 0 <= nei_coor[1] < ncols:
                if land_mask_data[nei_coor[0], nei_coor[1]] != land_mask_data.fill_value:
                    nei_type = land_mask_data[nei_coor[0], nei_coor[1]] - 1
                    if 0 <= nei_type < nTypes:
                        nei_count[nei_type] += 1
        self.nei_effect = nei_count / nei_window * wnb
        return self.nei_effect

    def cal_com_probs(self, nTypes: int, nei_window: float, wnb: np.ndarray, land_mask_data: MaskedArray,
                      probs_data: np.ndarray,
                      cost_matrix: np.ndarray) -> None:
        """
        calculate the combine probability but the land inertia
        :param nTypes:
        :param wnb:
        :param land_mask_data:
        :param probs_data:
        :param cost_matrix:
        :return:
        """

        # 1. get probability
        probs = probs_data[:, self.row, self.col]

        # 2. nei effect
        self.update_neis(nTypes=nTypes, nei_window=nei_window, land_mask_data=land_mask_data, wnb=wnb)

        # 3. get cost
        change_cost = cost_matrix[self.land_type, :]
        self.combine_prob = probs * self.nei_effect * change_cost


def get_nei_offset(nei_nums):
    """
    get the coordinate offsets of neighbors
    :param nei_nums: 邻域大小
    :return: 偏移量
    """
    # 获取邻域的偏移量
    nei_offset = []
    s = (nei_nums - 1) // 2
    r = c = -s
    for i in range(nei_nums):
        for j in range(nei_nums):
            if r + i == 0 and c + j == 0:
                continue
            else:
                nei_offset.append([r + i, c + j])
    return np.asarray(nei_offset)


def initial_land_cell(nei_offset: np.ndarray, land_data: MaskedArray, restricted_data: np.ndarray = None) -> List[
    LandCell]:
    # get the coordinate of valid pixel
    # rows: [...]
    # cols: [...]
    valid_coors = np.where(land_data.mask == False)
    valid_nums = len(valid_coors[0])
    land_cells = []
    for i in range(valid_nums):
        row = valid_coors[0][i]
        col = valid_coors[1][i]
        # skip the restricted area
        if restricted_data is not None and restricted_data[row][col] == 0:
            continue
        land_type = land_data[row][col] - 1
        lc = LandCell(valid_coors[0][i], valid_coors[1][i], land_type, nei_offset)
        land_cells.append(lc)
    return land_cells


def func_update_com_prob(cells: List[LandCell], nTypes: int, nei_window: float, wnb: np.ndarray,
                         land_mask_data: MaskedArray,
                         probs_data: np.ndarray,
                         cost_matrix: np.ndarray) -> None:
    t_start = time.time()
    for cell in cells:
        cell.cal_com_probs(nTypes=nTypes, nei_window=nei_window, wnb=wnb, land_mask_data=land_mask_data,
                           probs_data=probs_data, cost_matrix=cost_matrix)
    t_stop = time.time()
    print("执行完成，耗时%.2f" % (t_stop - t_start))


def start_simulation(config: dict):
    num_process = multiprocessing.cpu_count()
    # 邻域大小
    N = config['simConfig']['neighboorhoodOdd']
    nei_window = N * N - 1
    max_iter_nums = config['simConfig']['maxIterNum']
    nei_offset = get_nei_offset(N)
    lanuse_demand = np.asarray(config['landUseDemand'])
    cost_matrix = np.asarray(config['costMatrix'])
    wnb = np.asarray(config['weightOfNeighborhood'])
    degree = config['degree']

    # 读取需要的tif文件
    land_raster = Raster(config['landUsePath'])
    land_mask_data = land_raster.maskedData
    nrows, ncols = land_mask_data.shape
    num_classes = land_raster.vmax

    if 'restrictedPath' in config:
        restricted_data = Raster(config['restrictedPath']).data
    else:
        restricted_data = None

    probs_raster = Raster(config['probsPath'])
    probs_data = np.asarray(probs_raster.data)

    # get valid land cells
    land_cells = initial_land_cell(nei_offset, land_mask_data, restricted_data)

    save_count = np.asarray([len(np.where(land_mask_data == i + 1)[0]) for i in range(num_classes)])
    sumPixels = save_count.sum()
    # difference between demand and current land in the beginning
    initialDist = np.copy(save_count)
    # t - 2 difference
    dynaDist = np.copy(save_count)
    # the minimum difference
    best_dist = np.copy(save_count)
    # t - 1 difference
    mIminDis2goal = np.copy(save_count)
    # 1 means resist transition
    oppo_trans = np.zeros_like(save_count)
    # land inertia
    adjustment = np.zeros_like(save_count)
    adjustment_effect = np.ones_like(save_count)

    # roulette
    mroulette = np.zeros((num_classes + 1,))

    print("start simulation: ")
    print("landuse demand: ", lanuse_demand)
    print("initial count: ", save_count)
    st = time.time()
    for k in range(max_iter_nums):
        print("=====================================")
        print("interation: ", k)
        print("curr land use: ", save_count)
        print("diff: ", lanuse_demand - save_count)

        # ============= update inertia ======================
        stui = time.time()
        print("-----------------------------")
        print("update inertia...")
        for i in range(num_classes):
            mIminDis2goal[i] = lanuse_demand[i] - save_count[i]
            if k == 0:
                initialDist[i] = mIminDis2goal[i]
                dynaDist[i] = initialDist[i] * 1.01
                best_dist[i] = initialDist[i]

            if abs(best_dist[i] > abs(mIminDis2goal[i])):
                best_dist[i] = mIminDis2goal[i]
            else:
                if abs(mIminDis2goal[i] - abs(best_dist[i])) / abs(initialDist[i] > 0.05):
                    oppo_trans[i] = 1

            adjustment[i] = mIminDis2goal[i] / dynaDist[i] if dynaDist[i] != 0 else 1

            if 0 < adjustment[i] < 1:
                dynaDist[i] = mIminDis2goal[i]

                if initialDist[i] > 0 and adjustment[i] > 1 - degree:
                    adjustment_effect[i] = adjustment_effect[i] * (adjustment + degree)

                if initialDist[i] < 0 and adjustment[i] > 1 - degree:
                    adjustment_effect[i] = adjustment_effect[i] * (1.0 / (adjustment[i] + degree))

            if initialDist[i] > 0 and adjustment[i] > 1:
                adjustment_effect[i] = adjustment_effect[i] * adjustment[i] * adjustment[i]

            if initialDist[i] < 0 and adjustment[i] > 1:
                adjustment_effect[i] = adjustment_effect[i] * (1.0 / adjustment[i]) * (1.0 / adjustment[i])
        print("update inertia end!!! Time used: ", stui - time.time())
        # ===================================================

        # ============= cal combine probability =============
        stui = time.time()
        print("--------------------------")
        print("cal combine probability...")

        # To Do 多进程还有些问题，进程中修改对象，不对外部对象起作用
        # process = []
        # prange = int(len(land_cells) / num_process)
        # process_index = [[i * prange, (i + 1) * prange] if i != num_process - 1 else [(i - 1) * prange, len(land_cells)]
        #                  for i in range(num_process)]
        # for pi in process_index:
        #     pro = multiprocessing.Process(target=func_update_com_prob, args=(
        #         land_cells[pi[0]: pi[1]], num_classes, nei_window, wnb, land_mask_data, probs_data, cost_matrix))
        #     pro.start()
        #     process.append(pro)
        # for p in process:
        #     p.join()

        # 使用进程池比不使用进程池还要慢很多
        # pool = multiprocessing.Pool(processes=num_process)
        # for index, cell in enumerate(land_cells):
        #     pool.apply_async(func_update_com_prob,
        #                      (index, cell, num_classes, nei_window, wnb, land_mask_data, probs_data, cost_matrix))
        # pool.close()
        # pool.join()
        for cell in land_cells:
            func_update_com_prob(0, cell, num_classes, nei_window, wnb, land_mask_data, probs_data, cost_matrix)
        print("cal combine probability end!!!, Time used: ", stui - time.time())
        # ===================================================

        # =================== do transition =================
        stui = time.time()
        for land_cell in land_cells:
            i = land_cell.row
            j = land_cell.col
            old_type = land_cell.land_type

            # get land inertia
            land_inertia = 10 * num_classes * adjustment_effect[old_type]
            land_cell.combine_prob[old_type] *= land_inertia

            # roulette choice
            mroulette[0] = 0
            for ii in range(num_classes):
                mroulette[ii + 1] = mroulette[ii] + land_cell.combine_prob[ii]

            # get a random float number and do roulette choice
            temp_rand = random.random()
            isConvert = False
            for ii in range(num_classes):
                new_type = ii
                if mroulette[ii] < temp_rand <= mroulette[ii + 1]:
                    # if save_count[new_type] != lanuse_demand[new_type] or \
                    #         save_count[old_type] != lanuse_demand[old_type]:
                    #
                    #     save_count[new_type] += 1
                    #     save_count[old_type] -= 1

                    if cost_matrix[old_type][new_type] != 0 and new_type != old_type:
                        isConvert = True
                    else:
                        isConvert = False

                    _disChangeFrom = mIminDis2goal[old_type]
                    _disChangeTo = mIminDis2goal[new_type]
                    if initialDist[new_type] >= 0 and _disChangeTo == 0:
                        adjustment_effect[new_type] = 1
                        isConvert = False

                    if initialDist[old_type] <= 0 and _disChangeFrom == 0:
                        adjustment_effect[old_type] = 1
                        isConvert = False

                    if initialDist[old_type] >= 0 and oppo_trans[old_type] == 1:
                        isConvert = False
                    if initialDist[new_type] <= 0 and oppo_trans[new_type] == 1:
                        isConvert = False

                    if isConvert:
                        # update datas
                        land_cell.land_type = new_type
                        land_mask_data[i][j] = new_type + 1
                        save_count[new_type] += 1
                        save_count[old_type] -= 1
                        mIminDis2goal[new_type] = lanuse_demand[new_type] - save_count[new_type]
                        mIminDis2goal[old_type] = lanuse_demand[old_type] - save_count[old_type]
                        break

                    oppo_trans[old_type] = 0
                    oppo_trans[new_type] = 0

        sumDis = np.fabs(mIminDis2goal).sum()
        if sumDis == 0 or sumDis < sumPixels * 0.0001:
            break
        print("-------------------------------------")
        print("Time used: ", time.time() - stui)

    print("--------------------------------------------------")
    # the NoDataValue in the simulation file should be zero
    land_raster.NoDataValue = 0
    land_raster.data = land_mask_data.data
    land_raster.write(config['saveSimPath'])
    print("simulation end! The file is saved to ", config['saveSimPath'])
    print("Time used: ", time.time() - st)


if __name__ == '__main__':
    params1 = {
        "landUsePath": "./dg2001coor.tif",
        "probsPath": "./Probability-of-occurrence.tif",
        "saveSimPath": "./sim_file/test_sim.tif",
        "restrictedPath": "./restriction/restrictedarea.tif",
        "simConfig": {
            "maxIterNum": 35,
            "neighboorhoodOdd": 3
        },
        "landUseDemand": [80016, 54427, 43599, 42433, 28446],
        "costMatrix": [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 0, 1],
        ],
        "weightOfNeighborhood": [1, 0.9, 0.5, 1, 0.1],
        'degree': 0.1
    }

    params = {
        "landUsePath": "../test/hgs/hgs_datas/12n.tif",
        "probsPath": "../test/hgs/hgs_datas/sim_file/hgs_probs.tif",
        "saveSimPath": "../test/hgs/hgs_datas/sim_file/黄瓜山my_sim.tif",
        "simConfig": {
            "maxIterNum": 35,
            "neighboorhoodOdd": 3
        },
        "landUseDemand": [337942, 131006, 203277, 11992, 67340, 19631, 458, 4894],
        "costMatrix": [
            [1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 0, 1]
        ],
        "weightOfNeighborhood": [0.2, 0.1, 0.6, 0.2, 0.5, 0.2, 0, 0.1],
        'degree': 0.1
    }
    start_simulation(params)
