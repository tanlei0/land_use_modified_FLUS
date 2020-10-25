# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:29:54 2020

@author: think
"""
from GeoData import GeoData
import numpy as np

class NNProba:
    def __init__(self, GeoData):
        self.y_data = GeoData.Init_data_masked
        self.y_min = GeoData.Init_Raster.vmin
        self.y_max = GeoData.Init_Raster.vmax
        self.X_masked = GeoData.X_masked
    
    def sampling(self,mode = "u", rate = 2):
        """
        mode = "u" or "r"
        """
        
        distrib_dict = {}
        for i in range(self.y_min, self.y_max + 1):
            distrib_dict[i] = (self.y_data == i).sum()
            
        n_sample = int(self.y_data.shape[0] * self.y_data.shape[1] * rate / 100)
        
        if mode == "r":
            temp = np.array(list(distrib_dict.values()))
            dis_n_sampling = (temp / temp.sum() * n_sample).astype("int")
        else:
            dis_n_sampling = [int(n_sample / len(distrib_dict))  for i in range(len(distrib_dict))]

        y_reshape = self.y_data.reshape(-1)
        
        #采样时去除两部分样本，1是X中有部分被mask的，因此找到mask和为0的就可以.2是y中被mask的
        sampling_index1 = set(np.argwhere((self.X_masked.mask).sum(axis = 1) == 0).reshape(-1).tolist())
        sampling_index2 = set(np.argwhere(y_reshape.mask == False).reshape(-1).tolist())
        y_index = list(sampling_index1 & sampling_index2)
        
        sampling_index = []
        for i in range(self.y_min, self.y_max+1):
            temp_list = np.argwhere(y_reshape == i)
            temp_list = list(set(y_index) & set(temp_list.reshape(-1).tolist()))
            try:
                index_list = np.random.choice(temp_list, dis_n_sampling[i-1], replace=False).tolist()
            except ValueError:
                raise ValueError("比例过大，数据不够 \n rate is so large that there is no enough data.")
            sampling_index = sampling_index + index_list
            
        X_train = self.X_masked[sampling_index]
        y_train = y_reshape[sampling_index]
        
        return X_train, y_train
        
    
FACTORS_DIR = "Factors/"
INIT_DATA_FILE = "dg2001coor.tif"
gd = GeoData(FACTORS_DIR, INIT_DATA_FILE)
nn = NNProba(gd)
X_train, y_train = nn.sampling(rate = 20)

    