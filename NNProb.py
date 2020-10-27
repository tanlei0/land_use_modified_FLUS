# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:29:54 2020

@author: think
"""
from GeoData import GeoData
import numpy as np
from sklearn import preprocessing
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Reshape
import keras

def scale(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std
    return X_norm

class NNProba:
    def __init__(self, GeoData):
        self.y_data = GeoData.Init_data_masked
        self.y_min = GeoData.Init_Raster.vmin
        self.y_max = GeoData.Init_Raster.vmax
        self.X_masked = GeoData.X_masked
        self.y_raster = GeoData.Init_Raster
        
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
    

    
    def train(self, X_train, y_train):
        
        model = Sequential([
            Dense(32, input_dim=self.X_masked.shape[1]),
            Activation('relu'),
            Dense(64),
            Activation('relu'),
            Dense(32),
            Activation('relu'),
            Dense(16),
            Activation('relu'),
            Dense(5),
            Activation('softmax'),
            ])
            
        model.compile(optimizer="Adam",
                  loss='categorical_crossentropy',
                  metrics=["accuracy"])
        
        y_train = np_utils.to_categorical(y_train - 1)

        X_train_norm = scale(X_train)
        
        model.fit(X_train_norm, y_train, epochs=10, batch_size=16)
        
        X_pre_norm = scale(self.X_masked)
        probe = model.predict(X_pre_norm, batch_size=16)
        
        return probe
    
    
    def write_probs(self, probs, y_raster, save_path='./probs.tif'):
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
        for i in range(y_raster.bandsCount):
            temp = np.ma.masked_array(probs[:, i], y_reshape.mask).reshape(y_raster.rows, y_raster.cols)
            temp = np.ma.masked_values(temp, -1)
            y_raster.data[i] = temp
        y_raster.NoDataValue = [-1] * y_raster.bandsCount
        y_raster.write(save_path)
        print("概率文件保存在" + save_path)
    
    def start(self, mode='u', rate = 2, save_path='./probs.tif'):
        X_train, y_train = self.sampling(mode =mode, rate=rate)
        probe = self.train(X_train, y_train)
        self.write_probs(probe, self.y_raster, save_path=save_path)






if __name__ == "__main__":
    Factors_Dir = "Factors/"
    Init_data_file = "dg2001coor.tif"
    
    gd = GeoData(Factors_Dir, Init_data_file)
    nn = NNProba(gd)
    nn.start(save_path = "prob_1.tif")
    
    
    

    