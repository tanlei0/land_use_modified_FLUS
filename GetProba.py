# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:14:19 2020

@author: tanlei0
"""

from Raster import Raster
import numpy as np
from osgeo import gdal
import os 
import matplotlib.pyplot as plt

from sklearn import preprocessing
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Reshape
import keras


    
factor_list = []
FactorDir = "Factors/"
factor_list = [FactorDir + file for file in os.listdir(FactorDir)]
y_file = "dg2001coor.tif"
raster = Raster(factor_list[0])
n_row = raster.rows
n_col = raster.cols

data_factors = []
X = np.empty(shape=[n_row * n_col, len(factor_list)])
temp = [None for i in range(len(factor_list))]
i = 0
for file in factor_list:
    temp[i] = Raster(file).data.reshape(-1)
    X[:,i] = temp[i]
    i = i + 1
X_masked = np.ma.masked_values(X, raster.NoDataValue)
y_raster = Raster(y_file)
y = y_raster.maskedData

distrib_dict = {}
for i in range(y_raster.vmin, y_raster.vmax + 1):
    distrib_dict[i] = (y == i).sum()
    
# In[1] Sampling
mode = "u"
rate = 2
n_sample = int(y.shape[0] * y.shape[1] * rate / 100)
if mode is "r":
    temp = np.array(list(distrib_dict.values()))
    dis_n_sampling = (temp / temp.sum() * n_sample).astype("int")
else:
    dis_n_sampling = [int(n_sample / len(distrib_dict))  for i in range(len(distrib_dict))]

y_reshape = y.reshape(-1)

sampling_index1 = set(np.argwhere((X_masked.mask).sum(axis = 1) == 0).reshape(-1).tolist())
sampling_index2 = set(np.argwhere(y_reshape.mask == False).reshape(-1).tolist())
y_index = list(sampling_index1 & sampling_index2)

sampling_index = []
for i in range(y_raster.vmin, y_raster.vmax+1):
    temp_list = np.argwhere(y_reshape == i)
    temp_list = list(set(y_index) & set(temp_list.reshape(-1).tolist()))
    try:
        index_list = np.random.choice(temp_list, dis_n_sampling[i-1], replace=False).tolist()
    except ValueError:
        raise ValueError("比例过大，数据不够 \n rate is so large that there is no enough data.")
    sampling_index = sampling_index + index_list
    
X_train = X_masked[sampling_index]
y_train = y_reshape[sampling_index]

# In[1.1]
def scale(X):
    mean = X.mean(axis = 0)
    std = X.std(axis = 0)
    X_norm = (X - mean) / std
    return X_norm

y_train = np_utils.to_categorical(y_train - 1)

X_train_norm = scale(X_train)

# In[2] ANN

model = Sequential([
    Dense(32, input_dim=8),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dense(128),
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

model.fit(X_train_norm, y_train, epochs=2, batch_size=16) 

   
X_pre_norm = scale(X_masked)
probe = model.predict(X_pre_norm , batch_size = 16)
# In[3]
# todo 写create
#raster_file3 = "probe.tif"
#r = Raster(raster_file3)
#for i in range(5):
#    temp = np.ma.masked_array(probe[:,i], y_reshape.mask).reshape(r.rows, r.cols)
#    temp = np.ma.masked_values(temp, -1)
#    r.data[i] = temp
##r.write("probe_test.tif")
#r.draw()
# In[3.1]


bands_data = []
for i in range(5):
    temp = np.ma.masked_array(probe[:,i], y_reshape.mask).reshape(raster.rows, raster.cols)
    temp = np.ma.masked_values(temp, raster.NoDataValue)
    bands_data.append(temp)
    
Raster.Create("probe_test_1.tif", raster.cols,raster.rows, len(bands_data), bands_data,
              raster.NoDataValue,trans=y_raster.geotrans,proj=y_raster.proj)

#r.draw()
    





