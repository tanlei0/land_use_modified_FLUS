# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:54:25 2020

@author: tanlei0
"""

import numpy as np
import random
from osgeo import gdal
from sklearn import preprocessing
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Reshape



# In[1]
# Read land use classification layer data
file_land2001 = 'dg2001coor.tif'
data_land2001 = gdal.Open(file_land2001)

file_land2006 = 'dg2006true.tif'
data_land2006 = gdal.Open(file_land2006)

dataset = data_land2001


# In[1.2]

im_height = data_land2001.RasterYSize  # Number of rows in the raster matrix
im_width = data_land2001.RasterXSize  # Number of columns in the raster matrix

# Land use data
im_data_land2001 = data_land2001.ReadAsArray(0, 0, im_width, im_height)
im_data_land2006 = data_land2006.ReadAsArray(0, 0, im_width, im_height)


# In[2]
import os 
factor_list = []
FactorDir = "Factors/"
factor_list = [FactorDir + file for file in os.listdir(FactorDir)]

im_data_factor_list = []
for file in factor_list:
    temp_data = gdal.Open(file)
    im_data_factor_list.append(temp_data.ReadAsArray(0, 0, im_width, im_height))
    

# In[3]
X = np.array(im_data_factor_list).reshape(im_width*im_height, 8)

# Normalization function
#min_max_scaler = preprocessing.MinMaxScaler()

#X_3d_norm = min_max_scaler.fit_transform(X_3d)

y = im_data_land2001

print("the shape of X_3d_Norm:", X.shape)
print("the shape of y:", y.shape)
# In[4]
import cv2
import matplotlib.pyplot as plt

img = X[:,0].reshape(im_height,im_width)
plt.imshow(img)




# In[4.1]

# acquire training samples
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []

    for i in range(length):
        random_list.append(random.randint(start, stop))
    return np.array(random_list)


random_list = random_int_list(0,Sample_2005_Norm.shape[0]-1,int(Sample_2005_Norm.shape[0]* 0.2))

ann_data = np.zeros((int(Sample_2005_Norm.shape[0]* 0.2),23))
ann_data_label = np.zeros((int(Sample_2005_Norm.shape[0]* 0.2),1))
for i in range(int(Sample_2005_Norm.shape[0]* 0.2)):
    temp = random_list[i]
    ann_data[i]= Sample_2005_Norm[temp]               #ann_data: 20% of origin samples
    ann_data_label[i] = Label_2005[temp]         #ann_data_label

train_num = int(ann_data.shape[0] * 0.7)    # set number of training samples
test_num = ann_data.shape[0] - train_num    # set number of test samples

# acquire training samples from ann_data
ann_data_train = np.zeros((train_num,23))
ann_data_test = np.zeros((test_num,23))
ann_data_train_label = np.zeros((train_num,1))
ann_data_test_label = np.zeros((test_num,1))
for i in range(train_num):
    ann_data_train[i] = ann_data[i]
    ann_data_train_label[i] = ann_data_label[i]
for j in range(train_num,ann_data.shape[0]):
    ann_data_test[j - train_num] = ann_data[j]
    ann_data_test_label[j - train_num] = ann_data_label[j]

# transform label into one-hot
ann_data_train_label = np_utils.to_categorical(ann_data_train_label, num_classes=7)
ann_data_test_label = np_utils.to_categorical(ann_data_test_label, num_classes=7)

# set neural networks
model = Sequential([
    Dense(32, input_dim=23),
    Activation('relu'),
    Dense(64),
    Activation('relu'),
    Dropout(0.02),
    Dense(7),
    Activation('softmax'),
])

# select optimizer
rmsprop = RMSprop(lr=0.02, rho=0.8, epsilon=1e-08, decay=0.0)   

# The loss function and the precision evaluation index are selected
model.compile(optimizer=rmsprop,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Start training using training data
print('Training ------------')

model.fit(ann_data_train, ann_data_train_label, epochs=10, batch_size=16)

# Optimize with test data
print('\nTesting ------------')

loss, accuracy = model.evaluate(ann_data_test, ann_data_test_label)
# Output training results
print('test loss: ', loss)
print('test accuracy: ', accuracy)

# The simulation was performed using 2010 image data
predict = model.predict(Sample_2010_Norm, batch_size=32, verbose=0)

predict_out = np.zeros((predict.shape[0], predict.shape[1]))
# Find the maximum value for each row and determine if it is greater than the threshold
theshold = 0.8
Change_sum = 0
for i in range(predict.shape[0]):
    for j in range(predict.shape[1]):
        if predict[i][j] >= theshold:
            predict_out[i][j] = 1
            Change_sum = Change_sum + 1
        else:
            predict_out[i][j] = 0


print("The number of predict > theshold:", Change_sum)
print("The rate of predict > theshold:", Change_sum / number)

# get data_new after simulation
Label_predict = np.zeros((number, 1))

index = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            Label_predict[index][0] = im_data_land2010[row][col]

            for j in range(predict.shape[1]):
                if predict_out[index][j] == 1:
                    Label_predict[index][0] = j

            index += 1

# Gets the simulated array data
data_new = np.zeros((im_height, im_width))
index = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            data_new[row][col] = Label_predict[index][0]
            index = index + 1

same_label_origin = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            if im_data_land2010[row][col] == im_data_land2015[row][col]:
                same_label_origin = same_label_origin + 1

print("The same label between im_data_land2010 and im_data_land2015 = ", same_label_origin)

same_label = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            if im_data_land2010[row][col] == data_new[row][col]:
                same_label = same_label + 1

print("The same label between im_data_land2010 and data_new = ", same_label)

same = 0
for row in range(12, im_height - 12):
    for col in range(12, im_width - 12):
        if im_data_land2010[row][col] != 0:
            if im_data_land2015[row][col] == data_new[row][col]:
                same = same + 1

print("The same label between im_data_land2015 and data_new = ", same)
print("the accuracy of predict is:", same / number)

data_new_outtxt = '/home/anaconda_workspace/qyh/SH_100m/ExperimentalResult/CNNs_Merely/ANN_CA.txt'
np.savetxt(data_new_outtxt, data_new, fmt='%s', newline='\n')