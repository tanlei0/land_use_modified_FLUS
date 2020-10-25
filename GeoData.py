# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 12:38:14 2020

@author: tanlei0
"""
from Raster import Raster
import numpy as np
import os
class Cell():
    def __init__(self, row, col, **kwargs):
        self.row = row
        self.col = col
        
        self.paras = kwargs

class GeoData():
    def __init__(self, Factors_Dir, Init_data_file):
        self.Init_Raster  = Raster(Init_data_file)
        self.Init_data = self.Init_Raster.data
        self.Init_NoDataValue = self.Init_Raster.NoDataValue
        self.Init_data_masked = self.Init_Raster.maskedData
        self.Factors_Name = []
        self.Factors_Data = []
        self.Factors_NoDataValue = []
        
        
        for i, file in enumerate(os.listdir(Factors_Dir)):
            if i == 0:
                temp_Raster = Raster(Factors_Dir + file)
                NoDataValue = temp_Raster.NoDataValue
            else:
                temp_Raster = Raster(Factors_Dir +file, NoDataValue = NoDataValue)
            self.Factors_Data.append(temp_Raster.data)
            self.Factors_Name.append(temp_Raster.filename)
            self.Factors_NoDataValue.append(temp_Raster.NoDataValue)
        
        self.n_col = self.Init_Raster.cols
        self.n_row = self.Init_Raster.rows
        self.n_factors = len(self.Factors_Data)
        
        self.X = np.empty(shape=[self.n_row * self.n_col, self.n_factors])
        
        
        temp = [None for i in range(self.n_factors)]
        i = 0
        for factor_data in self.Factors_Data:
            temp[i] = factor_data.reshape(-1)
            self.X[:,i] = temp[i]
            i = i + 1
            
        if len(set(self.Factors_NoDataValue)) == 1:    
            self.X_masked = np.ma.masked_values(self.X, self.Factors_NoDataValue[0])
        else:
            raise TypeError("There are multiple NoDataValue for Factors")
        
        
        
    
            