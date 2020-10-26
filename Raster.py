# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:07:17 2020

@author: tanlei0
"""

from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
class Raster():
    def __init__(self, filename, NoDataValue = None):
        self.filename = filename
        self.dataset = gdal.Open(filename)
        self.rows = self.dataset.RasterYSize    #栅格矩阵的列数
        self.cols = self.dataset.RasterXSize   #栅格矩阵的行数
        self.bandsCount = self.dataset.RasterCount    
        self.geotrans = self.dataset.GetGeoTransform()  #仿射矩阵
        self.proj = self.dataset.GetProjection() #地图投影信息
        # 处理单通道灰度文件
        if self.bandsCount == 1:    
            self.band = self.dataset.GetRasterBand(1)
            if NoDataValue != None:
                self.band.SetNoDataValue(NoDataValue)
                
            self.NoDataValue =  self.band.GetNoDataValue()
            self.data = self.band.ReadAsArray()
            
            
            # masked data
            array = self.band.ReadAsArray()
            
            if self.NoDataValue is not None:
                self.maskedData = np.ma.masked_values(array, self.NoDataValue)
                self.vmin, self.vmax = self.maskedData.min(), self.maskedData.max()
        else:
            
            print("This is a multiBand file")
            self.band = []
            self.data = []
            self.NoDataValue = []
            for i in range(self.bandsCount):
                self.band.append(self.dataset.GetRasterBand(i+1))
                self.NoDataValue.append(self.band[i].GetNoDataValue())
                self.data.append(self.band[i].ReadAsArray())
            
            
        
        
    def write(self, path):
        if self.bandsCount == 1:
            if 'int8' in self.data.dtype.name:
                datatype = gdal.GDT_Byte
            elif 'int16' in self.data.dtype.name:
                datatype = gdal.GDT_UInt16
            elif "int32" in self.data.dtype.name:
                datatype = gdal.GDT_Int32
            else:
                datatype = gdal.GDT_Float32
        else:
            datatype = gdal.GDT_Float32
        
        driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(path, self.cols, self.rows, self.bandsCount, datatype)
        dataset.SetGeoTransform(self.geotrans)              #写入仿射变换参数
        dataset.SetProjection(self.proj)                    #写入投影
        
        if self.bandsCount == 1:
            band = dataset.GetRasterBand(1)
            band.SetNoDataValue(self.NoDataValue)
        
            dataset.GetRasterBand(1).WriteArray(self.data)  #写入数组数据
        else:
            for i in range(self.bandsCount):
                band = dataset.GetRasterBand(i+1)
                band.SetNoDataValue(self.NoDataValue[i])
                dataset.GetRasterBand(i+1).WriteArray(self.data[i])
        
        del dataset
        
    def draw(self):
        # todo 改好
        if self.bandsCount > 1:
            data = np.array(self.data[:3]).reshape()
            # plt.imshow(, cmap = "rgb")
        else:
            plt.imshow(self.data, cmap = "gray")
        plt.show()
        
    def __del__(self):
        del self.dataset
    

    @staticmethod
    def Create(filename, cols,rows, bands, bands_data, NoDataValue,trans,proj ,driver = gdal.GetDriverByName("GTiff")):
        # todo 导入flus后NODATAVALUE显示黑色问题。
        datatype = gdal.GDT_Float32
        dataset = driver.Create(filename, cols, rows, bands, datatype)
        dataset.SetGeoTransform(trans)
        dataset.SetProjection(proj)
        for i in range(bands):
            band = dataset.GetRasterBand(i+1)
            dataset.GetRasterBand(i+1).WriteArray(bands_data[i])
            band.SetNoDataValue(NoDataValue)
        