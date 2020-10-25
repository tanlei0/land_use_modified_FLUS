# -*- coding: utf-8 -*-
"""
Created on Fri May  8 13:40:31 2020

@author: tanlei0
"""

from Raster import Raster
from osgeo import gdal
import numpy as np

raster_file = "Factors/Aspect.tif"
raster_file2 = "Factors/dem_dg.tif"
raster_file3 = "probe.tif"
#raster = Raster(raster_file)
#raster2 = Raster(raster_file2)
#raster.draw()
#raster.write("test1.tif")
#raster2.draw()

#r = gdal.Open(raster_file3)
r = Raster(raster_file3)

data = []
for i in range(5):
    r.data[i] = probe[:,i].reshape(r.rows, r.cols)
    
r.write("probe_test.tif")
