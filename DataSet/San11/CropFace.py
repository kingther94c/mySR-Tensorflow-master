#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 21:37:31 2017

@author: Kingther
"""
import numpy as np
from PIL import Image
import os
filedir = '/Users/Kingther/百度云同步盘/【★快盘★】/【毕业设计】/DataSet/San11/'
for filename in os.listdir(filedir):
    if filename[-4:] != '.jpg':
        continue
    filepath = filedir + filename
    im = Image.open(filepath)
    im = np.array(im,dtype=np.int)
    #(h,w,c)
    box = (60,70,160,170)
    region = im.crop(box)
    #im.show()
    region.show()