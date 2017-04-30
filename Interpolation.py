#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:21:55 2017

@author: Kingther
"""

import numpy as np
from PIL import Image
import math  
import os

def bilinear(img,H,W):  
    img = np.array(img,dtype=np.int)
    height,width,channels =img.shape  
    emptyImage=np.zeros((H,W,channels),np.uint8)  
    value=[0,0,0]  
    sh=H/height  
    sw=W/width  
    for i in range(H):  
        for j in range(W):  
            x = i/sh  
            y = j/sw  
            p=(i+0.0)/sh-x  
            q=(j+0.0)/sw-y  
            x=int(x)-1  
            y=int(y)-1  
            for k in range(3):  
                if x+1<H and y+1<W:  
                    value[k]=int(img[x,y][k]*(1-p)*(1-q)+img[x,y+1][k]*q*(1-p)+img[x+1,y][k]*(1-q)*p+img[x+1,y+1][k]*p*q)  
            emptyImage[i, j] = (value[0], value[1], value[2])  
    return emptyImage 
        
def bicubic(img,H,W):  
    def S(x):  
        x = np.abs(x)  
        if 0 <= x < 1:  
            return 1 - 2 * x * x + x * x * x  
        if 1 <= x < 2:  
            return 4 - 8 * x + 5 * x * x - x * x * x  
        else:  
            return 0 
    img = np.array(img,dtype=np.int)
    height,width,channels =img.shape  
    emptyImage=np.zeros((H,W,channels),np.uint8)  
    sh=H/height  
    sw=W/width  
    for i in range(H):  
        for j in range(W):  
            x = i/sh  
            y = j/sw  
            p=(i+0.0)/sh-x  
            q=(j+0.0)/sw-y  
            x=int(x)-2  
            y=int(y)-2  
            A = np.array([  
                [S(1 + p), S(p), S(1 - p), S(2 - p)]  
            ])  
            if x>=H-3:  
                H-1  
            if y>=W-3:  
                W-1  
            if x>=1 and x<=(H-3) and y>=1 and y<=(W-3):  
                B = np.array([  
                    [img[x-1, y-1], img[x-1, y],  
                     img[x-1, y+1],  
                     img[x-1, y+1]],  
                    [img[x, y-1], img[x, y],  
                     img[x, y+1], img[x, y+2]],  
                    [img[x+1, y-1], img[x+1, y],  
                     img[x+1, y+1], img[x+1, y+2]],  
                    [img[x+2, y-1], img[x+2, y],  
                     img[x+2, y+1], img[x+2, y+1]],  
  
                    ])  
                C = np.array([  
                    [S(1 + q)],  
                    [S(q)],  
                    [S(1 - q)],  
                    [S(2 - q)]  
                ])  
                blue = np.dot(np.dot(A, B[:, :, 0]), C)[0, 0]  
                green = np.dot(np.dot(A, B[:, :, 1]), C)[0, 0]  
                red = np.dot(np.dot(A, B[:, :, 2]), C)[0, 0]  
  
                # ajust the value to be in [0,255]  
                def adjust(value):  
                    if value > 255:  
                        value = 255  
                    elif value < 0:  
                        value = 0  
                    return value  
  
                blue = adjust(blue)  
                green = adjust(green)  
                red = adjust(red)  
                emptyImage[i, j] = np.array([blue, green, red], dtype=np.uint8)  
    return emptyImage   
    
def nearest_neighbor(img,H,W):  
    img = np.array(img,dtype=np.int)
    height,width,channels =img.shape  
    emptyImage=np.zeros((H,W,channels),np.uint8)  
    sh=H/height  
    sw=W/width  
    for i in range(H):  
        for j in range(W):  
            x=int(i/sh)  
            y=int(j/sw)  
            emptyImage[i,j]=img[x,y]  
    return emptyImage  

    
dataset_name = 'San11'
_dir = '/Users/Kingther/百度云同步盘/【★快盘★】/【毕业设计】/DataSet/'
dataset_dir = _dir + dataset_name +'/'
dataset_path = '/Users/Kingther/百度云同步盘/【★快盘★】/【毕业设计】/DataSet/'+ dataset_name +'/'
for img_name in os.listdir(dataset_path): 
    if img_name[-4:] !='.jpg':
        continue
    img_path = dataset_path + img_name #每一个图片的地址
#imgpath = '/Users/Kingther/百度云同步盘/【★快盘★】/【毕业设计】/DataSet/San11/0000_杨奉_1.jpg'
    img_GT = Image.open(img_path)
#img_GT.show()
    img_LR = img_GT.resize((100,100))
#img_LR.show()
    img_HR1=bilinear(img_LR,240,240)  
#Image.fromarray(img_HR).show()
    img_HR2=bicubic(img_LR,240,240)  
#Image.fromarray(img_HR).show()
    img_Res1 = np.array(img_GT,float)-np.array(img_HR1,float)
    img_Res2 = np.array(img_GT,float)-np.array(img_HR2,float)
    img_Res1 = ((img_Res1 +225)//2).astype('uint8')
    img_Res2 = ((img_Res2 +225)//2).astype('uint8')
    Image.fromarray(img_Res1).show()
    Image.fromarray(img_Res2).show()