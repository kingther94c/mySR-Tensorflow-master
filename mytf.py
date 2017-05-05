#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 09:52:11 2017

@author: Kingther
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import csv
import math
from PIL import Image
#%% Model Constructor
def weight_variable(shape, name = None, stddev=0.1 ): #shape = [patch_height, patch_width, in_channels, out_channels]
    initial = tf.truncated_normal(shape, stddev = stddev, name=name)
    return tf.Variable(initial)

def bias_variable(shape, name = None ,value=0.0):
    initial = tf.constant(value, shape=shape, name = name)
    return tf.Variable(initial)

def prelu(_x,i=0):
    """
    PreLU tensorflow implementation
    """
    #alphas = tf.get_variable('alpha{}'.format(i), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    alphas = tf.Variable(tf.constant(0.0, shape=[_x.get_shape()[-1]]))
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

def conv2d(x, W,strides=[1, 1, 1, 1]):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding='SAME')

def deconv2d(x, W, output_shape,strides=[1, 1, 1, 1]):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=strides, padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def res_unit(input_layer,i,num_kernels=64,act_func='prelu'):
    if act_func == 'prelu':
        act_func_ = prelu
    if act_func == 'relu':
        act_func_ = tf.nn.relu
    with tf.variable_scope("Res_Unit"+str(i)):
        part1 = slim.batch_norm(input_layer,activation_fn=None)
        part2 = act_func_(part1)
        part3 = slim.conv2d(part2,num_kernels,[3,3],activation_fn=None)
        part4 = slim.batch_norm(part3,activation_fn=None)
        part5 = act_func_(part4)
        part6 = slim.conv2d(part5,num_kernels,[3,3],activation_fn=None)
        output = input_layer + part6
        return output
    
def whitening(img):
    #tf.image.per_image_standardization()
    return

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    img1 = tf.image.rgb_to_grayscale(img1)
    img2 = tf.image.rgb_to_grayscale(img2)
    window = fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def ms_ssim(img1, img2, mean_metric=True, level=3):
    img1 = tf.image.rgb_to_grayscale(img1)
    img2 = tf.image.rgb_to_grayscale(img2)
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value
        
        
        
        
#%% Save & Load & Record
def save(sess, saver, model_name, dataset_name, step):
    checkpoint_dir1 = model_name+"_checkpoint/"
    checkpoint_dir2 = model_name+"_checkpoint/"+dataset_name+'/'
    checkpoint_name = model_name +'_'+dataset_name+'.ckpt'
    if not os.path.exists(checkpoint_dir1):
        os.makedirs(checkpoint_dir1)
    if not os.path.exists(checkpoint_dir2):
        os.makedirs(checkpoint_dir2)
    saver.save(sess,checkpoint_dir2+checkpoint_name,global_step=step)

def load(sess, saver, model_name, dataset_name):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = model_name+"_checkpoint/"+dataset_name+'/'
    if not os.path.exists(checkpoint_dir):
        print('Not Found: checkpoints')
        return 0
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, checkpoint_dir+ckpt_name)
        return int(ckpt.model_checkpoint_path[ckpt.model_checkpoint_path.index('ckpt-')+5:])
    else:
        print('Not Found: checkpoints')
        return 0

def record(step,model_name, dataset_name,pSNR,MS_SSIM,MSE,time_interval):
    with open('Record_'+model_name+'_On_'+dataset_name,'a') as csvfile:
        csvwriter = csv.writer(csvfile,dialect='excel',delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow([str(step),str(pSNR),str(MS_SSIM),str(MSE),str(time_interval)])

def plot_imagepairs(img_SR,img_HR,num_img =5,num_col=1):
    #plt.figure(figsize=(10,10)) 
    num_row = math.ceil(num_img/num_col)
    fig = plt.figure(figsize=(5*2*num_col,5.3*num_row))
    for i in np.arange(0, num_img):
        ax2 = fig.add_subplot(num_row,num_col*2, 2*i + 1)
        ax2.imshow(img_SR[i])
        ax2.set_title('Output: Super Resolution')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(num_row,num_col*2, 2*i + 2)
        ax3.imshow(img_HR[i])
        ax3.set_title('Ground Truth')
        ax3.axis('off')
    #fig.show()
    #plt.close()
    
def plot_imagegroups(img_LR,img_SR,img_HR,num_img =5,num_col=1):
    '''plot one batch size
    '''
    #plt.figure(figsize=(10,10)) 
    num_row = math.ceil(num_img/num_col)
    fig = plt.figure(figsize=(5*3*num_col,5.3*num_row))
    for i in np.arange(0, num_img):
        ax1 = fig.add_subplot(num_row,num_col*3, 3*i + 1)
        ax1.imshow(img_LR[i])
        ax1.set_title('Input: Low Resolution')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(num_row,num_col*3, 3*i + 2)
        ax2.imshow(img_SR[i])
        ax2.set_title('Output: Super Resolution')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(num_row,num_col*3, 3*i + 3)
        ax3.imshow(img_HR[i])
        ax3.set_title('Ground Truth')
        ax3.axis('off')
        
    #fig.show()
    #plt.close()
def plot_allimage(img_LR,img_SR,img_bl,img_bc,img_HR,num_img =5,num_col=1):
    '''plot one batch size
    '''
    #plt.figure(figsize=(10,10)) 
    num_row = math.ceil(num_img/num_col)
    fig = plt.figure(figsize=(5*5*num_col,5.3*num_row))
    for i in np.arange(0, num_img):
        ax1 = fig.add_subplot(num_row,num_col*5, 5*i + 1)
        ax1.imshow(img_LR[i])
        ax1.set_title('Input: Low Resolution')
        ax1.axis('off')

        ax2 = fig.add_subplot(num_row,num_col*5, 5*i + 2)
        ax2.imshow(img_SR[i])
        ax2.set_title('Output: Super Resolution')
        ax2.axis('off')
        
        ax4 = fig.add_subplot(num_row,num_col*5, 5*i + 3)
        ax4.imshow(img_bl[i])
        ax4.set_title('Bilinear')
        ax4.axis('off')
        
        ax5 = fig.add_subplot(num_row,num_col*5, 5*i + 4)
        ax5.imshow(img_bc[i])
        ax5.set_title('Bicubic')
        ax5.axis('off')

        ax3 = fig.add_subplot(num_row,num_col*5, 5*i + 5)
        ax3.imshow(img_HR[i])
        ax3.set_title('Ground Truth')
        ax3.axis('off')
        
    #fig.show()
    #plt.close()        