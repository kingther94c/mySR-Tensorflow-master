#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:50:33 2017

@author: Kingther
"""
#%%
import tensorflow as tf
import input_data
import math  
import mytf
import time
from PIL import Image

#%%
#Dataset
# Dataset Parameter
model_name = 'SRCNN_1.0_'
train_dataset_name = 'San11'
test_dataset_name = 'San10'
_dir = '/Users/Kingther/百度云同步盘/【★快盘★】/【毕业设计】/DataSet/'
train_dataset_path = _dir + train_dataset_name +'/'
test_dataset_path = _dir + test_dataset_name +'/'
train_tfrecords_filename = train_dataset_name+"_train.tfrecords"
test_tfrecords_filename = test_dataset_name+"_test.tfrecords"
BATCH_SIZE = 30
TEST_BATCH_SIZE = 30
# Data.shape = [num,image_height,image_width,image_channel]
checkpoint_dir = './Saver/SRCNN/'
input_height = 128
input_width = 128
input_channel = 3
output_height = 128
output_width = 128
output_channel = 3
#%%
# CNN Hyperparameters
step_save = 500
f1 = 9 
f2 = 1
f3 = 5
n1 = 64
n2 = 32

#%%
#Placeholder
img_LR = tf.placeholder('float', shape=[None, input_height, input_width, input_channel])
img_HR = tf.placeholder('float', shape=[None, output_height, output_width, output_channel])
#%%
#img_LR__ = img_LR*2.0 /255 - 1
#img_HR__ = img_HR*2.0 /255 - 1
#1st Layer-conv1 [Patch extraction and representation]
W_conv1 = mytf.weight_variable([f1, f1, input_channel, n1])
b_conv1 = mytf.bias_variable([n1])
#z_conv1 = tf.nn.relu(mytf.conv2d(img_LR, W_conv1) + b_conv1)
z_conv1 = mytf.prelu(mytf.conv2d(img_LR, W_conv1) + b_conv1,1)

#2nd Layer-conv2 [Non-linear mapping]
W_conv2 = mytf.weight_variable([f2, f2, n1, n2])
b_conv2 = mytf.bias_variable([n2])
#z_conv2 = tf.nn.relu(mytf.conv2d(z_conv1, W_conv2) + b_conv2)
z_conv2 = mytf.prelu(mytf.conv2d(z_conv1, W_conv2) + b_conv2,2)
#Dropout
keep_prob = tf.placeholder('float')
z_conv2_dropout = tf.nn.dropout(z_conv2, keep_prob)

#3nd Layer-conv2 [Reconstruction]
W_conv3 = mytf.weight_variable([f3, f3, n2, output_channel])
b_conv3 = mytf.bias_variable([input_channel])
#img_SR = tf.nn.relu(mytf.conv2d(z_conv2_dropout, W_conv3) + b_conv3)
img_SR = mytf.prelu(mytf.conv2d(z_conv2_dropout, W_conv3) + b_conv3,3)

#%%
#Model and Loss Function
MSE = tf.reduce_mean(tf.square(img_SR - img_HR))
train_step = tf.train.AdamOptimizer(1e-4).minimize(MSE)

#Train & Evaluate
#pSNR = 10*tf.log(255*255/MSE)/tf.log(10.)
#SSIM = mytf.ssim(img_LR,img_HR)
#MS_SSIM = mytf.ms_ssim(img_LR,img_HR)


#%%Train
##############生产数据，仅第一次使用##############
#input_data.convert_to_tfrecord(train_dataset_name, label='train' )
#input_data.convert_to_tfrecord(test_dataset_name, label='test' )
##############################################
def train(step_save=step_save,step_train=2000):
    saver = tf.train.Saver(tf.trainable_variables())
    img_HRs_train, img_LRs_train = input_data.read_and_decode(train_tfrecords_filename)
    img_HR_batch_train, img_LR_batch_train = input_data.get_batch(img_HRs_train, img_LRs_train, batch_size=BATCH_SIZE)
    
    init=tf.global_variables_initializer()  
    
    with tf.Session() as sess:  
        sess.run(init)  
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess = sess,coord=coord)
        lasttime = time.clock()
        for i in range(step_train+1):#每run一次，就会指向下一个样本，一直循环  
            img_HR_, img_LR_ = sess.run([img_HR_batch_train, img_LR_batch_train]) 
            train_step.run(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 0.5})
            if i%10==0:
                time_interval = time.clock()-lasttime
                train_MSE = MSE.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 1.0})
                #train_pSNR = pSNR.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 1.0})
                train_pSNR = 10*math.log(255*255/train_MSE,10)
                train_SSIM = -1
                #train_SSIM = SSIM.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 1.0})
                train_MS_SSIM =-1
                #train_MS_SSIM = MS_SSIM.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 1.0})
                print('step %d, pSNR %g, MSE %g, SSIM %g, MS-SSIM %g'%(i, train_pSNR, train_MSE, train_SSIM, train_MS_SSIM))
                mytf.record(model_name, train_dataset_name, train_pSNR, train_SSIM, train_MS_SSIM, train_MSE,i,time_interval)
                lasttime = time.clock()
            
            if i%step_save ==0:
                mytf.save(sess, saver, model_name, train_dataset_name, i)
        coord.request_stop()#queue需要关闭，否则报错  
        coord.join(threads)  
        
#%%Test    
def test(test_batch_size = TEST_BATCH_SIZE):
    saver = tf.train.Saver(tf.trainable_variables())
    img_HRs_test, img_LRs_test = input_data.read_and_decode(test_tfrecords_filename)
    #img_HR_batch_test, img_LR_batch_test = input_data.get_batch(img_HRs_test, img_LRs_test, batch_size=test_batch_size, shuffle = False)
    img_HR_batch_test, img_LR_batch_test = input_data.get_batch(img_HRs_test, img_LRs_test, batch_size=test_batch_size)
    with tf.Session() as sess:
        mytf.load(sess, saver, model_name, train_dataset_name)
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess = sess,coord=coord)
        gen_time0 = time.clock()
        img_HR_test, img_LR_test = sess.run([img_HR_batch_test, img_LR_batch_test]) 
        img_SR.eval(feed_dict={img_LR:img_LR_test, img_HR:img_HR_test, keep_prob: 1.0})
        gen_time_per_img= (time.clock()-gen_time0)/TEST_BATCH_SIZE
        test_MSE = MSE.eval(feed_dict={img_LR:img_LR_test, img_HR:img_HR_test, keep_prob: 1.0})
        #train_pSNR = pSNR.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 1.0})
        test_pSNR = 10*math.log(255*255/test_MSE,10)
        test_SSIM = -1
        #test_SSIM = SSIM.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 1.0})
        test_MS_SSIM =-1
        #test_MS_SSIM = MS_SSIM.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 1.0})
        print('test on %g images, pSNR %g, MSE %g, SSIM %g, MS-SSIM %g, time_per_img %g'%(test_batch_size, test_pSNR, test_MSE, test_SSIM, test_MS_SSIM,gen_time_per_img))
        coord.request_stop()#queue需要关闭，否则报错  
        coord.join(threads)
        img_SR_gen = img_SR.eval(feed_dict={img_LR:img_LR_test, keep_prob: 1.0})
        img_SR_gen[img_SR_gen<=0]=0
        img_SR_gen[img_SR_gen>=255]=255
        img_SR_gen = img_SR_gen.astype('uint8')
        mytf.plot_imagegroups(img_LR_test,img_SR_gen,img_HR_test,num_img = test_batch_size)
        return img_LR_test,img_SR_gen,img_HR_test

#%% For Debug
'''
img_HRs_test, img_LRs_test = input_data.read_and_decode(test_tfrecords_filename)
img_HR_batch_test, img_LR_batch_test = input_data.get_batch(img_HRs_test, img_LRs_test, batch_size=BATCH_SIZE,shuffle = False)

with tf.Session() as sess:  
    coord = tf.train.Coordinator()  
    threads = tf.train.start_queue_runners(sess = sess,coord=coord)
    for i in range(101):#每run一次，就会指向下一个样本，一直循环  
        img_HR_, img_LR_ = sess.run([img_HR_batch_test, img_LR_batch_test]) 
        print('step %d'%(i))
    coord.request_stop()#queue需要关闭，否则报错  
    coord.join(threads)  
'''