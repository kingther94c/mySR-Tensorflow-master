#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:50:33 2017

@author: Kingther
"""
#%%
import tensorflow as tf
import input_data_HR as input_data
import math  
import mytf
import time
#from PIL import Image
#import progressbar

#%%
#Dataset
HR_size = (96,96)
crop_size = (96,96)
LR_size = (24,24)

# Dataset Parameter
model_name = 'FSRResCNN_edge_3_0'
#train_dataset_name = 'San11_mini'
#test_dataset_name = 'San10_mini'
train_dataset_name = 'General100'
test_dataset_name = 'Test'
train_dataset_label='train_C96'
test_dataset_label='test_H96'


_dir = './DataSet/'
train_dataset_path = _dir + train_dataset_name +'/'
test_dataset_path = _dir + test_dataset_name +'/'
train_tfrecords_filename = train_dataset_name+"_cropped_"+train_dataset_label+".tfrecords"
test_tfrecords_filename = test_dataset_name+"_"+test_dataset_label+".tfrecords"
BATCH_SIZE = 20
TEST_BATCH_SIZE = 6
# Data.shape = [num,image_height,image_width,image_channel]
checkpoint_dir = './Saver/'+model_name+'/'


input_height = LR_size[0]
input_width = LR_size[1]
input_channel = 3
output_height = HR_size[0]
output_width = HR_size[1]
output_channel = 3
#%%
# CNN Hyperparameters
step_save = 200
f1=5
f2=1
fm=3
f4=1
f5=9

d = 52
s = 12
m = 6
LEARNING_RATE = 1e-3
loss_alpha = 0.84
loss_lambda = 0.03
#%%
#Placeholder
img_LR = tf.placeholder('float', shape=[None, input_height, input_width, input_channel],name = 'img_LR')
img_HR = tf.placeholder('float', shape=[None, output_height, output_width, output_channel],name = 'img_HR')
#%%
img_LR_bl = tf.image.resize_images(img_LR,HR_size,method=tf.image.ResizeMethod.BILINEAR)
img_edge = 1.0*img_HR - 1.0*img_LR_bl

X = img_LR*2.0 /255. - 1
Y = img_edge*2.0 /255.


#1st Layer-FE [Feature Extraction]
with tf.variable_scope("Feature_Extraction"):
    W_FE = mytf.weight_variable([f1, f1, input_channel, d], stddev=0.0378, name='W')
    b_FE = mytf.bias_variable([d], name='b')
    z_FE = mytf.prelu(mytf.conv2d(X, W_FE) + b_FE,1)

#2nd Layer-Shr [Shrinking]
with tf.variable_scope("Shrinking"):
    W_Shr = mytf.weight_variable([f2, f2, d, s], stddev=0.3536, name='W')
    b_Shr = mytf.bias_variable([s], name='b')
    z_Shr = mytf.prelu(mytf.conv2d(z_FE, W_Shr) + b_Shr,1)

#3nd Layer-Map [Mapping]
with tf.variable_scope("Mapping"):
    prev_layer = z_Shr
        # Mapping (# mapping layers = m)
    for i in range(3, m + 3):
        prev_layer = mytf.res_unit(prev_layer,i-2,num_kernels=s,act_func='prelu')
    #    weight_name, bias_name = 'w{}'.format(i), 'b{}'.format(i)
    #    weights[weight_name] = mytf.weight_variable([fm, fm, s, s],stddev=0.1179,name=weight_name)
    #    biases[bias_name] = mytf.bias_variable([s],name=bias_name)
    #    W_Map, b_Map = weights['w{}'.format(i)], biases['b{}'.format(i)]
    #    prev_layer = mytf.prelu(mytf.conv2d(prev_layer, W_Map) + b_Map)

z_Map = prev_layer

#keep_prob = tf.placeholder('float')
#z_Map_dropout = tf.nn.dropout(z_Map, keep_prob)
#z_Map_dropout = z_Map

#4nd Layer-Exp [Expanding]
with tf.variable_scope("Expanding"):
    W_Exp = mytf.weight_variable([f4, f4, s, d], stddev=0.189, name='W')
    b_Exp = mytf.bias_variable([d], name='b')
    z_Exp = mytf.prelu(mytf.conv2d(z_Map, W_Exp) + b_Exp,1)

#5nd Layer-Deconv [Deconvolution]
with tf.variable_scope("Deconvolution"):
    W_Deconv = mytf.weight_variable([f5, f5, output_channel, d], stddev=0.0001, name='W')
    b_Deconv = mytf.bias_variable([output_channel], name='b')
    output_shape = [BATCH_SIZE,output_height,output_width,output_channel]
    Y_ = tf.nn.conv2d_transpose(z_Exp,W_Deconv, output_shape, strides=[1,4,4,1], padding='SAME')+b_Deconv

img_SR = Y_*255. /2.+ img_LR_bl


#%%
#Model and Loss Function
MSE = tf.reduce_mean(tf.square(img_SR - img_HR))
MS_SSIM = mytf.ms_ssim(img_SR,img_HR)


loss_L2 = tf.reduce_mean(tf.square(Y - Y_))
loss_L1 = tf.reduce_mean(tf.abs(Y - Y_))
loss_MS_SSIM = 1 - mytf.ms_ssim(Y,Y_)
loss_SSIM = 1 - mytf.ssim(Y,Y_)
loss_MS_Mix = loss_alpha *loss_L1 + (1-loss_alpha)*loss_MS_SSIM
loss_Mix = loss_alpha *loss_L1 + (1-loss_alpha)*loss_SSIM

#MSE_img = tf.reduce_mean(tf.square(img_SR - img_HR))

#Train & Evaluate
#pSNR = 10*tf.log(255*255/MSE)/tf.log(10.)
#SSIM = mytf.ssim(img_LR,img_HR)
#MS_SSIM = mytf.ms_ssim(img_LR,img_HR)


#%%Train
##############生产数据，仅第一次使用##############
input_data.encode_to_tfrecords_cropped(train_dataset_name, label=train_dataset_label, crop_size=crop_size)
input_data.encode_to_tfrecords(test_dataset_name, label=test_dataset_label, HR_size=HR_size)
##############################################
def train(step_save=step_save,step_train=2000,train_continue = False,learning_rate = LEARNING_RATE,loss='L2',loss_lambda = loss_lambda,op = 'Adam'):
#    globaltime = time.clock()
    loss_regularizer = tf.contrib.layers.l2_regularizer(loss_lambda)(W_Deconv) + tf.contrib.layers.l2_regularizer(loss_lambda)(W_Exp) + tf.contrib.layers.l2_regularizer(loss_lambda)(W_FE) + tf.contrib.layers.l2_regularizer(loss_lambda)(W_Shr)
    if loss.upper() == 'L1':
        loss_func = loss_L1
    if loss.upper() == 'L2' or loss.upper() == 'MSE':
        loss_func = loss_L2
    if loss.upper() == 'SSIM':
        loss_func = loss_SSIM
    if loss.upper() == 'MS-SSIM' or loss.upper() == 'MS_SSIM' or loss.upper() == 'MSSSIM':
        loss_func = loss_MS_SSIM
    if loss.upper() == 'MIX':
        loss_func = loss_Mix
    if loss.upper() == 'MS_MIX' or loss.upper() == 'MS-MIX':
        loss_func = loss_MS_Mix
    loss_func = loss_func + loss_regularizer
#    globaltime = time.clock()
    if op == 'Adam':
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_func)
    elif op == 'Momentum':       
        train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss_func)
    elif op == 'AdaGrad':       
        train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss_func)
    elif op == 'RMSProp':       
        train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(loss_func)
        
    saver = tf.train.Saver(tf.trainable_variables())
    
    img_HRs_train = input_data.decode_from_tfrecords(train_tfrecords_filename,HR_size = HR_size)
    img_HRs_aug_train = input_data.data_augment(img_HRs_train,crop_size)
    img_HR_batch_train = input_data.get_batch(img_HRs_aug_train, batch_size=BATCH_SIZE)
    
    init=tf.global_variables_initializer()  
    
    
    #Progressbar
     
    
    with tf.Session() as sess: 
        sess.run(init)
        if train_continue==True:
            current_step = mytf.load(sess, saver, model_name, train_dataset_name)
        else:
            current_step = 0
        #progress = progressbar.ProgressBar()
        #pbar = progress.start()       
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess = sess,coord=coord)
        lasttime = time.clock()
        
        #pbar = progress.start( )
        for i in range(current_step+1,current_step+step_train+1):#每run一次，就会指向下一个样本，一直循环  
            img_HR_ = sess.run(img_HR_batch_train)
            img_LR_ = tf.image.resize_images(img_HR_,LR_size).eval().astype('uint8')
            

            train_op.run(feed_dict={img_LR:img_LR_, img_HR:img_HR_})
            if (i-current_step)%10==0:
                time_interval = time.clock()-lasttime
                #train_MSE = MSE.eval(feed_dict={img_HR:img_HR_})
                train_MSE = MSE.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_})
                train_loss = loss_func.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_})
#                if train_MSE_prev< 9998 and abs(train_MSE_prev-train_MSE)/train_MSE_prev<0.05:
#                    learning_rate = 0.5*learning_rate
#                    train_op = tf.train.AdamOptimizer(learning_rate).minimize(MSE)
                #train_pSNR = pSNR.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 1.0})
                train_pSNR = 10*math.log(255*255/train_MSE,10)
                train_MS_SSIM = MS_SSIM.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_}) 
                #pbar.update(int(((i-current_step)/(step_train))*100))
                print('step %d, Loss %g, pSNR %g, MSE %g, MS-SSIM %g'%(i, train_loss, train_pSNR, train_MSE, train_MS_SSIM))
                mytf.record(i,model_name, train_dataset_name, train_pSNR, train_MS_SSIM, train_MSE,time_interval)
                lasttime = time.clock()
            
            if (i-current_step)%step_save ==0:
                mytf.save(sess, saver, model_name, train_dataset_name, i)
        
        #pbar.finish()
        coord.request_stop()#queue需要关闭，否则报错  
        coord.join(threads)  
#        print(time.clock()-globaltime)
        
#%%Test    
def test(test_batch_size = TEST_BATCH_SIZE):
#test_batch_size = TEST_BATCH_SIZE

    saver = tf.train.Saver(tf.trainable_variables())
    img_HRs_test = input_data.decode_from_tfrecords(test_tfrecords_filename,HR_size = HR_size)
    #img_HR_batch_test, img_LR_batch_test = input_data.get_batch(img_HRs_test, img_LRs_test, batch_size=test_batch_size, shuffle = False)
    img_HR_batch_test = input_data.get_batch(img_HRs_test, batch_size=test_batch_size, shuffle = False)
    with tf.Session() as sess:
        current_step = mytf.load(sess, saver, model_name, train_dataset_name)
        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(sess = sess,coord=coord)
        gen_time0 = time.clock()
        img_HR_test = sess.run(img_HR_batch_test)
        img_LR_test = tf.image.resize_images(img_HR_test,LR_size).eval().astype('uint8')
        img_bl = tf.image.resize_images(img_LR_test,HR_size,method=tf.image.ResizeMethod.BILINEAR)
        img_bc = tf.image.resize_images(img_LR_test,HR_size,method=tf.image.ResizeMethod.BICUBIC)
        img_LR_show = tf.image.resize_images(img_LR_test,HR_size,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img_bl = img_bl.eval()
        img_bc = img_bc.eval()
        img_LR_show = img_LR_show.eval()
        gen_time_per_img= (time.clock()-gen_time0)/TEST_BATCH_SIZE
        test_MSE = MSE.eval(feed_dict={img_LR:img_LR_test, img_HR:img_HR_test})
        #train_pSNR = pSNR.eval(feed_dict={img_LR:img_LR_, img_HR:img_HR_, keep_prob: 1.0})
        test_pSNR = 10*math.log(255*255/test_MSE,10)
        test_MS_SSIM = MS_SSIM.eval(feed_dict={img_LR:img_LR_test, img_HR:img_HR_test})
        print('test on %g images, pSNR %g, MSE %g, MS-SSIM %g, time_per_img %g'%(test_batch_size, test_pSNR, test_MSE, test_MS_SSIM,gen_time_per_img))
        coord.request_stop()#queue需要关闭，否则报错  
        coord.join(threads)
        img_SR_gen = img_SR.eval(feed_dict={img_LR:img_LR_test,img_HR:img_HR_test})
        img_SR_gen[img_SR_gen<=0]=0
        img_SR_gen[img_SR_gen>=255]=255
        img_bl[img_bl>=255]=255
        img_bl[img_bl<=0]=0
        img_bc[img_bc>=255]=255
        img_bc[img_bc<=0]=0
        img_LR_show = img_LR_show.astype('uint8')
        img_bl = img_bl.astype('uint8')
        img_bc = img_bc.astype('uint8')
        img_SR_gen = img_SR_gen.astype('uint8')
        #mytf.plot_imagegroups(img_LR_show,img_SR_gen,img_HR_test,num_img = test_batch_size)
        mytf.plot_allimage(img_LR_show,img_SR_gen,img_bl,img_bc,img_HR_test,num_img = test_batch_size)

        return img_LR_show,img_SR_gen,img_bl,img_bc,img_HR_test
