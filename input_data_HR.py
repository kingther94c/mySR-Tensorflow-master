#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:10:34 2017

@author: Kingther
"""
#%%
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import progressbar

#%%
# Parameter
_dir = './DataSet/'
HR_SIZE = (240,240)
BATCH_SIZE = 18
CROP_SIZE = (96,96)
#for test:
'''
dataset_name = 'San10'
dataset_path = _dir + dataset_name +'/'
tfrecords_filename = dataset_name+"_train.tfrecords"
'''
#%%
def encode_to_tfrecords(dataset_name, save_dir = "TFRecord_DataSet/", label='train' ,HR_size = HR_SIZE):
    
    name=dataset_name+"_"+label+".tfrecords"
    dataset_path = _dir + dataset_name +'/'
    filename = save_dir +  name
    
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for img_name in os.listdir(dataset_path): 
        if img_name[-4:] not in ['.jpg','.bmp'] :
            continue
        img_path = dataset_path + img_name
#        img_path = '/Users/Kingther/百度云同步盘/【★快盘★】/【毕业设计】/DataSet/San11/0000_杨奉_1.jpg'
        try:
            img = Image.open(img_path) #PIL.Image---RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            ##图片处理
            #img_distorted = tf.random_crop(img, [crop_size, crop_size, 3])#随机裁剪  
            #img_distorted = tf.image.random_flip_up_down(img_distorted)#上下随机翻转  
            #img_distorted = tf.image.random_brightness(img_distorted,max_delta=63)#亮度变化  
            #img_distorted = tf.image.random_contrast(img_distorted,lower=0.2, upper=1.8)#对比度变化 
            #img = img_distorted 
            ##
            #img = cv2.imread(img_path) #opencv---BGR
        
            img_HR = img.resize(HR_size)
            img_HR_raw = np.array(img_HR).tobytes()#将图片转化为二进制格式

            example = tf.train.Example(features=tf.train.Features(feature={
                'img_HR_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_HR_raw]))
            })) #dataset对象对img_HR和img_LR数据进行封装
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', img_name)
            print('error: %s' %e)
            print('Skip it!\n')
            
    writer.close()
    print('Transform done!')
    
    
def encode_to_tfrecords_cropped(dataset_name, save_dir = "TFRecord_DataSet/", label='train' ,crop_size = CROP_SIZE):
    name=dataset_name+"_cropped_"+label+".tfrecords"
    dataset_path = _dir + dataset_name +'/'
    filename = save_dir +  name
    # wait some time here, transforming need some time based on the size of your data.
    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for img_name in os.listdir(dataset_path): 
        if img_name[-4:] not in ['.jpg','.bmp'] :
            continue
        img_path = dataset_path + img_name
        #print(img_name)
#        img_path = '/Users/Kingther/百度云同步盘/【★快盘★】/【毕业设计】/MySR/DataSet/San11/0000_杨奉_1.jpg'
        try:
            img = Image.open(img_path) #PIL.Image---RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = np.array(img)
            for repeat in range((img.shape[0]//CROP_SIZE[0])*(img.shape[1]//CROP_SIZE[1]) +1):
            #for repeat in range(3):
                img_crop = tf.random_crop(img, [crop_size[0], crop_size[1], 3])
                with tf.Session() as sess:
                    img_HR = sess.run(img_crop)
                img_HR_raw = np.array(img_HR).tobytes()#将图片转化为二进制格式
    
                example = tf.train.Example(features=tf.train.Features(feature={
                    'img_HR_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_HR_raw]))
                })) 
                writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', img_name)
            print('error: %s' %e)
            print('Skip it!\n')
            
    writer.close()
    print('Transform done!')
#%%
# Read Data from TFRecord


def decode_from_tfrecords(tfrecords_filename,HR_size = HR_SIZE):
    '''read and decode tfrecord file
    Args:
        tfrecords_file: the directory of tfrecord file
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
    '''
    # make an input queue from the tfrecord file
    filepath = "TFRecord_DataSet/"+tfrecords_filename
    filename_queue = tf.train.string_input_producer([filepath])
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
                                        serialized_example,
                                        features={
                                           'img_HR_raw': tf.FixedLenFeature([], tf.string)})
    img_HR = tf.decode_raw(features['img_HR_raw'], tf.uint8)
    
    ##########################################################
    # you can put data augmentation here, I didn't use it
    ##########################################################
    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.
    
    img_HRs = tf.reshape(img_HR, [HR_size[0], HR_size[1], 3])  
    #img_HR = tf.cast(img_HR, tf.float32) * (1. / 255) - 0.5 
    #img_LR = tf.cast(img_LR, tf.float32) * (1. / 255) - 0.5  

  

    return img_HRs

#%%
def data_augment(batch_from_tfrecords,crop_size=CROP_SIZE):
    img = batch_from_tfrecords
    #img_distorted = tf.random_crop(img, [crop_size[0], crop_size[1], 3])#随机裁剪  
    #img_distorted = tf.image.random_flip_up_down(img_distorted)#上下随机翻转  
    img_distorted = tf.image.random_flip_left_right(img)
    img_distorted = tf.image.random_brightness(img_distorted,max_delta=0.25)#亮度变化  
    img_distorted = tf.image.random_contrast(img_distorted,lower=0.9, upper=1.1)#对比度变化 
    return img_distorted
        
#%%
def get_batch(img_HRs,batch_size= BATCH_SIZE,num_threads=32,shuffle = True,capacity = 1000):
    if shuffle == True:
        img_HR_batch = tf.train.shuffle_batch([img_HRs],
                                            batch_size= batch_size,
                                            num_threads= num_threads, 
                                            min_after_dequeue = 100,
                                            capacity = capacity)
    else:
        img_HR_batch = tf.train.batch([img_HRs],
                                            batch_size= batch_size,
                                            num_threads= num_threads,
                                            allow_smaller_final_batch = True,
                                            capacity = capacity)    
    return img_HR_batch
#%% Convert data to TFRecord
#convert_to_tfrecord(dataset_name)

#%% TO test train.tfrecord file
def plot_image(img_SR,num_img =5,num_col=1):
    #plt.figure(figsize=(10,10)) 
    num_row = (num_img//num_col)
    fig = plt.figure(figsize=(5*num_col,5.3*num_row))
    for i in np.arange(0, num_img):
        ax2 = fig.add_subplot(num_row,num_col, i + 1)
        ax2.imshow(img_SR[i])
        ax2.set_title('Img')
        ax2.axis('off')

tfrecords_filename = 'General6_cropped_train_C96.tfrecords'
#encode_to_tfrecords_cropped("General6", save_dir = "TFRecord_DataSet/", label='cropped_train_C96' ,crop_size = CROP_SIZE)
def test(tfrecords_filename=tfrecords_filename):
    img_HRs = decode_from_tfrecords(tfrecords_filename,HR_size = (96,96))
    img_HR_batch = get_batch(img_HRs,batch_size= BATCH_SIZE,num_threads=32,shuffle = True)
    with tf.Session()  as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img_HR= sess.run(img_HR_batch)
        plot_image(img_HR)
        coord.request_stop()
        coord.join(threads)
