# coding: utf-8
from __future__ import print_function
import os
import time
import random
#from skimage import color
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob

batch_size = 10
patch_size = 48

sess = tf.Session()
#the input of decomposition net
input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
#the input of illumination adjustment net
input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
input_low_i_ratio = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i_ratio')
input_high_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_high_i')

[R_decom, I_decom] = DecomNet_simple(input_decom)
#the output of decomposition network
decom_output_R = R_decom
decom_output_I = I_decom
#the output of illumination adjustment net
output_i = Illumination_adjust_net(input_low_i, input_low_i_ratio)

#define loss

def grad_loss(input_i_low, input_i_high):
    x_loss = tf.square(gradient(input_i_low, 'x') - gradient(input_i_high, 'x'))
    y_loss = tf.square(gradient(input_i_low, 'y') - gradient(input_i_high, 'y'))
    grad_loss_all = tf.reduce_mean(x_loss + y_loss)
    return grad_loss_all

loss_grad = grad_loss(output_i, input_high_i)
loss_square = tf.reduce_mean(tf.square(output_i  - input_high_i))# * ( 1 - input_low_r ))#* (1- input_low_i)))

loss_adjust =  loss_square + loss_grad 

lr = tf.placeholder(tf.float32, name='learning_rate')

optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')

var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_adjust = [var for var in tf.trainable_variables() if 'Illumination_adjust_net' in var.name]

saver_adjust = tf.train.Saver(var_list=var_adjust)
saver_Decom = tf.train.Saver(var_list = var_Decom)
train_op_adjust = optimizer.minimize(loss_adjust, var_list = var_adjust)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")

### load data
### Based on the decomposition net, we first get the decomposed reflectance maps 
### and illumination maps, then train the adjust net.
###train_data
train_low_data = []
train_high_data = []
train_low_data_names = glob('./LOLdataset/our485/low/*.png') 
train_low_data_names.sort()
train_high_data_names = glob('./LOLdataset/our485/high/*.png') 
train_high_data_names.sort()
assert len(train_low_data_names) == len(train_high_data_names)
print('[*] Number of training data: %d' % len(train_low_data_names))
for idx in range(len(train_low_data_names)):
    low_im = load_images(train_low_data_names[idx])
    train_low_data.append(low_im)
    high_im = load_images(train_high_data_names[idx])
    train_high_data.append(high_im)

pre_decom_checkpoint_dir = './checkpoint/decom_net_train/'
ckpt_pre=tf.train.get_checkpoint_state(pre_decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No pre_decom_net checkpoint!')

#decomposed_low_r_data_480 = []
decomposed_low_i_data_480 = []
#decomposed_high_r_data_480 = []
decomposed_high_i_data_480 = []
for idx in range(len(train_low_data)):
    input_low = np.expand_dims(train_low_data[idx], axis=0)
    RR, II = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low})
    RR0 = np.squeeze(RR)
    II0 = np.squeeze(II)
    print(RR0.shape, II0.shape)
    #decomposed_high_r_data_480.append(result_1_sq)
    decomposed_low_i_data_480.append(II0)
for idx in range(len(train_high_data)):
    input_high = np.expand_dims(train_high_data[idx], axis=0)
    RR2, II2 = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_high})
    RR02 = np.squeeze(RR2)
    II02 = np.squeeze(II2)
    print(RR02.shape, II02.shape)
    #decomposed_high_r_data_480.append(result_1_sq)
    decomposed_high_i_data_480.append(II02)

eval_adjust_low_i_data = decomposed_low_i_data_480[451:480]
eval_adjust_high_i_data = decomposed_high_i_data_480[451:480]

train_adjust_low_i_data = decomposed_low_i_data_480[0:450]
train_adjust_high_i_data = decomposed_high_i_data_480[0:450]

print('[*] Number of training data: %d' % len(train_adjust_high_i_data))

learning_rate = 0.0001
epoch = 2000
eval_every_epoch = 200
train_phase = 'adjustment'
numBatch = len(train_adjust_low_i_data) // int(batch_size)
train_op = train_op_adjust
train_loss = loss_adjust
saver = saver_adjust

checkpoint_dir = './checkpoint/illumination_adjust_net_train/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)
else:
    print("No adjustment net pre model!")

start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))

sample_dir = './illumination_adjust_net_train/'
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

start_time = time.time()
image_id = 0

for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_low_i_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_high_i_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_low_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        batch_input_high_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        input_low_i_rand = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        input_high_i_rand = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        input_low_i_rand_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")
        input_high_i_rand_ratio = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

        for patch_id in range(batch_size):
            i_low_data = train_adjust_low_i_data[image_id]
            i_low_expand = np.expand_dims(i_low_data, axis = 2)
            i_high_data = train_adjust_high_i_data[image_id]
            i_high_expand = np.expand_dims(i_high_data, axis = 2)

            h, w = train_adjust_low_i_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            i_low_data_crop = i_low_expand[x : x+patch_size, y : y+patch_size, :]
            i_high_data_crop = i_high_expand[x : x+patch_size, y : y+patch_size, :]

            rand_mode = np.random.randint(0, 7)
            batch_input_low_i[patch_id, :, :, :] = data_augmentation(i_low_data_crop , rand_mode)
            batch_input_high_i[patch_id, :, :, :] = data_augmentation(i_high_data_crop, rand_mode)

            ratio = np.mean(i_low_data_crop/(i_high_data_crop+0.0001))
            #print(ratio)
            i_low_data_ratio = np.ones([patch_size,patch_size])*(1/ratio+0.0001)
            i_low_ratio_expand = np.expand_dims(i_low_data_ratio , axis =2)
            i_high_data_ratio = np.ones([patch_size,patch_size])*(ratio)
            i_high_ratio_expand = np.expand_dims(i_high_data_ratio , axis =2)
            batch_input_low_i_ratio[patch_id, :, :, :] = i_low_ratio_expand
            batch_input_high_i_ratio[patch_id, :, :, :] = i_high_ratio_expand

            rand_mode = np.random.randint(0, 2)
            if rand_mode == 1:
                input_low_i_rand[patch_id, :, :, :] = batch_input_low_i[patch_id, :, :, :]
                input_high_i_rand[patch_id, :, :, :] = batch_input_high_i[patch_id, :, :, :]
                input_low_i_rand_ratio[patch_id, :, :, :] = batch_input_low_i_ratio[patch_id, :, :, :]
                input_high_i_rand_ratio[patch_id, :, :, :] = batch_input_high_i_ratio[patch_id, :, :, :]
            else:
                input_low_i_rand[patch_id, :, :, :] = batch_input_high_i[patch_id, :, :, :]
                input_high_i_rand[patch_id, :, :, :] = batch_input_low_i[patch_id, :, :, :]
                input_low_i_rand_ratio[patch_id, :, :, :] = batch_input_high_i_ratio[patch_id, :, :, :]
                input_high_i_rand_ratio[patch_id, :, :, :] = batch_input_low_i_ratio[patch_id, :, :, :]

            image_id = (image_id + 1) % len(train_adjust_low_i_data)

        _, loss = sess.run([train_op, train_loss], feed_dict={input_low_i: input_low_i_rand,input_low_i_ratio: input_low_i_rand_ratio,\
                                                              input_high_i: input_high_i_rand, \
                                                              lr: learning_rate})
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        iter_num += 1
    if (epoch + 1) % eval_every_epoch == 0:
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
        
        for idx in range(10):
            rand_idx = idx#np.random.randint(26)
            input_uu_i = eval_adjust_low_i_data[rand_idx] 
            input_low_eval_i = np.expand_dims(input_uu_i, axis=0)
            input_low_eval_ii = np.expand_dims(input_low_eval_i, axis=3)
            h, w = eval_adjust_low_i_data[idx].shape
            rand_ratio = np.random.random(1)*2
            input_uu_i_ratio = np.ones([h,w]) * rand_ratio 
            input_low_eval_i_ratio = np.expand_dims(input_uu_i_ratio, axis=0)
            input_low_eval_ii_ratio = np.expand_dims(input_low_eval_i_ratio, axis=3)
            
            result_1 = sess.run(output_i, feed_dict={input_low_i: input_low_eval_ii, input_low_i_ratio: input_low_eval_ii_ratio})
            save_images(os.path.join(sample_dir, 'h_eval_%d_%d_%5f.png' % ( epoch + 1 , rand_idx + 1,   rand_ratio)), input_uu_i, result_1)
        

    saver.save(sess, checkpoint_dir + 'model.ckpt')

print("[*] Finish training for phase %s." % train_phase)



