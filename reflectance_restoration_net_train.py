# coding: utf-8
from __future__ import print_function
import os
import time
import random
from PIL import Image
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from glob import glob

batch_size = 4
patch_size = 384

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)
#the input of decomposition net
input_decom = tf.placeholder(tf.float32, [None, None, None, 3], name='input_decom')
#restoration input
input_low_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low_r')
input_low_i = tf.placeholder(tf.float32, [None, None, None, 1], name='input_low_i')
input_high_r = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high_r')

[R_decom, I_decom] = DecomNet_simple(input_decom)
#the output of decomposition network
decom_output_R = R_decom
decom_output_I = I_decom

output_r = Restoration_net(input_low_r, input_low_i)

#define loss
def grad_loss(input_r_low, input_r_high):
    input_r_low_gray = tf.image.rgb_to_grayscale(input_r_low)
    input_r_high_gray = tf.image.rgb_to_grayscale(input_r_high)
    x_loss = tf.square(gradient(input_r_low_gray, 'x') - gradient(input_r_high_gray, 'x'))
    y_loss = tf.square(gradient(input_r_low_gray, 'y') - gradient(input_r_high_gray, 'y'))
    grad_loss_all = tf.reduce_mean(x_loss + y_loss)
    return grad_loss_all

def ssim_loss(output_r, input_high_r):
    output_r_1 = output_r[:,:,:,0:1]
    input_high_r_1 = input_high_r[:,:,:,0:1]
    ssim_r_1 = tf_ssim(output_r_1, input_high_r_1)
    output_r_2 = output_r[:,:,:,1:2]
    input_high_r_2 = input_high_r[:,:,:,1:2]
    ssim_r_2 = tf_ssim(output_r_2, input_high_r_2)
    output_r_3 = output_r[:,:,:,2:3]
    input_high_r_3 = input_high_r[:,:,:,2:3]
    ssim_r_3 = tf_ssim(output_r_3, input_high_r_3)
    ssim_r = (ssim_r_1 + ssim_r_2 + ssim_r_3)/3.0
    loss_ssim1 = 1-ssim_r
    return loss_ssim1

loss_square = tf.reduce_mean(tf.square(output_r  - input_high_r))
loss_ssim = ssim_loss(output_r, input_high_r)
loss_grad = grad_loss(output_r, input_high_r)

loss_restoration = loss_square + loss_grad + loss_ssim

### initialize
lr = tf.placeholder(tf.float32, name='learning_rate')
global_step = tf.get_variable('global_step', [], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptimizer')
with tf.control_dependencies(update_ops):
    grads = optimizer.compute_gradients(loss_restoration)
    train_op_restoration = optimizer.apply_gradients(grads, global_step=global_step)

var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
var_restoration = [var for var in tf.trainable_variables() if 'Restoration_net' in var.name]

saver_restoration = tf.train.Saver(var_list=var_restoration)
saver_Decom = tf.train.Saver(var_list = var_Decom)
sess.run(tf.global_variables_initializer())
print("[*] Initialize model successfully...")

### load data
### Based on the decomposition net, we first get the decomposed reflectance maps 
### and illumination maps, then train the restoration net.
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

eval_low_data = []
eval_low_data_names = glob('./LOLdataset/eval15/low/*.png') 
eval_low_data_names.sort()
for idx in range(len(eval_low_data_names)):
    eval_low_im = load_images(eval_low_data_names[idx])
    eval_low_data.append(eval_low_im)

pre_decom_checkpoint_dir = './checkpoint/decom_net_train/'
ckpt_pre=tf.train.get_checkpoint_state(pre_decom_checkpoint_dir)
if ckpt_pre:
    print('loaded '+ckpt_pre.model_checkpoint_path)
    saver_Decom.restore(sess,ckpt_pre.model_checkpoint_path)
else:
    print('No pre_decom_net checkpoint!')

decomposed_low_r_data_480 = []
decomposed_low_i_data_480 = []
decomposed_high_r_data_480 = []
for idx in range(len(train_low_data)):
    input_low = np.expand_dims(train_low_data[idx], axis=0)
    RR, II = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_low})
    RR0 = np.squeeze(RR)
    II0 = np.squeeze(II)
    print(idx, RR0.shape, II0.shape)
    decomposed_low_r_data_480.append(RR0)
    decomposed_low_i_data_480.append(II0)
for idx in range(len(train_high_data)):
    input_high = np.expand_dims(train_high_data[idx], axis=0)
    RR2, II2 = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_high})
    ### To improve the constrast, we slightly change the decom_r_high by using decom_r_high**1.2
    RR02 = np.squeeze(RR2**1.2)
    print(idx, RR02.shape)
    decomposed_high_r_data_480.append(RR02)

decomposed_eval_low_r_data = []
decomposed_eval_low_i_data = []
for idx in range(len(eval_low_data)):
    input_eval = np.expand_dims(eval_low_data[idx], axis=0)
    RR3, II3 = sess.run([decom_output_R, decom_output_I], feed_dict={input_decom: input_eval})
    RR03 = np.squeeze(RR3)
    II03 = np.squeeze(II3)
    print(idx, RR03.shape, II03.shape)
    decomposed_eval_low_r_data.append(RR03)
    decomposed_eval_low_i_data.append(II03)


eval_restoration_low_r_data = decomposed_low_r_data_480[467:480] + decomposed_eval_low_r_data[0:15]
eval_restoration_low_i_data = decomposed_low_i_data_480[467:480] + decomposed_eval_low_i_data[0:15]

train_restoration_low_r_data = decomposed_low_r_data_480[0:466]
train_restoration_low_i_data = decomposed_low_i_data_480[0:466]
train_restoration_high_r_data = decomposed_high_r_data_480[0:466]
#train_restoration_high_i_data = train_restoration_high_i_data_480[0:466]
print(len(train_restoration_high_r_data), len(train_restoration_low_r_data),len(train_restoration_low_i_data))
print(len(eval_restoration_low_r_data),len(eval_restoration_low_i_data))
assert len(train_restoration_high_r_data) == len(train_restoration_low_r_data)
assert len(train_restoration_low_i_data) == len(train_restoration_low_r_data)
print('[*] Number of training data: %d' % len(train_restoration_high_r_data))

learning_rate = 0.0001
def lr_schedule(epoch):
    initial_lr = learning_rate
    if epoch<=800:
        lr = initial_lr
    elif epoch<=1250:
        lr = initial_lr/2
    elif epoch<=1500:
        lr = initial_lr/4 
    else:
        lr = initial_lr/10 
    return lr

epoch = 1000

sample_dir = './Restoration_net_train/'
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)

eval_every_epoch = 50
train_phase = 'Restoration'
numBatch = len(train_restoration_low_r_data) // int(batch_size)
train_op = train_op_restoration
train_loss = loss_restoration
saver = saver_restoration

checkpoint_dir = './checkpoint/Restoration_net_train/'
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver_restoration.restore(sess,ckpt.model_checkpoint_path)
else:
    print('No pre_restoration_net checkpoint!')

start_step = 0
start_epoch = 0
iter_num = 0
print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (train_phase, start_epoch, iter_num))
start_time = time.time()
image_id = 0

for epoch in range(start_epoch, epoch):
    for batch_id in range(start_step, numBatch):
        batch_input_low_r = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
        batch_input_low_i = np.zeros((batch_size, patch_size, patch_size, 1), dtype="float32")

        batch_input_high_r = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")

        for patch_id in range(batch_size):
            h, w, _ = train_restoration_low_r_data[image_id].shape
            x = random.randint(0, h - patch_size)
            y = random.randint(0, w - patch_size)
            i_low_expand = np.expand_dims(train_restoration_low_i_data[image_id], axis = 2)
            rand_mode = random.randint(0, 7)
            batch_input_low_r[patch_id, :, :, :] = data_augmentation(train_restoration_low_r_data[image_id][x : x+patch_size, y : y+patch_size, :] , rand_mode)#+ np.random.normal(0, 0.1, (patch_size,patch_size,3))  , rand_mode)
            batch_input_low_i[patch_id, :, :, :] = data_augmentation(i_low_expand[x : x+patch_size, y : y+patch_size, :] , rand_mode)#+ np.random.normal(0, 0.1, (patch_size,patch_size,3))  , rand_mode)

            batch_input_high_r[patch_id, :, :, :] = data_augmentation(train_restoration_high_r_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)

            image_id = (image_id + 1) % len(train_restoration_low_r_data)
            if image_id == 0:
                tmp = list(zip(train_restoration_low_r_data, train_restoration_low_i_data, train_restoration_high_r_data))
                random.shuffle(tmp)
                train_restoration_low_r_data, train_restoration_low_i_data, train_restoration_high_r_data = zip(*tmp)

        _, loss = sess.run([train_op, train_loss], feed_dict={input_low_r: batch_input_low_r,input_low_i: batch_input_low_i,\
                                                              input_high_r: batch_input_high_r, lr: lr_schedule(epoch)})
        print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
              % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
        iter_num += 1
    if (epoch + 1) % eval_every_epoch == 0:
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch + 1))
        for idx in range(len(eval_restoration_low_r_data)):
            input_uu_r = eval_restoration_low_r_data[idx] 
            input_low_eval_r = np.expand_dims(input_uu_r, axis=0)
            input_uu_i = eval_restoration_low_i_data[idx] 
            input_low_eval_i = np.expand_dims(input_uu_i, axis=0)
            input_low_eval_ii = np.expand_dims(input_low_eval_i, axis=3)
            result_1 = sess.run(output_r, feed_dict={input_low_r: input_low_eval_r, input_low_i: input_low_eval_ii})

            save_images(os.path.join(sample_dir, 'eval_%d_%d.png' % ( idx + 1, epoch + 1)), input_uu_r, result_1)
        saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=epoch)

print("[*] Finish training for phase %s." % train_phase)



