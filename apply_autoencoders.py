from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

import os, random

from PIL import Image

import functools
import itertools

import collections
import six

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.DEBUG)

"""
Inference script for the various STEM and TEM autoencoders
TEM and STEM autoencoders are trained for middle-convolution feature space sizes of 1, 4 and 16
Miscellaneous functions for parsing data are also included in this directory
The class-based autoencoder interface supports inference from 160x160 crops or can be
applied to an entire image. Functionality for whole-image applications is included in the class
"""

data_dir = "G:/unaltered_TEM_crops-171x171/"
#data_dir = "G:/unaltered_STEM_crops-171x171/"

modelSavePeriod = 2 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "G:/noise-removal-kernels-TEM/autoencoder/16/"
#model_dir = "G:/noise-removal-kernels-STEM/depth1/"

shuffle_buffer_size = 5000
num_parallel_calls = 4
num_parallel_readers = 4
prefetch_buffer_size = 5
batch_size = 32
num_gpus = 1

#batch_size = 8 #Batch size to use during training
num_epochs = 1 #Dataset repeats indefinitely

logDir = "C:/dump/train/"
log_file = model_dir+"log.txt"
val_log_file = model_dir+"val_log.txt"
variables_file = model_dir+"variables.txt"
log_every = 1 #Log every _ examples

channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 160
generator_input_size = cropsize
height_crop = width_crop = cropsize

#hparams = experiment_hparams(train_batch_size=batch_size, eval_batch_size=16)

num_workers = 1

increase_batch_size_by_factor = 1
effective_batch_size = increase_batch_size_by_factor*batch_size

val_skip_n = 10

save_result_every_n_batches = 50000

def architecture(input, encoding_features):

    phase = True

    def _instance_norm(net):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        shift = tf.Variable(tf.zeros(var_shape), trainable=False)
        scale = tf.Variable(tf.ones(var_shape), trainable=False)
        epsilon = 1e-3
        normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
        return scale * normalized + shift

    def _batch_norm_fn(input, train=True):
        batch_norm = tf.contrib.layers.batch_norm(
            input,
            decay=0.999,
            epsilon=0.001,
            center=True, 
            scale=True,
            is_training=True,
            fused=True,
            zero_debias_moving_mean=False,
            renorm=False)
        return batch_norm

    def batch_then_activ(input):
        _batch_then_activ = _batch_norm_fn(input)
        _batch_then_activ = tf.nn.relu(_batch_then_activ)
        return _batch_then_activ

    def strided_conv_block(input, filters, stride, rate=1, phase=phase, 
                           extra_batch_norm=True):
        
        strided_conv = slim.separable_convolution2d(
            inputs=input,
            num_outputs=filters,
            kernel_size=3,
            depth_multiplier=1,
            stride=stride,
            padding='SAME',
            data_format='NHWC',
            rate=rate,
            activation_fn=None,#tf.nn.relu,
            normalizer_fn=_batch_norm_fn if extra_batch_norm else None,
            normalizer_params=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=None,
            biases_initializer=tf.zeros_initializer(),
            biases_regularizer=None,
            reuse=None,
            variables_collections=None,
            outputs_collections=None,
            trainable=True)
        strided_conv = batch_then_activ(strided_conv)

        return strided_conv

    def deconv_block(input, filters, batch_norm_on=True):
        '''Transpositionally convolute a feature space to upsample it'''

        deconv = slim.conv2d_transpose(
            inputs=input,
            num_outputs=filters,
            kernel_size=3,
            stride=2,
            padding='SAME',
            activation_fn=None)
        if batch_norm_on:
            deconv = batch_then_activ(deconv)
        else:
            deconv = tf.nn.relu(deconv)

        return deconv

    a = strided_conv_block(input, 64, 2)
    a = strided_conv_block(a, 128, 2)
    a = strided_conv_block(a, 256, 2)

    a = strided_conv_block(a, encoding_features, 1)

    a = deconv_block(a, 256)
    a = deconv_block(a, 128)
    a = deconv_block(a, 64, batch_norm_on=False)

    a = slim.conv2d(
                inputs=a,
                num_outputs=1,
                kernel_size=3,
                padding="SAME",
                activation_fn=None,
                weights_initializer=None,
                biases_initializer=None)
    #a = _instance_norm(a)

    return a


def experiment(img, encoding_features):

    imgs = tf.reshape(img, [-1, cropsize, cropsize, 1])

    outputs = architecture(img, encoding_features)

    return outputs

def flip_rotate(img):
    """Applies a random flip || rotation to the image, possibly leaving it unchanged"""

    choice = np.random.randint(0, 8)
    
    if choice == 0:
        return img
    if choice == 1:
        return np.rot90(img, 1)
    if choice == 2:
        return np.rot90(img, 2)
    if choice == 3:
        return np.rot90(img, 3)
    if choice == 4:
        return np.flip(img, 0)
    if choice == 5:
        return np.flip(img, 1)
    if choice == 6:
        return np.flip(np.rot90(img, 1), 0)
    if choice == 7:
        return np.flip(np.rot90(img, 1), 1)

def load_image(addr, resize_size=None, img_type=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""

    try:
        img = imread(addr, mode='F')

        x = np.random.randint(0, img.shape[0]-cropsize)
        y = np.random.randint(0, img.shape[1]-cropsize)

        img = img[x:(x+cropsize),y:(y+cropsize)]
    except:
        img = 0.5*np.ones((cropsize,cropsize))
        print("Image read failed")

    return img.astype(img_type)

def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def norm_img(img):
    
    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.)
    else:
        a = 0.5*(min+max)
        b = 0.5*(max-min)

        img = (img-a) / b

    return img.astype(np.float32)

def record_parser(record):

    img = preprocess(flip_rotate(load_image(record)))

    if np.sum(np.isfinite(img)) != cropsize**2:
        img = np.ones((cropsize, cropsize), dtype=np.float32)

    return img

def disp(img):
    d = int(round(np.sqrt(img.size)))
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img.reshape((d,d))))
    cv2.waitKey(0)
    return

class RunConfig(tf.contrib.learn.RunConfig): 
    def uid(self, whitelist=None):
        """
        Generates a 'Unique Identifier' based on all internal fields.
        Caller should use the uid string to check `RunConfig` instance integrity
        in one session use, but should not rely on the implementation details, which
        is subject to change.
        Args:
          whitelist: A list of the string names of the properties uid should not
            include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
            includes most properties user allowes to change.
        Returns:
          A uid string.
        """
        if whitelist is None:
            whitelist = run_config._DEFAULT_UID_WHITE_LIST

        state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        # Pop out the keys in whitelist.
        for k in whitelist:
            state.pop('_' + k, None)

        ordered_state = collections.OrderedDict(
            sorted(state.items(), key=lambda t: t[0]))
        # For class instance without __repr__, some special cares are required.
        # Otherwise, the object address will be used.
        if '_cluster_spec' in ordered_state:
            ordered_state['_cluster_spec'] = collections.OrderedDict(
                sorted(ordered_state['_cluster_spec'].as_dict().items(), key=lambda t: t[0]))
        return ', '.join(
            '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state))

class Micrograph_Autoencoder(object):
    '''Class to access all the TEM and STEM autoencoders'''

    def __init__(self, 
                 checkpoint_loc="//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-13/model",
                 visible_cuda=None,
                 encoding_features=16):

        if visible_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda

        # Session configuration.
        log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device_placement,
            intra_op_parallelism_threads=1,
            gpu_options=tf.GPUOptions(force_gpu_compatible=True))

        img_ph = list([tf.placeholder(tf.float32, shape=(cropsize, cropsize, 1), name='img')])

        outputs = experiment(img_ph, encoding_features)

        sess =  tf.Session(config=sess_config)
        sess.run(tf.initialize_variables(tf.all_variables()))

        #print(tf.all_variables())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_loc))

        self.sess = sess
        self.inputs = img_ph
        self.outputs = outputs

    def preprocess(self, img, pad_width=0):

        img[np.isnan(img)] = 0.
        img[np.isinf(img)] = 0.

        img = scale0to1(img)

        img /= np.mean(img)

        img = np.pad(img, pad_width=pad_width, mode='reflect').reshape(
            img.shape[0]+2*pad_width,img.shape[1]+2*pad_width,1)

        return img.astype(np.float32)

    def denoise_crop(self, crop, preprocess=True, scaling=True, postprocess=True):

        if scaling:
            offset = np.min(crop)
            scale = np.mean(crop) - offset
            
            if scale:
                crop = (crop-offset) / scale
            else:
                crop.fill(1.)

        #print(scale, offset)

        pred = self.sess.run(self.outputs, 
                             feed_dict={self.inputs[0]: 
                                        self.preprocess(crop) if preprocess else crop})

        if scaling:
            pred = scale*pred+offset if scale else pred*offset/np.mean(pred)

        if postprocess:
            pred = pred.reshape((cropsize, cropsize))

        return pred

    def denoise(self, img, preprocess=True, postprocess=True, overlap=25, used_overlap=1):
        """
        img: Image to denoise
        preprocess: Remove nans and infs, make sure it has a (-1, 512, 512, 1) shape and 
        reflection pad it
        postprocess: Clip output values to [0.0, 1.0] and reshape to (512, 512)
        overlap: Amount of crop overlap. Predictions for the overlapping region won't be used
        """

        if overlap < used_overlap:
            overlap = used_overlap

        denoised = np.zeros((img.shape[0]+2*overlap, img.shape[1]+2*overlap))
        contributions = np.zeros((img.shape[0]+2*overlap, img.shape[1]+2*overlap))

        dims = img.shape

        if preprocess:
            img = self.preprocess(img, pad_width=overlap)

        len = cropsize-2*overlap
        len0 = len1 = len
        for x in range(0, img.shape[1], len):

            if img.shape[1] - x <= cropsize:
                x = img.shape[1] - cropsize

                for y in range(0, img.shape[2], len):

                    if img.shape[2] - y <= cropsize:
                        y = img.shape[2] - cropsize

                        crop = img[:, x:(x+cropsize), y:(y+cropsize), :]
                        offset = np.min(crop)
                        scale = 1. / (np.mean(crop) - offset)

                        if scale:
                            crop = (crop-offset)/scale
                        else: 
                            crop.fill(0.5)

                        pred = self.denoise_crop(
                            crop=crop, 
                            preprocess=False,
                            scaling=False, 
                            postprocess=False).reshape((cropsize,cropsize))
                        
                        #Rescale prediction so it has the same mean as the original crop
                        pred *= np.mean(crop)/np.mean(pred)

                        pred = pred*scale+offset if scale else pred*offset/np.mean(pred)
                        pred = pred.reshape(cropsize, cropsize)
                
                        denoised[(x+overlap-used_overlap):(x+cropsize-(overlap-used_overlap)),
                                 (y+overlap-used_overlap):(y+cropsize-(overlap-used_overlap))] += pred[
                                     (overlap-used_overlap):(cropsize-(overlap-used_overlap)),
                                     (overlap-used_overlap):(cropsize-(overlap-used_overlap))]

                        contributions[(x+overlap-used_overlap):(x+cropsize-(overlap-used_overlap)),
                                      (y+overlap-used_overlap):(y+cropsize-(overlap-used_overlap))] += 1

                    crop = img[:, x:(x+cropsize), y:(y+cropsize), :]
                    offset = np.min(crop)
                    scale = 1. / (np.mean(crop) - offset)

                    if scale:
                        crop = (crop-offset)/scale
                    else: 
                        crop.fill(0.5)

                    pred = self.denoise_crop(
                        crop=crop, 
                        preprocess=False,
                        scaling=False, 
                        postprocess=False).reshape((cropsize,cropsize))

                    pred = pred*scale+offset if scale else pred*offset/np.mean(pred)
                    pred = pred.reshape(cropsize, cropsize)
                
                    denoised[(x+overlap-used_overlap):(x+cropsize-(overlap-used_overlap)),
                                (y+overlap-used_overlap):(y+cropsize-(overlap-used_overlap))] += pred[
                                    (overlap-used_overlap):(cropsize-(overlap-used_overlap)),
                                    (overlap-used_overlap):(cropsize-(overlap-used_overlap))]

                    contributions[(x+overlap-used_overlap):(x+cropsize-(overlap-used_overlap)),
                                    (y+overlap-used_overlap):(y+cropsize-(overlap-used_overlap))] += 1

            for y in range(0, img.shape[2], len):

                if img.shape[2] - y <= cropsize:
                    y = img.shape[2] - cropsize

                    crop = img[:, x:(x+cropsize), y:(y+cropsize), :]
                    offset = np.min(crop)
                    scale = 1. / (np.mean(crop) - offset)

                    if scale:
                        crop = (crop-offset)/scale
                    else: 
                        crop.fill(0.5)

                    pred = self.denoise_crop(
                        crop=crop, 
                        preprocess=False,
                        scaling=False, 
                        postprocess=False).reshape((cropsize,cropsize))

                    pred = pred*scale+offset if scale else pred*offset/np.mean(pred)
                    pred = pred.reshape(cropsize, cropsize)
                
                    denoised[(x+overlap-used_overlap):(x+cropsize-(overlap-used_overlap)),
                             (y+overlap-used_overlap):(y+cropsize-(overlap-used_overlap))] += pred[
                                 (overlap-used_overlap):(cropsize-(overlap-used_overlap)),
                                 (overlap-used_overlap):(cropsize-(overlap-used_overlap))]

                    contributions[(x+overlap-used_overlap):(x+cropsize-(overlap-used_overlap)),
                                  (y+overlap-used_overlap):(y+cropsize-(overlap-used_overlap))] += 1

                crop = img[:, x:(x+cropsize), y:(y+cropsize), :]
                offset = np.min(crop)
                scale = 1. / (np.mean(crop) - offset)

                if scale:
                    crop = (crop-offset)/scale
                else: 
                    crop.fill(0.5)

                pred = self.denoise_crop(
                    crop=crop, 
                    preprocess=False,
                    scaling=False, 
                    postprocess=False).reshape((cropsize,cropsize))

                pred = pred*scale+offset if scale else pred*offset/np.mean(pred)
                pred = pred.reshape(cropsize, cropsize)
                
                denoised[(x+overlap-used_overlap):(x+cropsize-(overlap-used_overlap)),
                         (y+overlap-used_overlap):(y+cropsize-(overlap-used_overlap))] += pred[
                             (overlap-used_overlap):(cropsize-(overlap-used_overlap)),
                             (overlap-used_overlap):(cropsize-(overlap-used_overlap))]

                contributions[(x+overlap-used_overlap):(x+cropsize-(overlap-used_overlap)),
                              (y+overlap-used_overlap):(y+cropsize-(overlap-used_overlap))] += 1

        denoised = denoised[overlap:(denoised.shape[0]-overlap),
                            overlap:(denoised.shape[1]-overlap)] / contributions[
                                overlap:(contributions.shape[0]-overlap),
                                overlap:(contributions.shape[1]-overlap)]

        return denoised

if __name__ == '__main__':

    loc = r'G:\noise-removal-kernels-STEM\autoencoder\16\input-45000.tif'
    img = imread(loc, mode='F')
    img = 233*img[:160, :160]

    ckpt_loc = 'G:/noise-removal-kernels-TEM/autoencoder/16/model/'
    nn = Micrograph_Autoencoder(checkpoint_loc=ckpt_loc,
                                visible_cuda='1',
                                encoding_features=16)

    nn_img = nn.denoise_crop(img)
    disp(img)
    disp(nn_img)

    print(img)
    print(nn_img)