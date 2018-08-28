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
batch_size = 1
num_gpus = 1

#batch_size = 8 #Batch size to use during training
num_epochs = 1 #Dataset repeats indefinitely

logDir = "C:/dump/train/"
log_file = model_dir+"log.txt"
val_log_file = model_dir+"val_log.txt"
variables_file = model_dir+"variables.txt"
log_every = 1 #Log every _ examples

channels = 1 #Greyscale input image

#hparams = experiment_hparams(train_batch_size=batch_size, eval_batch_size=16)

num_workers = 1

increase_batch_size_by_factor = 1
effective_batch_size = increase_batch_size_by_factor*batch_size

val_skip_n = 10

save_result_every_n_batches = 50000

def architectures(inputs, inputs_truth, depths, widths):

    cropsize = widths[0]

    def pad(tensor, size):
        d1_pad = size[0]
        d2_pad = size[1]

        paddings = tf.constant([[0, 0], [d1_pad, d1_pad], [d2_pad, d2_pad], [0, 0]], dtype=tf.int32)
        padded = tf.pad(tensor, paddings, mode="REFLECT")
        return padded

    def make_layer(size, type):
        
        if type == 'biases':
            init = np.array([0.], dtype=np.float32)
        if type == 'weights':
            init = np.array([1./(size*size)], dtype=np.float32)

        #print("Init: {}".format(init))

        printij = False #Set to true to debug

        if printij:
            print("\nStart:")

        variables = [[None for _ in range(size)] for _ in range(size)]

        if printij:
            for i in range(3):
                for j in range(3):
                    if variables[i][j]:
                        print(i, j, variables[i][j].name)
                    else:
                        print(i,j)
            print("\n")

        #for i in range(size):
        #    for j in range(size):
        #        variables[i][j] = tf.get_variable('i-{}_j-{}'.format(i,j), dtype=tf.float32, initializer=init, trainable=True)

        offset = size//2
        for x in range(size//2+1):
            for y in range(x+1):

                with tf.variable_scope("var_x-{}_y-{}".format(x, y), reuse=False) as scope:

                    i, j = offset+x, offset+y
                    variables[i][j] = tf.get_variable('v', dtype=tf.float32, initializer=init, trainable=True)
                    if printij:
                        print(i,j,x,y,variables[i][j].name)

                        for i in range(3):
                            for j in range(3):
                                if variables[i][j]:
                                    print(i, j, variables[i][j].name)
                                else:
                                    print(i,j)
                        print("\n")

                    if x > 0:
                        if y == 0:
                            i, j = offset-x, offset
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset, offset+x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset, offset-x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                        elif y == x:
                            i, j = offset+x, offset-y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-x, offset+y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-x, offset-y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                        elif y != x:
                            i, j = offset-x, offset+y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset+x, offset-y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-x, offset-y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset+y, offset+x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-y, offset+x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset+y, offset-x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-y, offset-x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                #print(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="depth-1_size-3/var_x-{}_y-{}".format(x, y))))

        #print(variables)

        if printij:
            for i in range(3):
                for j in range(3):
                    print(i, j, variables[i][j].name)

        concats = []
        for i in range(size):
            concats.append(tf.concat(variables[i][:], axis=0))
        kernel = tf.stack(concats, axis=1)

        kernel = tf.expand_dims(kernel, axis=0)
        kernel = tf.expand_dims(kernel, axis=3)

        #kernel = tf.reshape(kernel, [-1, size, size, 1])

        #print(kernel)

        return kernel

    #depths = [1]
    #widths = [3]
    #depths = [i for i in range(1, 6)]
    #widths = [3, 5, 7, 9, 13, 17]

    filters = []
    filter_scopes = []
    filter_depths = []
    filter_widths = []
    outputs = []
    losses = []
    ps = []
    for depth in depths:
        print("Depth: {}".format(depth))
        for width in widths:
            print("Width: {}".format(width))

            default_scope = "depth-{}_size-{}".format(depth, width)

            #Filter creation
            def filter_fn(input):
                    with tf.variable_scope('w0'):
                        filter = make_layer(width, 'weights')*input

                    for i in range(1, depth):
                        with tf.variable_scope('b'+str(i)):
                            filter += make_layer(width, 'biases')

                        filter = tf.sigmoid(filter)
                        filter = tf.contrib.layers.fully_connected(
                            inputs=filter, 
                            num_outputs=1,
                            activation_fn=None,
                            weights_initializer=None,
                            biases_initializer=None)
                    
                        with tf.variable_scope('w'+str(i)):
                            filter = make_layer(width, 'weights')*filter

                    output = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(filter, axis=1), axis=1), axis=1)

                    return output

            filters.append(filter_fn)
            filter_scopes.append(default_scope)
            filter_depths.append(depth)
            filter_widths.append(width)

            padded = pad(inputs, (0, 0))

            #Generate outputs
            output = [[None for _ in range(cropsize-width+1)] for _ in range(cropsize-width+1)]
            for x in range(cropsize-width+1):
                for y in range(cropsize-width+1):
                    if not x+y:
                        with tf.variable_scope(default_scope) as filter_scope:
                            _x = x+width
                            _y = y+width
                            output[x][y] = filter_fn(padded[:, x:_x, y:_y, :])
                    else:
                        with tf.variable_scope(filter_scope, reuse=True) as filter_scope:
                            _x = x+width
                            _y = y+width
                            output[x][y] = filter_fn(padded[:, x:_x, y:_y, :])

                    #print(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="depth-1_size-3")))

            concats = []
            for i in range(cropsize-width+1):
                concats.append(tf.stack(output[i][:], axis=1))
            output = tf.stack(concats, axis=2)
            output = tf.expand_dims(output, 3)

            padded_truth = pad(inputs_truth, (0, 0))
            p = padded_truth[:, (width//2):(cropsize-width//2), (width//2):(cropsize-width//2), :]
            loss = tf.losses.mean_squared_error(output, p)#tf.reduce_mean(tf.abs(output-p))
            loss = tf.cond(loss > 1., lambda: tf.sqrt(loss), lambda: loss)

            outputs.append(output)
            losses.append(loss)

    return filters, filter_scopes, filter_depths, filter_widths, outputs, losses, p, padded_truth


def experiment(img, img_truth, depths, widths):

    filters, filter_scopes, filter_depths, filter_widths, \
        outputs, losses, truth, padded_truth = architectures(img, img_truth, depths, widths)

    return {'filters': filters, 'filter_scopes': filter_scopes, 'filter_depths': filter_depths,
            'filter_widths': filter_widths, 'outputs': outputs,
            'losses': losses, 'truth': [truth], 'padded_truth': [padded_truth]}

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

def record_parser(record, cropsize):

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

class Kernels_and_MLPS(object):
    '''Class to access all the TEM and STEM autoencoders'''

    def __init__(self, 
                 ckpt_loc="G:/noise-removal-kernels-TEM/results/1/model/",
                 visible_cuda=None,
                 depth=1,
                 width=3):

        depths = [depth]
        widths = [width]
        cropsize = width
        self.cropsize = cropsize

        if visible_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda

        # Session configuration.
        log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device_placement,
            intra_op_parallelism_threads=1,
            gpu_options=tf.GPUOptions(force_gpu_compatible=True))

        img_ph = [tf.placeholder(tf.float32, shape=(cropsize, cropsize, 1), name='img')]
        img_truth_ph = [tf.placeholder(tf.float32, shape=(cropsize, cropsize, 1), name='img_truth')]

        exp_dict = experiment(img_ph, img_truth_ph, depths, widths)
        outputs = exp_dict['outputs']

        sess =  tf.Session(config=sess_config)
        sess.run(tf.initialize_variables(tf.all_variables()))

        #print(tf.all_variables())
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_loc))

        self.sess = sess
        self.inputs = img_ph
        self.outputs = outputs

    def preprocess(self, img, pad_width=0):

        img[np.isnan(img)] = 0.
        img[np.isinf(img)] = 0.

        #img = scale0to1(img)
        #img /= np.mean(img)

        img = np.pad(img, pad_width=pad_width, mode='reflect').reshape(
            img.shape[0]+2*pad_width,img.shape[1]+2*pad_width,1)

        return img.astype(np.float32)

    def denoise_crop(self, crop):
        """
        It is assumed that the crop is correctly scaled using the entire image
        """

        cropsize = self.cropsize

        pred = self.sess.run(self.outputs, 
                             feed_dict={self.inputs[0]: 
                                        self.preprocess(crop)})

        return pred[0]

    def denoise(self, img, preprocess=True, postprocess=True, used_overlap=1):
        """
        img: Image to denoise
        preprocess: Remove nans and infs, make sure it has a (-1, 512, 512, 1) shape and 
        reflection pad it
        postprocess: Clip output values to [0.0, 1.0] and reshape to (512, 512)
        overlap: Amount of crop overlap. Predictions for the overlapping region won't be used
        """

        cropsize = self.cropsize
        overlap = used_overlap = 0

        if overlap < used_overlap:
            overlap = used_overlap

        denoised = np.zeros((img.shape[0]+2*overlap, img.shape[1]+2*overlap))
        contributions = np.zeros((img.shape[0]+2*overlap, img.shape[1]+2*overlap))

        if preprocess:
            img = self.preprocess(img, pad_width=cropsize//2)

            offset = np.min(img)
            if np.max(img) == offset:
                img.fill(1.)
            else:
                scale = (np.mean(img) - offset)
                img = (img-offset) / scale

        l_ax1 = img.shape[0]
        l_ax2 = img.shape[1]

        len = 1
        len0 = len1 = len
        for x in range(0, l_ax1, len):

            if l_ax1 - x <= cropsize:
                x = l_ax1 - cropsize

                for y in range(0, l_ax2, len):

                    if l_ax2 - y <= cropsize:
                        y = l_ax2 - cropsize

                        crop = img[x:(x+cropsize), y:(y+cropsize), :]

                        denoised[x, y] = self.denoise_crop(crop=crop)

                    crop = img[x:(x+cropsize), y:(y+cropsize), :]

                    denoised[x, y] = self.denoise_crop(crop=crop)

            for y in range(0, l_ax2, len):

                if l_ax2 - y <= cropsize:
                    y = l_ax2 - cropsize

                    crop = img[x:(x+cropsize), y:(y+cropsize), :]

                    denoised[x, y] = self.denoise_crop(crop=crop)

                crop = img[x:(x+cropsize), y:(y+cropsize), :]

                denoised[x, y] = self.denoise_crop(crop=crop)

        if postprocess:
            denoised = denoised*scale+offset if scale else denoised*offset/np.mean(pred)

        return denoised

if __name__ == '__main__':

    depth = 3
    width = 7
    num = 3
    locs = ['G:/noise-removal-kernels-TEM+STEM/examples/orig/'+str(i)+'.tif' for i in range(1, 6)]
    dst = 'G:/noise-removal-kernels-TEM+STEM/examples/kernels+mlps/'

    ckpt_loc = 'G:/noise-removal-kernels-TEM+STEM/results/'+str(num)+'/model/'
    nn = Kernels_and_MLPS(ckpt_loc=ckpt_loc,
                            visible_cuda='1',
                            depth=depth,
                            width=width)

    for i, loc in enumerate(locs, 1):
        img = imread(loc, mode='F')
        img = img[:160, :160]

        nn_img = nn.denoise(img)
        #Image.fromarray(nn_img).save( dst+'d'+str(depth)+'w'+str(width)+'/'+str(i)+'.tif' )
