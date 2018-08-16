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

data_dir1 = "G:/noise-removal-kernels-TEM+STEM/data/orig/"
data_dir2 = "G:/noise-removal-kernels-TEM+STEM/data/16/"
#data_dir = "E:/stills_hq-mini/"

modelSavePeriod = 4 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
#model_dir = "G:/noise-removal-kernels-TEM/depth1/"
model_dir = "G:/noise-removal-kernels-TEM+STEM/results/1/"

shuffle_buffer_size = 5000
num_parallel_calls = 4
num_parallel_readers = 4
prefetch_buffer_size = 5
batch_size = 1
num_gpus = 1

#batch_size = 8 #Batch size to use during training
num_epochs = 1000000 #Dataset repeats indefinitely

logDir = "C:/dump/train/"
log_file = model_dir+"log.txt"
val_log_file = model_dir+"val_log.txt"
variables_file = model_dir+"variables.txt"
log_every = 1 #Log every _ examples

numMeans = 64 // batch_size
scaleMean = 4 #Each means array index increment corresponds to this increase in the mean
numDynamicGrad = 1 #Number of gradients to calculate for each possible mean when dynamically updating training
lossSmoothingBoxcarSize = 5

channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 20
generator_input_size = cropsize
height_crop = width_crop = cropsize

#hparams = experiment_hparams(train_batch_size=batch_size, eval_batch_size=16)

num_workers = 1

increase_batch_size_by_factor = 1
effective_batch_size = increase_batch_size_by_factor*batch_size

val_skip_n = 10

depths = [1]
widths = [3, 5, 7, 11, 15]
num_filters = len(depths)*len(widths)

def architectures(inputs, inputs_truth):

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


def experiment(img, img_truth, learning_rate_ph):

    filters, filter_scopes, filter_depths, filter_widths, \
        outputs, losses, truth, padded_truth = architectures(img, img_truth)

    train_ops = []
    for i in range(len(losses)):
        optimizer = tf.train.AdamOptimizer(learning_rate_ph[0], beta1 = 0.5)
        train_op = optimizer.minimize(losses[i])
        train_ops.append(train_op)

    return {'filters': filters, 'filter_scopes': filter_scopes, 'filter_depths': filter_depths,
            'filter_widths': filter_widths, 'outputs': outputs, 'train_ops': train_ops,
            'losses': losses, 'truth': [truth], 'padded_truth': [padded_truth]}

def flip_rotate(img):
    """Applies a random flip || rotation to the image, possibly leaving it unchanged"""

    choice = 0#np.random.randint(0, 8)
    
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

        #x = np.random.randint(0, img.shape[0]-cropsize)
        #y = np.random.randint(0, img.shape[1]-cropsize)

        #img = img[x:(x+cropsize),y:(y+cropsize)]
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

def preprocess(img):

    img[np.isnan(img)] = 0.
    img[np.isinf(img)] = 0.

    img = scale0to1(img)

    img /= np.mean(img)

    return img.astype(np.float32)

def record_parser(record1, record2):

    #print(record1, record2)

    img1 = load_image(record1)
    img2 = load_image(record2)

    img1 = img1[:cropsize,:cropsize]
    img2 = img2[:cropsize,:cropsize]

    if (np.sum(np.isfinite(img1)) != cropsize**2) or (np.sum(np.isfinite(img2)) != cropsize**2):
        img1 = 0.5*np.ones((cropsize, cropsize), dtype=np.float32)
        img2 = 0.5*np.ones((cropsize, cropsize), dtype=np.float32)

    return img1, img2

def reshaper(img1, img2):
    img1 = tf.reshape(img1, [cropsize, cropsize, channels])
    img2 = tf.reshape(img2, [cropsize, cropsize, channels])
    return img1, img2


def input_fn(dir, subset, batch_size):
    """Create a dataset from a list of filenames and shard batches from it"""

    with tf.device('/cpu:0'):

        dataset1 = tf.data.Dataset.list_files(data_dir1+"*.tif", shuffle=False) #dir+subset+"/"+"*.tif"
        #dataset1 = dataset1.take(6076)
        dataset2 = tf.data.Dataset.list_files(data_dir2+"*.tif", shuffle=False) #dir+subset+"/"+"*.tif"
        #dataset2 = dataset2.take(6076)

        dataset = tf.data.Dataset.zip((dataset1, dataset2))

        #dataset1_1 = tf.data.Dataset.list_files(data_dir1+"*.tif", shuffle=False)
        #dataset1_2 = tf.data.Dataset.list_files(data_dir1+"*.tif", shuffle=False)
        #dataset2_1 = tf.data.Dataset.list_files(data_dir2+"*.tif", shuffle=False)
        #dataset2_2 = tf.data.Dataset.list_files(data_dir2+"*.tif", shuffle=False)

        #dataset1 = tf.data.Dataset.zip((dataset1_1, dataset1_2))
        #dataset2 = tf.data.Dataset.zip((dataset2_1, dataset2_2))

        #dataset = dataset1.concatenate(dataset2)

        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(
            lambda file1, file2: tf.py_func(record_parser, [file1, file2], [tf.float32, tf.float32]),
            num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

        iter = dataset.make_one_shot_iterator()
        img1_batch, img2_batch = iter.get_next()

        return img1_batch, img2_batch 

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
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

def main():

    print("Initializing")

    tf.reset_default_graph()

    temp = set(tf.all_variables())

    with open(log_file, 'a') as log:
        log.flush()

        with open(val_log_file, 'a') as val_log:
            val_log.flush()

            # The env variable is on deprecation path, default is set to off.
            #os.environ['TF_SYNC_ON_FINISH'] = '0'
            #os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
            with tf.control_dependencies(update_ops):

                # Session configuration.
                log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
                sess_config = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=log_device_placement,
                    intra_op_parallelism_threads=1,
                    gpu_options=tf.GPUOptions(force_gpu_compatible=True))

                config = RunConfig(session_config=sess_config, model_dir=model_dir)

                img, img_truth = input_fn(data_dir1, '', batch_size=batch_size)
                #img_truth = input_fn(data_dir1, '', batch_size=batch_size)
                #img_val = input_fn(data_dir, 'val', batch_size=batch_size)

                with tf.Session(config=sess_config) as sess:

                    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                    temp = set(tf.all_variables())

                    __img, __img_truth = sess.run([img, img_truth])

                    #disp(__img[0])
                    #disp(__img_truth[0])

                    img_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img')
                                for i in __img]
                    img_truth_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img_truth')
                                for i in __img_truth]

                    del __img, __img_truth

                    learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')

                    exp_dict = experiment(img_ph, img_truth_ph, learning_rate_ph)

                    #assign_ops = []
                    #for i in range(num_filters):
                    #    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, exp_dict['filter_scopes'][i])

                    #    sum = vars[0]+4.*tf.reduce_sum(vars[1:])
                    #    factor = 1./sum

                    #    assign_ops += [v.assign(v*factor) for v in vars]

                    #########################################################################################

                    sess.run( tf.initialize_variables(set(tf.all_variables()) - temp) )
                    train_writer = tf.summary.FileWriter( logDir, sess.graph )

                    saver = tf.train.Saver()
                    #saver.restore(sess, tf.train.latest_checkpoint(model_dir+"model/"))

                    #with open(variables_file, 'a') as variables:
                    #    variables.flush()

                    #    for i in range(num_filters):
                    #        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, exp_dict['filter_scopes'][i])
                    #        v = sess.run(vars)

                    #        variables.write(str(v))
                    #        variables.write("\r\n")
                    #quit()

                    counter = 0
                    save_counter = counter
                    counter_init = counter+1

                    print("Session started")

                    while counter < 10000:

                        counter += 1

                        lr = np.array([.01*(1.-counter/10001)])

                        base_dict = {learning_rate_ph: lr}

                        _img, _img_truth = sess.run([img, img_truth])

                        #disp(_img[0])
                        #disp(_img_truth[0])

                        feed_dict = base_dict.copy()
                        feed_dict.update({ph: img for ph, img in zip(img_ph, _img)})
                        feed_dict.update({ph: img.reshape((cropsize, cropsize)).T.reshape(
                            (cropsize, cropsize, 1)) for ph, img in zip(img_truth_ph, _img_truth)})

                        results = sess.run( exp_dict['train_ops']+exp_dict['losses'], feed_dict=feed_dict )

                        losses = results[num_filters:]

                        print("Iter: {}, Losses: {}".format(counter, losses))

                        try:
                            log.write("Iter: {}, {}".format(counter, losses))
                        except:
                            print("Write to discr pred file failed")

                        #if not counter % val_skip_n:

                        #    _img = sess.run(img_val)

                        #    feed_dict = base_dict.copy()
                        #    feed_dict.update({ph: img for ph, img in zip(img_ph, _img)})

                        #    losses = sess.run( exp_dict['losses'], feed_dict=feed_dict )

                        #    print("Iter: {}, Val losses: {}".format(counter, losses))

                        #    try:
                        #        val_log.write("Iter: {}, {}".format(counter, losses))
                        #    except:
                        #        print("Write to val log file failed")

                        if counter > 50:
                            #sess.run(assign_ops)

                            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, exp_dict['filter_scopes'][0])
                            vs = sess.run(vars)
                            print(vs)

                            #filtered_img, truth, padded_truth = sess.run([exp_dict['outputs'], 
                            #                                exp_dict['truth'],
                            #                                exp_dict['padded_truth']], 
                            #                               feed_dict=feed_dict)
                            #disp(_img[0])
                            ##disp(_img_truth[0])
                            #disp(filtered_img[0][0].reshape((cropsize-2,cropsize-2)).T)
                            #disp(truth[0][0].reshape((cropsize-2,cropsize-2)).T)
                            #disp(padded_truth[0][0].reshape((cropsize,cropsize)).T)

                        #    os.system("pause")

                        #Save the model
                        if not counter % 5000:
                            saver.save(sess, save_path=model_dir+"model/", global_step=counter)

                    #Save the model
                    saver.save(sess, save_path=model_dir+"model/", global_step=counter)

                    with open(variables_file, 'w') as variables:
                        variables.flush()
                        for i in range(num_filters):
                            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, exp_dict['filter_scopes'][i])
                            names = [v.name for v in vars]

                            print(names)
                            for n in names:
                                variables.write(n)
                            variables.write("\r\n")

                            v = sess.run(vars)
                            names = sess.run(names)

                            variables.write(str(v))
                            variables.write("\r\n")

    return 

if __name__ == '__main__':

    main()



