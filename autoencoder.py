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

#data_dir = "G:/unaltered_TEM_crops-171x171/"
data_dir = "G:/unaltered_STEM_crops-171x171/"

modelSavePeriod = 2 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
#model_dir = "G:/noise-removal-kernels-TEM/autoencoder/1/"
#model_dir = "G:/noise-removal-kernels-STEM/autoencoder/4/"
model_dir = "G:/noise-removal-kernels-TEM+STEM/autoencoder/32/"

shuffle_buffer_size = 5000
num_parallel_calls = 4
num_parallel_readers = 4
prefetch_buffer_size = 5
batch_size = 32
num_gpus = 1

#batch_size = 8 #Batch size to use during training
num_epochs = 1000000 #Dataset repeats indefinitely

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

save_result_every_n_batches = 15000

def architecture(input, bn_ph):

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
            is_training=bn_ph,
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

    a = strided_conv_block(a, 32, 1)

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

def adam_updates(params, cost_or_grads, lr=0.001, mom1=np.array([0.9], dtype=np.float32), 
                 mom2=np.array([0.999], dtype=np.float32), clip_norm=40):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    #grads = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1.e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)


def experiment(img, learning_rate_ph, bn_ph):

    imgs = tf.reshape(img, [-1, cropsize, cropsize, 1])

    outputs = architecture(img, bn_ph)

    loss = tf.losses.mean_squared_error(imgs, outputs)

    loss = tf.cond(loss > 1, lambda: tf.sqrt(loss), lambda: loss)

    #model_params = tf.trainable_variables()
    #loss += 5.e-6 * tf.add_n([tf.nn.l2_loss(v) for v in model_params])

    optimizer = tf.train.AdamOptimizer(learning_rate_ph[0])
    #optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 40.0)
    train_op = optimizer.minimize(loss)

    return {'outputs': [outputs], 'loss': [loss], 'train_op': [train_op]}

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

def preprocess(img):

    img[np.isnan(img)] = 0.
    img[np.isinf(img)] = 0.

    img = scale0to1(img)

    img /= np.mean(img)

    return img.astype(np.float32)

def record_parser(record):

    img = preprocess(flip_rotate(load_image(record)))

    if np.sum(np.isfinite(img)) != cropsize**2:
        img = np.ones((cropsize, cropsize), dtype=np.float32)

    return img

def reshaper(img):
    img = tf.reshape(img, [cropsize, cropsize, channels])
    return img


def input_fn(dir, subset, batch_size):
    """Create a dataset from a list of filenames and shard batches from it"""

    dir1 = "G:/unaltered_TEM_crops-171x171/"
    dir2 = "G:/unaltered_STEM_crops-171x171/"
    with tf.device('/cpu:0'):

        dataset1 = tf.data.Dataset.list_files(dir1+"*.tif") #dir+subset+"/"+"*.tif"
        dataset1 = dataset1.take(6077)
        dataset2 = tf.data.Dataset.list_files(dir2+"*.tif") #dir+subset+"/"+"*.tif"
        dataset2 = dataset2.take(6077)

        dataset = dataset1.concatenate(dataset2)

        #dataset = tf.data.Dataset.list_files(dir+"*.tif") #dir+subset+"/"+"*.tif"
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(
            lambda file: tf.py_func(record_parser, [file], [tf.float32]),
            num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

        iter = dataset.make_one_shot_iterator()
        img_batch = iter.get_next()

        return img_batch

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

                img = input_fn(data_dir, 'train', batch_size=batch_size)
                img_val = input_fn(data_dir, 'val', batch_size=batch_size)

                with tf.Session(config=sess_config) as sess:

                    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                    temp = set(tf.all_variables())

                    __img = sess.run(img)
                    img_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img')
                                for i in __img]
                    del __img

                    learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')
                    bn_ph = tf.placeholder(tf.bool, name='bn')

                    exp_dict = experiment(img_ph, learning_rate_ph, bn_ph)

                    #########################################################################################

                    sess.run( tf.initialize_variables(set(tf.all_variables()) - temp) )
                    train_writer = tf.summary.FileWriter( logDir, sess.graph )

                    #print(tf.all_variables())
                    saver = tf.train.Saver()
                    #saver.restore(sess, tf.train.latest_checkpoint(model_dir+"model/"))

                    print("Session started")

                    #with open(variables_file, 'w') as variables:
                    #    variables.flush()
                    #    for i in range(num_filters):
                    #        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, exp_dict['filter_scopes'][i])
                    #        names = [v.name for v in vars]

                    #        print(names)
                    #        for n in names:
                    #            variables.write(n)
                    #        variables.write("\r\n")

                    #        v = sess.run(vars)
                    #        names = sess.run(names)

                    #        variables.write(str(v))
                    #        variables.write("\r\n")
                    #quit()

                    counter = 0
                    counter_init = counter+1
                    save_counter = counter
                    max_count = 60000
                    div = 5000
                    while counter < max_count:

                        counter += 1

                        step = counter // div
                        max_step = max_count // div
                        lr = 0.01*np.array([(1.-step/(max_step))])**2

                        base_dict = {learning_rate_ph: lr,
                                     bn_ph: True}

                        _img = sess.run(img)

                        feed_dict = base_dict.copy()
                        feed_dict.update({ph: img for ph, img in zip(img_ph, _img)})

                        #Save outputs occasionally
                        if counter <= 1 or not counter % save_result_every_n_batches or (counter < 10000 and not counter % 2000) or counter == counter_init:

                            results = sess.run( exp_dict['train_op']+exp_dict['loss']+exp_dict['outputs'], feed_dict=feed_dict )

                            losses = results[1]

                            input_scaled = scale0to1(_img[0])
                            output_scaled = scale0to1(results[2][0])

                            try:
                                save_input_loc = model_dir+"input-"+str(counter)+".tif"
                                save_output_loc = model_dir+"output-"+str(counter)+".tif"
                                Image.fromarray(input_scaled.reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
                                Image.fromarray(output_scaled.reshape(cropsize, cropsize).astype(np.float32)).save( save_output_loc )
                            except:
                                print("Image save failed")

                        else:
                            _, losses = sess.run( exp_dict['train_op']+exp_dict['loss'], feed_dict=feed_dict )

                        print("Iter: {}, Losses: {}".format(counter, losses))
                        if not np.isfinite(losses):
                            counter = save_counter
                            saver.restore(sess, tf.train.latest_checkpoint(model_dir+"model/"))

                        try:
                            log.write("Iter: {}, {}".format(counter, losses))
                        except:
                            print("Write to discr pred file failed")

                        if False:#not counter % val_skip_n:

                            _img = sess.run(img_val)

                            feed_dict = base_dict.copy()
                            feed_dict.update({ph: img for ph, img in zip(img_ph, _img)})

                            losses = sess.run( exp_dict['loss'], feed_dict=feed_dict )

                            print("Iter: {}, Val losses: {}".format(counter, losses))

                            try:
                                val_log.write("Iter: {}, {}".format(counter, losses))
                            except:
                                print("Write to val log file failed")

                        #if counter > 0:
                        #    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, exp_dict['filter_scopes'][0])
                        #    v = sess.run(vars)
                        #    print(v)

                            #results = sess.run(exp_dict['outputs']+exp_dict['ps'], feed_dict=feed_dict)
                            #l = len(results)
                            #print(results[0])
                            #print(l)
                            #print(results[l//2])
                            ##disp(_img[0])
                            #disp(results[0][0].reshape((cropsize-2,cropsize-2)).T)
                            #disp(results[l//2][0].reshape((cropsize-2,cropsize-2)).T)

                            #os.system("pause")

                        #Save the model
                        if not counter % 5000:
                            saver.save(sess, save_path=model_dir+"model/", global_step=counter)
                            save_counter = counter

                    #Save the model
                    saver.save(sess, save_path=model_dir+"model/", global_step=counter)

    return 

if __name__ == '__main__':

    main()



