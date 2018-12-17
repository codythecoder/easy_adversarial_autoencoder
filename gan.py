"""
Loosely based on the MNIST GAN by aymericdamien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""


"""
TODO

Batch normalization
Incremental scale increase
NVIDIA style based generator

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
import argparse
import os
from PIL import Image
from math import ceil
import random
import sys
from src.images import load_images, save_image, get_reshaped_folder
from consts import *
from src.misc import Flip_Flopper
from collections import defaultdict

dense = tf.layers.dense
flatten = tf.contrib.layers.flatten
reshape = tf.reshape
conv2d = tf.layers.conv2d
upscale = tf.image.resize_images
BICUBIC = tf.image.ResizeMethod.BICUBIC



class Batcher:
    def __init__(self, iterable):
        self.data = list(iterable)

    def batch(self, count):
        count = min(len(self.data), count)
        return random.sample(self.data, count)

    def extend(self, data):
        self.data.extend(data)

    def append(self, data):
        self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __nonzero__(self):
        return bool(self.data)

    __bool__ = __nonzero__


def slide(start, end, steps):
    yield start
    for i in range(1, steps-1):
        yield start * (steps - 1 - i) / (steps - 1) + end * i / (steps - 1)
    yield end


assert target_shape[0] % (2**deconvolution_layers) == 0
assert target_shape[1] % (2**deconvolution_layers) == 0
image_dim = target_shape[0] * target_shape[1] * color_channels

start_time = datetime.now().strftime('%Y%m%d_%H%M')

parser = argparse.ArgumentParser()
parser.add_argument('--images', default='images')
parser.add_argument('--out', default='reshaped_images')
parser.add_argument('--no_load', action='store_true')
parser.add_argument('--no_save', action='store_true')
parser.add_argument('--show_transition', action='store_true')

args = parser.parse_args()

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape, dtype=tf.float32, partition_info=None):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.), dtype=tf.float32)

def noisy(net_in, std):
    noise = tf.random_normal(shape=tf.shape(net_in), mean=0.0, stddev=std, dtype=tf.float32)
    return net_in + noise

def conv2d_batch_norm(weights, *args, is_train=False, **kwargs):
    activation = None
    if 'activation' in kwargs:
        activation = kwargs['activation']
        kwargs['activation'] = None
    elif len(args) >= 8:
        args = list(args)
        activation = args[7]
        args[7] = None
    weights = conv2d(weights, *args, **kwargs)
    weights = tf.layers.batch_normalization(weights, training=is_train)
    if activation is not None:
        weights = activation(weights)
    return weights

# Generator
def encoder(net_in, reuse=False):
    start_shape = [
        target_shape[0] // 2**deconvolution_layers,
        target_shape[1] // 2**deconvolution_layers,
    ]
    layers = start_layers // 2 ** deconvolution_layers
    start_size = start_shape[0] * start_shape[1] * start_layers
    with tf.variable_scope("encoder", reuse=reuse):
        weights = conv2d_batch_norm(net_in, layers, (5, 5), activation=tf.nn.leaky_relu, padding='SAME')
        weights = conv2d_batch_norm(weights, layers, (3, 3), activation=tf.nn.leaky_relu, padding='SAME')
        for _ in range(deconvolution_layers):
            weights = conv2d_batch_norm(weights, layers, (5, 5), (2, 2), activation=tf.nn.leaky_relu, padding='SAME')
            weights = conv2d_batch_norm(weights, layers, (3, 3), activation=tf.nn.leaky_relu, padding='SAME')
            layers *= 2
        weights = flatten(weights)
        # weights = dense(weights, 3000, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, 1000, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, 700, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        # weights = dense(weights, 550, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, 400, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, 300, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, 200, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, latent_size, kernel_initializer=glorot_init, activation=tf.nn.tanh)
    return weights

# Generator
def decoder(net_in, reuse=False):
    start_shape = [
        target_shape[0] // 2**deconvolution_layers,
        target_shape[1] // 2**deconvolution_layers,
    ]
    start_size = start_shape[0] * start_shape[1] * start_layers
    with tf.variable_scope("decoder", reuse=reuse):
        weights = dense(net_in, 200, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, 300, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, 400, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        # weights = dense(weights, 550, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, 700, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, 1000, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        # weights = dense(weights, 3000, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = dense(weights, start_size, kernel_initializer=glorot_init, activation=tf.nn.tanh)
        weights = reshape(weights, [-1, *start_shape, start_layers])
        curr_shape = start_shape
        layers = start_layers
        for _ in range(deconvolution_layers):
            curr_shape = curr_shape[0] * 2, curr_shape[1] * 2
            weights = upscale(weights, curr_shape)
            weights = conv2d_batch_norm(weights, layers, (5, 5), activation=tf.nn.leaky_relu, padding='SAME')
            weights = conv2d_batch_norm(weights, layers, (3, 3), activation=tf.nn.leaky_relu, padding='SAME')
            layers /= 2

        weights = conv2d_batch_norm(weights, 3, (5, 5), activation=tf.nn.leaky_relu, padding='SAME')
        weights = conv2d_batch_norm(weights, 3, (3, 3), activation=tf.nn.sigmoid, padding='SAME')
    return weights



# Discriminator
def discriminator(net_in, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        weights = conv2d_batch_norm(net_in, 32, (4, 4), (3, 3), activation=tf.nn.leaky_relu, padding='VALID')
        weights = conv2d_batch_norm(weights, 64, (3, 3), activation=tf.nn.leaky_relu, padding='VALID')
        weights = conv2d_batch_norm(weights, 128, (3, 3), activation=tf.nn.leaky_relu, padding='VALID')
        weights = conv2d_batch_norm(weights, 256, (3, 3), activation=tf.nn.leaky_relu, padding='VALID')
        # weights = flatten(net_in)
        weights = flatten(weights)
        window = tf.nn.pool(net_in, (6, 6), 'AVG', 'VALID', strides=(3, 3))
        window = flatten(window)
        weights = tf.concat((window, weights), 1)
        weights = dense(weights, 1000, kernel_initializer=glorot_init, activation=tf.nn.sigmoid)
        weights = dense(weights, 500, kernel_initializer=glorot_init, activation=tf.nn.sigmoid)
        weights = dense(weights, 256, kernel_initializer=glorot_init, activation=tf.nn.sigmoid)
        out_layer = dense(weights, 1, kernel_initializer=glorot_init, activation=tf.nn.sigmoid)
    return out_layer

# Build Networks
gen_input = tf.placeholder(tf.float32, shape=[None, latent_size], name='gen_input')
disc_input = tf.placeholder(tf.float32, shape=[None, *target_shape, color_channels], name='disc_input')
is_train = tf.placeholder(tf.bool, name="is_train")

# noisy_input = noisy(disc_input, 0.03)

latent_space = encoder(disc_input)

# # Build Generator Network
gen_sample = decoder(gen_input)
ae_sample = decoder(latent_space, reuse=True)
#
# # Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample, True)
#
# # Build Loss
gen_loss = -tf.reduce_mean(tf.log(disc_fake))
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
# ae_loss = tf.reduce_mean(tf.losses.absolute_difference(disc_input, ae_sample))
ae_loss = tf.reduce_mean(tf.losses.mean_squared_error(disc_input, ae_sample))
#
# # Training Variables for each optimizer
g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
#
# # Create training operations
train_gen = tf.train.AdamOptimizer(learning_rate).minimize(gen_loss, var_list=g_vars)
train_disc = tf.train.AdamOptimizer(learning_rate).minimize(disc_loss, var_list=d_vars)
ae_train = tf.train.AdamOptimizer(learning_rate).minimize(ae_loss)



print ('loading images...')
images = load_images(args.images, get_reshaped_folder(args.out, target_shape), target_shape)

input_data = Batcher(images)

print ('loaded')

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
random_seeds = [np.random.uniform(-1., 1., size=[1, latent_size]) for _ in range(save_count)]
outputted = defaultdict(int)
gen_examples = Batcher([])
ael = dl = gl = 1.0
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(0, num_steps):
        if i % print_freq == 0:
            print('Step %i' % (i), end='\r')
        # Prepare Data
        batch_x = input_data.batch(batch_size)
        # Generate noise to feed to the decoder

        # Train
        optimizer = flip_flop.flip()
        if ael > autoencoder_target or optimizer == 'autoencoder':
            feed_dict = {disc_input: batch_x, is_train: True}
            _, ael = sess.run([ae_train, ae_loss], feed_dict=feed_dict)

            z = np.random.uniform(-1., 1., size=[6, latent_size])
            g = sess.run([gen_sample], feed_dict={gen_input: z, is_train: True})
            feed_dict = {disc_input: batch_x[:6], gen_sample: g[0], is_train: True}
            _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict)

        elif optimizer == 'gan':
            z = np.random.uniform(-1., 1., size=[len(batch_x), latent_size])
            feed_dict = {disc_input: batch_x, gen_input: z, is_train: True}
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict=feed_dict)

        elif optimizer == 'gan-replay':
            z = np.random.uniform(-1., 1., size=[1, latent_size])
            g = sess.run([gen_sample], feed_dict={gen_input: z, is_train: True})
            gen_examples.append(g[0][0])
            gen_batch = gen_examples.batch(batch_size)
            z = np.random.uniform(-1., 1., size=[min(len(gen_batch), batch_size), latent_size])
            _, gl = sess.run([train_gen, gen_loss], feed_dict={gen_input: z, is_train: True})
            feed_dict = {disc_input: batch_x[:len(gen_batch)], gen_sample: gen_batch[:len(batch_x)], is_train: True}
            _, dl = sess.run([train_disc, disc_loss], feed_dict=feed_dict)


        if np.isnan(ael):
            print('Step %i: Autoencoder Loss: %f, Generator Loss: %f, Discriminator Loss: %f' % (i, ael, gl, dl))
            break
        if i % test_freq == 0:
            print('Step %i: Autoencoder Loss: %f, Generator Loss: %f, Discriminator Loss: %f' % (i, ael, gl, dl))

        if i % save_freq == 0:
            # ls = sess.run([latent_space], feed_dict=feed_dict)
            # print (ls[0][0])
            for z in random_seeds[::3]:
                g = sess.run([ae_sample], feed_dict={latent_space: z, is_train: False})
                save_image(start_time, g[0][0], outputted['out'])
                outputted['out'] += 1
            for _ in random_seeds[1::3]:
                g = sess.run([ae_sample], feed_dict={latent_space: np.random.uniform(-1., 1., size=[1, latent_size]), is_train: False})
                save_image(start_time, g[0][0], outputted['out'])
                outputted['out'] += 1
            for item in input_data.batch(save_count//3):
                g = sess.run([ae_sample], feed_dict={disc_input: [item], is_train: False})
                save_image(start_time, g[0][0], outputted['out'])
                outputted['out'] += 1
            if args.show_transition:
                start, end = input_data.batch(2)
                start_image = sess.run([latent_space], feed_dict={disc_input: [start], is_train: False})[0]
                end_image = sess.run([latent_space], feed_dict={disc_input: [end], is_train: False})[0]
                for state in slide(start_image, end_image, save_count):
                    g = sess.run([ae_sample], feed_dict={latent_space: state, is_train: False})
                    save_image(start_time, g[0][0], outputted['slide'], '_s')
                    outputted['slide'] += 1


    # draw(sess)
