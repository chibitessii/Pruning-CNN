from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_model_optimization as tfmot

from datetime import datetime
import time
import os
import numpy as np
import tensorflow as tf
from data import distorted_inputs
import re
import tensorflow.keras as keras
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.contrib.layers import *
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import Model
import tensorflow_datasets as tfds
from tensorflow import data
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.examples.tutorials.mnist import input_data

    epochs = 50
    batch_size = 1228 # Entire training set
    weight_decay = 0.0005
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    image_set = input_data.read_data_sets('~/tensor/AgeGenderDeepLearning-master/Folds/test-folds/gender_test_fold_is_3_DefaultRun', one_hot=True)

    #image = tf.placeholder(tf.float32, [None, 784])
    #label = tf.placeholder(tf.float32, [None, 10])

    layer1 = layers.masked_fully_connected(images, 512)
    layer2 = layers.masked_fully_connected(layer1, 512)
    logits = tf.nn.dropout(layer2, pkeep, name='drop1')

    batches = int(len(image_set.train.images) / batch_size)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=nlabels, logits=logits))

    with tf.variable_scope("Prune_Layer", "Prune_Layer", [images]) as scope:

        pruning_hparams = pruning.get_pruning_hparams()
        print("Pruning Hyperparameters:", pruning_hparams)

        # Change hyperparameters to meet our needs
        pruning_hparams.begin_pruning_step = 0
        pruning_hparams.end_pruning_step = 250
        pruning_hparams.pruning_frequency = 1
        pruning_hparams.sparsity_function_end_step = 250
        pruning_hparams.target_sparsity = .9
        global_step = tf.train.get_or_create_global_step()

        #train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)
        reset_global_step_op = tf.assign(global_step, 0)
        
        p = pruning.Pruning(pruning_hparams, global_step=global_step, sparsity=.9)
        prune_op = p.conditional_mask_update_op()
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(nlabels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            # Reset the global step counter and begin pruning
            sess.run(reset_global_step_op)
            for epoch in range(epochs):
                for batch in range(batches):
                    batch_xs, batch_ys = image_set.train.next_batch(batch_size)
                    # Prune and retrain
                    sess.run(prune_op)
                    #sess.run(train_op, feed_dict={images: batch_xs, label: batch_ys})

                #if epoch % 10 == 0:
                   # acc_print = sess.run(accuracy, feed_dict={images: image_set.test.images, nlabels: image_set.test.labels})
                    #print("Pruned model step %d test accuracy %g" % (epoch, acc_print))
                    #print("Weight sparsities:", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))

            #acc_print = sess.run(accuracy, feed_dict={images: image_set.test.images, label: image_set.test.labels})
            #print("Final accuracy:", acc_print)
            #print("Final sparsity by layer (should be 0)", sess.run(tf.contrib.model_pruning.get_weight_sparsity()))

    with tf.variable_scope('output') as scope:

        weights = tf.Variable(tf.random_normal([512, nlabels], mean=0.0, stddev=0.01), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[nlabels], dtype=tf.float32), name='biases')
        output = tf.add(tf.matmul(logits, weights), biases, name=scope.name)
    return output
