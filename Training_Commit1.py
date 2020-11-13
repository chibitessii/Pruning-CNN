from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import tempfile
import numpy as np
import tensorflow as tf
import json
import re
import tensorflow_model_optimization as tfmot
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt
import pandas as pd

from six.moves import xrange
from datetime import datetime
from data import distorted_inputs
from model import select_model
from tensorflow import keras
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.contrib.layers import *
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import Model
from tensorflow import data
from tensorflow.contrib.model_pruning.python import pruning
from tensorflow.contrib.model_pruning.python.layers import layers
from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_boolean('prune', False,
                           'no pruning')

def loss(logits, labels):  # Levi Hassner 
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    losses = tf.get_collection('losses')
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = cross_entropy_mean + LAMBDA * sum(regularization_losses)
    tf.summary.scalar('tl (raw)', total_loss)
    #total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss

 if is_pruning is True:
            k_input, logits, model = model_fn(md['nlabels'], images, 1-FLAGS.pdrop, True)
            print('-------------------------MODEL------------------------ %s' % model)

        elif FLAGS.model_type == 'prune bn':
            logits = model_fn(md['nlabels'], images, 1-FLAGS.pdrop, True)
            print('-------------------------LOGITS------------------------ %s' % logits)

        else: 
            _, logits, _, = model_fn(md['nlabels'], images, 1-FLAGS.pdrop, True)
            print('-------------------------LOGITS------------------------ %s' % logits)

        total_loss = loss(logits, labels)
        train_op = optimizer(FLAGS.optim, FLAGS.eta, total_loss, FLAGS.steps_per_decay, FLAGS.eta_decay_rate)       
       
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))

        tf.global_variables_initializer().run(session=sess)

 		checkpoint_path = '%s/%s' % (run_dir, FLAGS.checkpoint)
        if tf.gfile.Exists(run_dir) is False:
            print('Creating %s' % run_dir)
            tf.gfile.MakeDirs(run_dir)

        tf.train.write_graph(sess.graph_def, run_dir, 'model.pb', as_text=True)

        tf.train.start_queue_runners(sess=sess)


        summary_writer = tf.summary.FileWriter(run_dir, sess.graph)
        steps_per_train_epoch = int(md['train_counts'] / FLAGS.batch_size)
        num_steps = FLAGS.max_steps if FLAGS.epochs < 1 else FLAGS.epochs * steps_per_train_epoch

        #validation_split = 0.1

        #num_images = k_input.shape[0] * (1 - validation_split)
        #end_step = np.ceil(num_images / FLAGS.batch_size).astype(np.int32) * FLAGS.epochs
        print('Requested number of steps [%d]' % num_steps)


        if is_pruning is True:

            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

            pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=0.80,
                                                               begin_step=0,
                                                               end_step=num_steps)
            }

            print('-----------------PRUNING------------------')

            logdir = tempfile.mkdtemp()
            print('-----------------------Writing training logs to----------------------' + logdir)

            model_for_pruning = prune_low_magnitude(model, **pruning_params)

            model_for_pruning.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

            model_for_pruning.fit(k_input, labels,
            epochs=FLAGS.epochs, 
            callbacks=[tfmot.sparsity.keras.UpdatePruningStep()], 
            steps_per_epoch=num_steps,
            verbose=1)

        tf.saved_model.save(model_for_pruning, FLAGS.checkpoint)
        saver.save(sess, checkpoint_path, global_step=num_steps)