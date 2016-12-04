# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import pandas as pd
import re
import sys
import tarfile
import numpy as np
import train_utils

from six.moves import urllib
import tensorflow as tf
import tflearn

from tensorflow.models.image.cifar10 import cifar10_input

FLAGS = tf.app.flags.FLAGS

# # Basic model parameters.
# tf.app.flags.DEFINE_integer('batch_size', 128,
#                             """Number of images to process in a batch.""")
# tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar10_data',
#                            """Path to the CIFAR-10 data directory.""")
# tf.app.flags.DEFINE_boolean('use_fp16', False,
#                             """Train the model using fp16.""")

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Some network parameters
# TODO(smurching): Make these flags?
NUM_CONV_LAYERS = 5
NUM_RECC_LAYERS = 2
CONV_FILTER_SIZE = [5, 5]
RECC_LAYER_SIZE = 30

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % train_utils.TOWER_NAME, '', x.op.name)
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def build_conv_layers(input_tensor, num_layers, filter_size=None):
  '''
  Returns a tensor corresponding to <num_layers> connected convolutional layers 
  with max pooling between layers. The first convolutional layer takes 
  input_tensor as its input.

  filter_size: shape (along the [num_subsamples, num_coeffs] axes) of filter
  '''

  # Set filter size to a default if none is specified
  if filter_size is None:
    filter_size = [2, 2]

  final_layer = input_tensor
  for i in xrange(num_layers):
    # Expects input tensor [batch_size, height, width, in_channels]
    # These dimensions correspond to [batch_size, num_subsamples, num_coeffs, 1] in our case
    with tf.variable_scope('conv_%d'%(i + 1)) as scope:
      conv = tflearn.layers.conv.conv_2d (final_layer, nb_filter=1, filter_size=filter_size, strides=1,
        padding='same', activation='linear', bias=True, weights_init='uniform_scaling',
        bias_init='zeros', regularizer=None, weight_decay=0.001,
        restore=True, reuse=None, scope=scope)

    # TODO(smurching): Intelligently pick values for the kernel size/stride here
    final_layer = tflearn.layers.conv.max_pool_2d (conv, kernel_size=[1, 3, 3, 1],
      strides=[1, 2, 2, 1], padding='same', name='MaxPool2D_%d'%i)
  return final_layer

def build_recurrent_layers(input_tensor, num_layers, units_per_layer=3, activation='sigmoid', dropout=0.8):
  final_layer = input_tensor
  for i in xrange(num_layers):
    is_last_layer = (i == num_layers - 1)
    if is_last_layer:
      # We don't apply softmax for the last layer because 
      # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits 
      # and performs the softmax internally for efficiency.      
      curr_activation = 'linear'      
    else:
      curr_activation = activation      

    with tf.variable_scope('recurrent_%d'%(i + 1)) as scope:
      # TODO(smurching): Pick dropout probability more intelligently, currently just a random guess
      # Get output of recurrent layer as a <timesteps>-length list of prediction tensors of shape
      # [batch_size, num_units] if this isn't our final recurrent layer. Otherwise, just get a single 2D
      # output tensor of shape [batch_size, num_units]
      with tf.device("/cpu:0"):
        final_layer = tflearn.layers.recurrent.lstm(final_layer, n_units=units_per_layer, scope=scope,
          reuse=None, activation=curr_activation, dropout=dropout, return_seq=(not is_last_layer))
      if not is_last_layer:
        final_layer = tf.pack(final_layer, axis=1)
      
  return final_layer


def inference(songs):
  """Build the CIFAR-10 model.

  Args:
    songs: MFCC data (numpy array of shape [batch_size, num_subsamples, num_coeffs, 1]) 
    returned from train_utils.inputs().


  Returns:
    Logits.

  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  print("Flags num examples: %s"%FLAGS.num_examples)
  print("Training network with %s convolutional layers, %s recurrent layers using learning rate %s"%(
    NUM_CONV_LAYERS, NUM_RECC_LAYERS, INITIAL_LEARNING_RATE))
  print("Convolutional layer filter size: %s"%CONV_FILTER_SIZE)
  print("Each recurrent layer has %s units"%RECC_LAYER_SIZE)

  convolutional_layers = build_conv_layers(songs, num_layers=NUM_CONV_LAYERS,
    filter_size=CONV_FILTER_SIZE)
  conv_reshaped = tf.squeeze(convolutional_layers, squeeze_dims=[3])
  print("RNN input shape (batch_size x timesteps x num_coeffs): %s"%conv_reshaped.get_shape())

  return build_recurrent_layers(conv_reshaped, num_layers=NUM_RECC_LAYERS,
    units_per_layer=RECC_LAYER_SIZE, activation='sigmoid')

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  # labels = tf.Print(labels, [labels], "Labels for batch: ", summarize=10)
  # logits = tf.Print(logits, [tf.nn.softmax(logits)], "Preds for batch: ", summarize=10)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(l.op.name +' (raw)', l)
    tf.scalar_summary(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.scalar_summary('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.histogram_summary(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.histogram_summary(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op
