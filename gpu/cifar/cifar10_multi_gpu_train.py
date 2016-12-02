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

"""A binary to train CIFAR-10 using multiple GPU's with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cifar10

import train_utils

import freeze_graph

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 8,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_boolean('prod_dataset', True,
                            """
                            If true, expects data to be in the form of our 'production' dataset,
                            which has a corrupt CSV header
                            """)

# TODO(smurching): Assert this while reading training examples
tf.app.flags.DEFINE_integer('num_subsamples', 1324, 
                            """Number of sampled values for each MFCC coefficient""")

tf.app.flags.DEFINE_integer('num_coeffs', 100, 
                            """Number of MFCC coefficients""")

tf.app.flags.DEFINE_string('train_data', None, 'Training data HDF5 file')

READER = train_utils.HDF5BatchProcessor(filename=FLAGS.train_data,
  batch_size=FLAGS.batch_size)

def tower_loss(scope, images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  logits = cifar10.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = cifar10.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % train_utils.TOWER_NAME, '', l.op.name)
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.scalar_summary(loss_name +' (raw)', l)
    tf.scalar_summary(loss_name, loss_averages.average(l))

  with tf.control_dependencies([loss_averages_op]):
    total_loss = tf.identity(total_loss)
  return total_loss

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def build_feed_dict(placeholder_dict, reader):
  result = {}
  for features_placeholder, labels_placeholder in placeholder_dict.values():
    features, labels = train_utils.inputs(reader)
    result[features_placeholder] = features
    result[labels_placeholder] = labels
  return result

def build_placeholder_dict():
    placeholder_dict = {}
    for i in xrange(FLAGS.num_gpus):
      features_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 
        FLAGS.num_subsamples, FLAGS.num_coeffs, 1])
      label_placeholder = tf.placeholder(tf.float32, shape=[FLAGS.batch_size,])
      scope_name = train_utils.get_scope_name(i)
      placeholder_dict[scope_name] = (features_placeholder, label_placeholder)
    return placeholder_dict  

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    # TODO(smurching): Update this?
    num_batches_per_epoch = (cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             FLAGS.batch_size)
    decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    cifar10.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)


    # Set up dict mapping GPU scopes to placeholders
    placeholder_dict = build_placeholder_dict()


    # Calculate the gradients for each model tower.
    tower_grads = []
    print("Num gpus = {}".format(FLAGS.num_gpus))
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        scope_name = train_utils.get_scope_name(i)
        with tf.name_scope(scope_name) as scope:
          print("Reusing variables for scope %s?: %s"%(tf.get_variable_scope().name, tf.get_variable_scope().reuse))
          # Look up placeholder images, labels for current device
          image, labels = placeholder_dict[scope_name]

          # Calculate the loss for one tower of the CIFAR model. This function
          # constructs the entire CIFAR model but shares the variables across
          # all towers.
          loss = tower_loss(scope, image, labels)

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Calculate the gradients for the batch of data on this CIFAR tower.
          grads = opt.compute_gradients(loss)

          # Keep track of the gradients across all towers.
          tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.scalar_summary('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.histogram_summary(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.histogram_summary(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.merge_summary(summaries)

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in xrange(FLAGS.max_steps):

      # Map each placeholder in placeholder_dict to a new batch of data
      feed_dict = build_feed_dict(placeholder_dict, READER)

      start_time = time.time()
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))
        sys.stdout.flush()
        sys.stderr.flush()

      if step % 100 == 0:
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

      # Save the model checkpoint periodically and freeze the graph.
      if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)        
        # # Save model checkpoint.
        # print('Saving model checkpoint...')
        # checkpoint_prefix = os.path.join(FLAGS.train_dir, 'model.ckpt')
        # checkpoint_state_name = 'checkpoint_state'
        # checkpoint_path = saver.save(sess, checkpoint_prefix, global_step=step,
        #                              latest_filename=checkpoint_state_name)
        
        # input_graph_name = 'input_graph.pb'
        # output_graph_name = "output_graph.pb"
        # # Save model structure.
        # print('Saving model structure...')
        # tf.train.write_graph(sess.graph_def, FLAGS.train_dir, input_graph_name)

        # # print(sess.graph_def)
        # # Get names of all tensors
        # names = [n.name for n in sess.graph_def.node if '/' not in n.name and n.name != 'init']
        # all_names = ','.join(names)

        # # Freeze graph.
        # input_graph_path = os.path.join(FLAGS.train_dir, input_graph_name)
        # input_saver_def_path = ''
        # input_binary = False
        # output_node_names = all_names 
        # # output_node_names = 'global_step'
        # restore_op_name = 'save/restore_all'
        # filename_tensor_name = 'save/Const:0'
        # output_graph_path = os.path.join(FLAGS.train_dir, output_graph_name)
        # clear_devices = True

        # # print('===== output_node_names = {} ====='.format(output_node_names))
        # print('Freezing graph...')
        # s = time.time()
        # try:
        #   freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
        #                             input_binary, checkpoint_path,
        #                             output_node_names, restore_op_name,
        #                             filename_tensor_name, output_graph_path,
        #                             True, '')
        # except Exception as e:
        #   print('Freezing graph failed with exception {}'.format(e))
        # print("Froze graph, took %s sec"%(time.time() - s))

def redirect_output():
  prefix = datetime.now().strftime("%b-%d-%y-%I:%M:%S")
  tf.gfile.MakeDirs("runs/")
  stdout_file = "runs/%s.out"%prefix
  stderr_file = "runs/%s.err"%prefix
  print("Redirecting stdout to %s, stderr to %s"%(stdout_file, stderr_file))
  sys.stdout = open(stdout_file, 'w')
  sys.stderr = open(stderr_file, 'w')


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  redirect_output()
  train()


if __name__ == '__main__':
  tf.app.run()
