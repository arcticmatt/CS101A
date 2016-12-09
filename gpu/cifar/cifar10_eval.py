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

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10
import train_utils
import hdf5_reader

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")


READER = hdf5_reader.HDF5Reader(filename=FLAGS.train_data,
  batch_size=FLAGS.batch_size, eval_batch_size=FLAGS.eval_batch_size)

def eval_once(saver, top_k_op, feed_dict):
  """Run Eval once.

  Args:
    saver: Saver.
    top_k_op: Top K op.
    feed_dict: Feed dictionary to use in evaluating ops
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path and saver is not None:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
    elif saver is None:
      print('Not restoring model from checkpoint before eval: saver passed to eval_once is None')
    else:
      print('Failed to restore model from checkpoint for eval: no checkpoint file found')
      return

    # Evaluate on all the eval set points
    total_sample_count = FLAGS.eval_batch_size
    predictions = sess.run([top_k_op], feed_dict=feed_dict)
    true_count = np.sum(predictions) # Counts the number of correct predictions.

    # Begin evaluation on FLAGS.num_examples training points
    num_iter = int(math.ceil(READER.num_eval_rows / FLAGS.batch_size))
    print("Taking %s iters to eval on full set"%num_iter)
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = num_iter * FLAGS.batch_size

    for step in xrange(num_iter):
      predictions = sess.run([top_k_op], feed_dict=feed_dict)
      true_count += np.sum(predictions)

    # Compute precision @ 1.
    precision = true_count / total_sample_count
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
    return precision

def build_eval_dict(reader, features_placeholder, label_placeholder):
  result = {}
  features, labels = train_utils.get_eval_batch(reader)
  result[features_placeholder] = features
  result[label_placeholder] = labels
  return result

def build_eval_graph():
  '''
  Adds ops to the current graph for evaluating out-of-sample error.

  Returns a tuple (features_placeholder, label_placeholder, eval_op) comprising
  the input placeholders for evaluation, along with the op for computing
  error.
  '''
    # Get feature and label placeholders
  features_placeholder = tf.placeholder(tf.float32, shape=[None,
    FLAGS.num_subsamples, FLAGS.num_coeffs, 1])
  label_placeholder = tf.placeholder(tf.int32, shape=[None,])

  # Build a Graph that computes the logits predictions from the
  # inference model.
  logits = cifar10.inference(features_placeholder)
  
  # print("Predictions size: %s"%logits.get_shape())
  # logits = tf.Print(logits, [logits], "Logits: ", summarize=10)
  # label_print = tf.Print(label_placeholder, [label_placeholder], "Labels: ", summarize=10)

  # Calculate predictions.
  top_k_op = tf.nn.in_top_k(tf.nn.softmax(logits), label_placeholder, 1)
  return (features_placeholder, label_placeholder, top_k_op) 

def evaluate(restore=True):
  """Eval CIFAR-10 for a number of steps."""
  with tf.Graph().as_default() as g:
    
    # Build the eval graph
    features_placeholder, label_placeholder, top_k_op = build_eval_graph()

    # Restore the moving average version of the learned variables for eval.
    if restore:
      variable_averages = tf.train.ExponentialMovingAverage(
          cifar10.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
    else:
      saver = None

    while True:
      feed_dict = build_eval_dict(READER, features_placeholder, label_placeholder)
      eval_once(saver, top_k_op, feed_dict)
      if FLAGS.run_once:
        break
      time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
