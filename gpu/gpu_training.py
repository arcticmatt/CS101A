from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
from tflearn.models.dnn import DNN
import tensorflow.contrib.slim as slim


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('subset', 'train',
                           """Either 'train' or 'validation'.""")

# Flags governing the hardware employed for running TensorFlow.
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

# Flags governing the type of training.
tf.app.flags.DEFINE_boolean('fine_tune', False,
                            """If set, randomly initialize the final layer """
                            """of weights in order to train the network on a """
                            """new task.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# With 8 Tesla K40's and a batch size = 256, the following setup achieves
# precision@1 = 73.5% after 100 hours and 100K steps (20 epochs).
# Learning rate decay factor selected from http://arxiv.org/abs/1404.5997.
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16,
                          """Learning rate decay factor.""")

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.




# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# Sid's Code
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================

# Hard-coded label/id columns
LABEL_COL = "decade"
ID_COL = "song_id"

def encode_labels(df):
  # Transform year into a 0/1 binary classification label
  df[LABEL_COL] = (df[LABEL_COL].apply(lambda x: x >= 1980)).astype(int)

def get_data(filename):
    # Read data from CSV, encode label columns
    df = pd.read_csv(filename)
    encode_labels(df)

    # Drop ID column, separate data and label cols
    labels = df[LABEL_COL].as_matrix()

    df.drop([LABEL_COL, ID_COL], axis=1, inplace=True)
    data = df.as_matrix()
    return (data, labels)


def get_synthetic_data(nfeatures, nclasses):
    nfeatures = 12
    nclasses = 2
    npoints = int(2 ** nfeatures)
    points = []
    labels = []
    for i in xrange(npoints):
        bin_rep = map(int, list(bin(i)[2:]))
        padded_bin_rep = [0] * (nfeatures - len(bin_rep)) + bin_rep
        # Classification rule: If feature 0 is 1, point has label 1
        if padded_bin_rep[0] == 1:
            labels.append([0, 1])
        else:
            labels.append([1, 0])
        points.append(padded_bin_rep)

    X = np.array(points)
    Y = np.array(labels)
    return (X, Y)



# Loads an array of training data batches from the passed-in data file
def split_batches(data, labels):
    # Split the batch of images and labels for towers.
    data_splits = np.array_split(data, FLAGS.num_gpus)
    labels_splits = np.array_split(labels, FLAGS.num_gpus)
    return (data_splits, labels_splits)

def error(model, data, labels):
    '''
    Compute classification accuracy of passed-in model on <data>, which is
    a 2D numpy array such that data[i] has label labels[i]
    '''
    npoints = len(data)
    nfeatures = len(data[0])
    numCorrect = 0.0
    for i in xrange(len(data)):
        data_point = data[i].reshape(1, nfeatures)
        print(data_point, data_point.shape)
        pred = model.predict(data_point)
        lab = labels[i][0]
        print("Pred: %s, lab: %s"%(pred, lab))                    
        numCorrect += (pred == lab)
    return numCorrect / npoints

# Define a model builder
def build_model(placeholder_X):
    '''
    Returns a DNN model that can be trained given a batch of input data
    as a 2D tensor of shape (nsamples, input_dim)
    '''

    model = tflearn.fully_connected(placeholder_X, 10, activation='relu', 
        scope='layer1', regularizer='L2', weight_decay=0.001)    
    model = tflearn.fully_connected(model, 10, activation='relu', 
        scope='layer2', regularizer='L2', weight_decay=0.001) 
    model = tflearn.fully_connected(model, 2, activation='softmax', 
        scope='layer3', regularizer='L2', weight_decay=0.001) 

    sgd = tflearn.SGD(learning_rate=0.3, lr_decay=0.98, decay_step=1000)
    model = tflearn.regression(model, optimizer=sgd, loss='categorical_crossentropy')
    return DNN(model)

def train_from_file(filename):
    data, labels = get_data(filename)
    data_batches, label_batches = split_batches(data, labels)
    return train(data_batches, label_batches)

def train(data_batches, label_batches):
    first_batch = data_batches[0]
    # Number of features = number of cols in first row of first batch
    nfeatures = len(first_batch[0])
    # TODO(smurching): num batches won't always equal num gpus
    for i in xrange(FLAGS.num_gpus):
        # with tf.device("/gpu:%s"%i):
            # # Force all Variables to reside on the CPU.
            # with slim.arg_scope([slim.variables.variable], device='/cpu:0'):   
            data = data_batches[i]
            labels = label_batches[i]

            # Number of points in the current batch
            npoints = len(data)
            print(len(labels), npoints)
            labels = labels.reshape([npoints, 2])
            # print("Mean label: %s"%np.mean(labels))
            # print("Input dim: %s"%(nfeatures))
            placeholder_X = tflearn.input_data(shape=[None, nfeatures])
            model = build_model(placeholder_X)
            model.fit(data, labels, n_epoch=10, show_metric=True, validation_set=0.2)

            # Reuse variales acros GPUs
            tf.get_variable_scope().reuse_variables()

    # TODO(smurching): Can't predict using
    # Create a dummy model, fit on one point
    placeholder_X = tflearn.input_data(shape=[None, nfeatures])
    model = build_model(placeholder_X)

    data = data_batches[0]
    labels = label_batches[0]

    # Number of points in the current batch
    npoints = len(data)
    labels = labels.reshape([npoints, 2])

    print(data, data.shape)
    print(labels, labels.shape)

    model.fit(data, labels, n_epoch=10, show_metric=True, validation_set=0.2)
    return model

def train_and_eval(filename):
    '''
    Train a model on the data in the passed-in file
    '''
    model = train_from_file(filename)

    # As a sanity check, compute error of the returned model on the training set
    data, labels = get_data(filename)

    print("Error: %s"%(error(model, data, labels)))

def train_and_eval(nfeatures, nclasses):
    '''
    Trains and evaluates a model on synthetic data generated via
    get_synthetic_data
    '''
    data, labels = get_synthetic_data(nfeatures, nclasses)
    data_batches, label_batches = split_batches(data, labels)
    model = train(data_batches, label_batches)

    print("Error: %s"%(error(model, data, labels)))



# Model can now be trained by multiple GPUs (see gradient averaging)

if __name__ == "__main__":
    # filename = sys.argv[1]
    # train(filename)
    train_and_eval(nfeatures=12, nclasses=2)