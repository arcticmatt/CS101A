# -*- coding: utf-8 -*-

""" Deep Neural Network for MNIST dataset classification task.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist

NFEATURES = 10
NCLASSES = 2
NPOINTS = int(2 ** NFEATURES)
points = []
labels = []
for i in xrange(NPOINTS):
    bin_rep = map(int, list(bin(i)[2:]))
    padded_bin_rep = [0] * (NFEATURES - len(bin_rep)) + bin_rep
    # if sum(padded_bin_rep) >= NFEATURES / 2:
    if padded_bin_rep[0] == 1:
        labels.append([0, 1])
    else:
        labels.append([1, 0])
    points.append(padded_bin_rep)

X = np.array(points)
Y = np.array(labels)



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







# Building deep neural network
input_layer = tflearn.input_data(shape=[None, NFEATURES])
dense1 = tflearn.fully_connected(input_layer, 10, activation='relu',
                                 regularizer='L2')
dense2 = tflearn.fully_connected(dense1, 10, activation='relu',
                                 regularizer='L2')
softmax = tflearn.fully_connected(dense2, NCLASSES, activation='softmax')

# Regression using SGD with learning rate decay 
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.98, decay_step=1000)
net = tflearn.regression(softmax, optimizer=sgd, 
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=100, validation_set=0.2,
          show_metric=True, run_id="dense_model")

numCorrect = 0.0
for j in xrange(NPOINTS):
    pred = model.predict(X[j].reshape(1, NFEATURES))
    lab = Y[j][0]
    print("Pred: %s, lab: %s"%(pred, lab))                    
    numCorrect += (pred == lab)
print("Actual accuracy: %s"%(numCorrect / NPOINTS))