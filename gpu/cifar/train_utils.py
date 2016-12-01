import os
import sys
import numpy as np
import time

import h5py
import tensorflow as tf
from hdf5_utils import *

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def get_scope_name(tower_idx):
  '''
  Returns the name of the scope for a specific tower.
  '''
  return '%s_%d' % (TOWER_NAME, tower_idx)

class HDF5BatchProcessor:
    '''
    Reads batches of data from an HDF5 file assumed to contain a root group with the 
    following keys:

    <LABELS_KEY>: Key for an array of all labels
    <NUM_ROWS_KEY>: Key for the integer number of rows in the training dataset 
    <ROW_KEY>_i, for i in xrange(num_rows): Keys for each row in the training dataset
    <DIMS_KEY>: Key for an array containing the shape of each training point (excluding label)
    '''

    def __init__(self, filename, batch_size):
        self.data_file = h5py.File(filename, 'r')
        self.num_rows = self.data_file[NUM_ROWS_KEY][...][0]
        self.labels = self.data_file[LABELS_KEY][...]
        self.batch_size = batch_size
        self.batch_dims = np.append([batch_size], self.data_file[DIMS_KEY][...])
        self.row_idx = 0

    def get_batch(self):
        features = np.zeros(self.batch_dims, dtype='float32')
        labels = np.zeros(self.batch_size, dtype='float32')
        # Get <batch_size> lines, splitting each one into a tuple of (features, label)        
        for i in xrange(self.batch_size):
            data, label = self.get_row()
            features[i][...] = data
            labels[i] = label
        return (features, labels)

    def get_row(self):
        row_key = get_key_for_row(self.row_idx)
        data = self.data_file[row_key][...]
        label = self.labels[self.row_idx]
        self.row_idx = (self.row_idx + 1) % self.num_rows
        return (data, label)


def inputs(batch_processor):
  """Construct input for training our song classification model.

  Args:
    batch_processor: BatchProcessor object to use 

  Returns:
    images: Images. 4D numpy array of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D numpy array of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  return batch_processor.get_batch()

if __name__ == "__main__":
    for i in xrange(10):
        start = time.time()
        inputs(batch_processor)
        print("Got batch in %s seconds"%(time.time() - start))
