import sys
import numpy as np
import time
from hdf5_utils import *

import h5py

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
