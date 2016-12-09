import sys
import numpy as np
import time
from hdf5_utils import *

import h5py

class HDF5Reader:
    '''
    Reads batches of data from an HDF5 file assumed to contain a root group with the 
    following keys:

    <LABELS_KEY>: Key for an array of all labels
    <NUM_ROWS_KEY>: Key for the integer number of rows in the training dataset 
    <ROW_KEY>_i, for i in xrange(num_rows): Keys for each row in the training dataset
    <DIMS_KEY>: Key for an array containing the shape of each training point (excluding label)
    '''

    def __init__(self, filename, batch_size, eval_batch_size, eval_set_size=0.2):
        assert(eval_set_size >= 0 and eval_set_size <= 1.0)
        self.data_file = h5py.File(filename, 'r')
        self.labels = self.data_file[LABELS_KEY][...]
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        # Compute number of rows in the training and eval sets. Set aside the
        # first <self.eval_rows> rows as our evaluation set
        num_rows = self.data_file[NUM_ROWS_KEY][...][0]            
        self.num_eval_rows = int(eval_set_size * num_rows)
        self.eval_row_idx = 0

        ## Ensure that our eval batch size is no larger than the size of our eval set
        if self.eval_batch_size > self.num_eval_rows:
            print("WARNING: Eval batch size %s is "%(self.eval_batch_size) + 
                "bigger than eval set size %s, "%(self.num_eval_rows) + 
                "shrinking eval batch size to %s"%(self.num_eval_rows))
            self.eval_batch_size = self.num_eval_rows

        self.num_train_rows = num_rows - self.num_eval_rows
        self.train_row_idx = 0
        print("""Reading training batches of size %s from %s training examples, 
            eval batches of size %s from eval set of size %s"""%(self.batch_size,
                self.num_train_rows, self.eval_batch_size, self.num_eval_rows))

    def _get_batch_dims(self, batch_size):
        return np.append([batch_size], self.data_file[DIMS_KEY][...])

    def _get_row(self, i):
        '''
        Returns the ith dataset in our HDF5 file (the ith row in our entire
        dataset, not considering the train/eval split).
        '''        
        row_key = get_key_for_row(i)        
        data = self.data_file[row_key][...]
        label = self.labels[i]
        return (data, label)

    def _get_train_row(self):
        # The first training example is at index <num_eval_rows>, so the
        # current training example is at index <train_row_idx> + <num_eval_rows>
        result = self._get_row(self.train_row_idx + self.num_eval_rows)
        self.train_row_idx = (self.train_row_idx + 1) % self.num_train_rows
        return result

    def _get_eval_row(self):
        # Read the next eval example
        result = self._get_row(self.eval_row_idx)
        self.eval_row_idx = (self.eval_row_idx + 1) % self.num_eval_rows
        return result

    def _get_batch(self, batch_size, eval_batch=False):
        features = np.zeros(self._get_batch_dims(batch_size), dtype='float32')
        labels = np.zeros(batch_size, dtype='float32')
        # Get <batch_size> lines, splitting each one into a tuple of (features, label)        
        for i in xrange(batch_size):
            if eval_batch:
                data, label = self._get_eval_row()
            else:
                data, label = self._get_train_row()
            features[i][...] = data
            labels[i] = label
        return (features, labels)        

    def get_eval_batch(self):
        return self._get_batch(batch_size=self.eval_batch_size, eval_batch=True)

    def get_train_batch(self):
        return self._get_batch(batch_size=self.batch_size, eval_batch=False)
