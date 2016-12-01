import os
import sys
import numpy as np
import time

import h5py
import tensorflow as tf

# Hard-coded label/id columns
LABEL_COL = "decade"
ID_COL = "song_id"
CSV_DELIM = ","

# Keys for HDF5 storage
LABELS_KEY = "labels"
NUM_ROWS_KEY = "num_rows"
ROW_KEY = "row"
DIMS_KEY = "dims"

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def get_scope_name(tower_idx):
  '''
  Returns the name of the scope for a specific tower.
  '''
  return '%s_%d' % (TOWER_NAME, tower_idx)

def _get_key_for_row(i):
    '''
    Return HDF5 key for row i.
    '''
    return "%s_%s"%(ROW_KEY, i)

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
        row_key = _get_key_for_row(self.row_idx)
        data = self.data_file[row_key][...]
        label = self.labels[self.row_idx]
        self.row_idx = (self.row_idx + 1) % self.num_rows
        return (data, label)


class BatchProcessor:
    '''
    Reads batches of <batch_size> lines from the passed-in file.
    Also used to convert CSV datasets to HDF5 datasets.
    '''
    def __init__(self, filename, batch_size, num_subsamples, num_coeffs, prod_dataset):
        self.batch_size = batch_size
        self.num_subsamples = num_subsamples
        self.num_coeffs = num_coeffs
        self.prod_dataset = prod_dataset
        self.num_features = num_subsamples * num_coeffs
        self.num_epochs = 0

        # Set schema of processor
        self.line_processor = SongFeatureExtractor(num_subsamples, num_coeffs, prod_dataset)
        self.line_processor.set_schema(filename)
        assert(len(self.line_processor.schema) == self.num_features + 2)

        # Initialize file handle
        self.handle = open(filename)
        self.reset_handle()

    def reset_handle(self):
        '''
        Resets file handle to training data file.
        '''
        self.handle.seek(0)
        # Skip past header
        self.handle.readline()
        # Increment number of epochs (number of full passes through file)
        self.num_epochs += 1

    def getline(self):
        line = self.handle.readline()
        if len(line) > 0:
            return line
        self.reset_handle()
        res = self.handle.readline()
        return res

    def get_row(self):
        return self.line_processor.process(self.getline())

    def _get_batch(self):
        data = []
        labels = []
        # Get <batch_size> lines, splitting each one into a tuple of (features, label)        
        for i in xrange(self.batch_size):
            features, label = self.get_row()
            data.append(features)
            labels.append(label)
        return (data, labels)

    def get_batch(self):
        data, labels = self._get_batch()
        batch_size = len(data)
        data = np.float32(np.reshape(data, [batch_size, self.num_subsamples, self.num_coeffs, 1]))
        return (data, np.float32(labels))

    def write_all(self, output_filename):
        '''
        Writes all data for the current file to the output file in HDF5 format
        '''
        labels = []
        num_rows = 0
        with h5py.File(output_filename, 'w') as output_file:
            self.reset_handle()
            num_epochs = self.num_epochs
            while self.num_epochs == num_epochs:
                features, label = self.get_row()
                row_name = "%s_%s"%(ROW_KEY, num_rows)
                dset = output_file.create_dataset(row_name, shape=(self.num_subsamples, self.num_coeffs, 1))
                dset[...] = np.reshape(features, [self.num_subsamples, self.num_coeffs, 1])
                labels.append(label)
                num_rows += 1

            # Save all labels to a single array
            label_dset = output_file.create_dataset(LABELS_KEY, shape=(len(labels),), dtype='float32')
            label_dset[...] = labels

            # Save single-element datasets describing the number of rows and dimensionality
            # of the training data
            num_rows_dset = output_file.create_dataset(NUM_ROWS_KEY, shape=(1,), dtype='int32')
            num_rows_dset[...] = num_rows

            dims_dset = output_file.create_dataset(DIMS_KEY, shape=(3,), dtype='int32')
            dims_dset[...] = [self.num_subsamples, self.num_coeffs, 1]

class SongFeatureExtractor:
    def __init__(self, num_subsamples, num_coeffs, prod_dataset, id_col=ID_COL, label_col=LABEL_COL):
        self.prod_dataset = prod_dataset
        self.num_subsamples = num_subsamples
        self.num_coeffs = num_coeffs
        self.id_col = id_col
        self.label_col = label_col
        self.schema = None

    def set_schema(self, filename):
        '''
        Sets schema of feature extractor given a filename
        '''
        with open(filename) as f: 
            if self.prod_dataset:
                nfeatures = self.num_coeffs * self.num_subsamples
                self.col_names = [LABEL_COL] + map(str, range(nfeatures)) + [ID_COL]
            else:
                header = f.readline().strip()
                self.col_names = header.split(CSV_DELIM)
            self.schema = {self.col_names[i] : i for i in xrange(len(self.col_names))}

    def drop(self, vals, *cols):
        result = []
        cols = set(cols)
        for i in xrange(len(vals)):
            if self.col_names[i] not in cols:
                result.append(float(vals[i]))
        return result

    def encode_label(self, label):
        if self.prod_dataset:
            # From label (3rd char of year), subtract 6 (starting decade = 1960)
            return int(label) - 6
        # If we're running on the local dataset
        if int(label) > 1980:
            return 1.0
        return 0.0            

    def process(self, line):
        assert(self.schema is not None)
        vals = line.split(CSV_DELIM)
        # <features> is a matrix of num_coeffs x num_samples. We transpose it so that
        # its shape matches what's desired for the RNEN implementation (num_samples x num_coeffs)
        features = self.drop(vals, self.id_col, self.label_col)

        features = np.transpose(np.reshape(features, [self.num_coeffs, self.num_subsamples]))

        label_idx = self.schema[self.label_col]
        label = self.encode_label(vals[label_idx])
        return (features, label)

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
  start = time.time()
  result = batch_processor.get_batch()
  print("Got batch in %s sec"%(time.time() - start))
  return result


if __name__ == "__main__":
    for i in xrange(10):
        start = time.time()
        inputs(batch_processor)
        print("Got batch in %s seconds"%(time.time() - start))
