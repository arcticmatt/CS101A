import os
import sys
import numpy as np
import time

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

# Hard-coded label/id columns
LABEL_COL = "decade"
ID_COL = "song_id"
CSV_DELIM = ","

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def get_scope_name(tower_idx):
  '''
  Returns the name of the scope for a specific tower.
  '''
  return '%s_%d' % (TOWER_NAME, tower_idx)

class BatchReader:
    '''
    Reads batches of <batch_size> lines from the passed-in file.
    '''
    def __init__(self, filename, batch_size, line_processor, num_features):
        self.batch_size = batch_size
        self.line_processor = line_processor
        self.num_features = num_features

        # Set schema of processor
        self.line_processor.set_schema(filename)
        assert(len(self.line_processor.schema) == num_features + 2)

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

    def getline(self):
        line = self.handle.readline()
        if len(line) > 0:
            return line
        self.reset_handle()
        return self.handle.readline()

    def get_batch(self):
        data = []
        labels = []
        # Get <batch_size> lines, splitting each one into a tuple of (features, label)        
        for i in xrange(self.batch_size):
            features, label = self.line_processor.process(self.getline())
            data.append(features)
            labels.append(label)
        return (data, labels)


class SongFeatureExtractor:
    def __init__(self, id_col=ID_COL, label_col=LABEL_COL):
        self.id_col = id_col
        self.label_col = label_col
        self.schema = None

    def set_schema(self, filename):
        '''
        Sets schema of feature extractor given a filename
        '''
        with open(filename) as f: 
            if FLAGS.prod_dataset:
                nfeatures = FLAGS.num_coeffs * FLAGS.num_subsamples
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
                result.append(vals[i])
        return result

    def encode_label(self, label):
        if FLAGS.prod_dataset:
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
        # its shape matches what's desired for the RNN implementation (num_samples x num_coeffs)
        features = map(float, self.drop(vals, self.id_col, self.label_col))
        features = np.transpose(np.reshape(features, [FLAGS.num_coeffs, FLAGS.num_subsamples]))

        label_idx = self.schema[self.label_col]
        label = self.encode_label(vals[label_idx])
        return (features, label)

def inputs(reader):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  start = time.time()
  data, labels = reader.get_batch()
  batch_size = len(data)
  data = np.float32(np.reshape(data, [batch_size, FLAGS.num_subsamples, FLAGS.num_coeffs, 1]))
  # print("In inputs(): Got batch in %s sec"%(time.time() - start))
  return (data, np.float32(labels))
