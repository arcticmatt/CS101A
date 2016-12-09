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

def get_eval_batch(batch_processor):
  '''
  Construct batch upon which to evaluate our song classification model.
  '''
  return batch_processor.get_eval_batch()

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
  return batch_processor.get_train_batch()
