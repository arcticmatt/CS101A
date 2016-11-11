# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/wide_n_deep_tutorial.py

"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from six.moves import urllib

import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", "Base directory for output models.")
flags.DEFINE_integer("train_steps", 200, "Number of training steps.")
flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")

LABEL_COLUMN = "decade"
ID_COLUMN = "song_id"

def get_col_names(df):
  return df.columns.tolist()

def get_feature_cols(df):
  cols = set(get_col_names(df))
  cols.remove(LABEL_COLUMN)
  cols.remove(ID_COLUMN)
  return list(cols)

def build_estimator(model_dir, df):
  """Build an estimator to be fit on the passed-in dataframe"""

  # Get columns containing continuous feature values
  cols = get_feature_cols(df)

  # Continuous base columns.
  deep_columns = [tf.contrib.layers.real_valued_column(col) for col in cols]
  m = tf.contrib.learn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns,
                                     hidden_units=[100, 50], n_classes=3)
  return m

def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in get_feature_cols(df)}

  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)

  # Returns the feature columns and the label.
  return continuous_cols, label

def encode_labels(df):
  # Transform year into a 0/1 binary classification label
  df[LABEL_COLUMN] = (df[LABEL_COLUMN].apply(lambda x: x >= 1980)).astype(int)

def train_and_eval():
  """Train and evaluate the model."""
  train_file_name = FLAGS.train_data
  model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
  train_steps = FLAGS.train_steps
  return train_eval_helper(train_file_name, model_dir, train_steps)

def train_eval_helper(train_file_name, model_dir, train_steps):
  all_data = pd.read_csv(train_file_name, engine="python")
  encode_labels(all_data)

  df_train, df_test = train_test_split(all_data, test_size = 0.2)
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, df_train)
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))
  return m  

def main(_):
  return train_and_eval()

def getModel(filename, model_dir="", train_steps=200):
  return train_eval_helper(filename, model_dir, train_steps)

if __name__ == "__main__":
  tf.app.run()