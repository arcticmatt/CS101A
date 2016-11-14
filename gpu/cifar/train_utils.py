import os
import sys
import numpy as np
import time

# Hard-coded label/id columns
LABEL_COL = "decade"
ID_COL = "song_id"
CSV_DELIM = ","

NCOEFFS = 42
NSAMPLES = 86

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
    Reads blocks of roughly <batch_size> bytes from the passed-in file.
    '''
    def __init__(self, num_devices, filename, batch_size, line_processor):
        self.filename = filename
        self.filesize = os.path.getsize(filename) 
        self.batch_size = batch_size
        self.line_processor = line_processor

        # Set schema of processor
        line_processor.set_schema(filename)

        # Assign batches to each device
        num_batches = (self.filesize + self.batch_size - 1) / self.batch_size

        self.assignments = {get_scope_name(i): [] for i in xrange(num_devices)}
        self.offsets = {get_scope_name(i) : 0 for i in xrange(num_devices)}

        # Assign each batch to a device
        for i in xrange(num_batches):
            device_idx = i % num_devices
            scope = get_scope_name(device_idx)
            self.assignments[scope].append(i)

    def read_batch(self, batch_idx):
        with open(self.filename) as f:
            # Seek to the starting offset of our batch
            f.seek(batch_idx * self.batch_size)
            # Read to the end of the next line. Note that we might skip
            # over the first line of our batch (if we've seeked to the exact start
            # of a CSV line). This (feature not a bug lol) lets us skip the header line of 
            # the CSV.
            data = []
            labels = []
            line = f.readline()
            bytes_read = 0
            old_size = 0
            while bytes_read < self.batch_size:
                line = f.readline()
                if len(line) == 0:
                    break
                features, label = self.line_processor.process(line)
                data.append(features)
                labels.append(label)

                new_size = sys.getsizeof(data) + sys.getsizeof(labels)
                bytes_read += (new_size - old_size)
                old_size = new_size

            return (data, labels)

    def get_offset(self, batch_idx):
        return self.batch_idx * batch_idx

    def get_batch(self, scope):
        start = time.time()
        # Get offset of next batch for current tower
        next_batch_offset = self.offsets[scope]

        # Look up index of next batch
        batch_idx = self.assignments[scope][next_batch_offset] 

        # Read the batch
        batch = self.read_batch(batch_idx)

        # Update reader (increment batch offset for current tower)
        num_batches_for_tower = len(self.assignments[scope])
        self.offsets[scope] = (self.offsets[scope] + 1) % num_batches_for_tower
        end = time.time()
        print("Got batch of size %s in time %s"%(len(batch[0]), end - start))
        return batch

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
            header = f.readline()
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
        if int(label) > 1980:
            return 1
        return 0

    def process(self, line):
        assert(self.schema is not None)
        vals = line.split(CSV_DELIM)
        features = map(float, self.drop(vals, self.id_col, self.label_col))
        label_idx = self.schema[self.label_col]
        label = self.encode_label(vals[label_idx])
        return (features, label)

def inputs(reader, scope):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """

  data, labels = reader.get_batch(scope)
  batch_size = len(data)
  data = np.reshape(data, [batch_size, NCOEFFS, NSAMPLES, 1])
  return data, labels


if __name__ == "__main__":
    path = sys.argv[1]
    num_devices = int(sys.argv[2])

    processor = SongFeatureExtractor()
    reader = BatchReader(num_devices=num_devices, filename=path, batch_size=int(1e7), line_processor=processor)
    print reader.assignments
    print reader.offsets

