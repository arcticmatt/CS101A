import train_utils
import os
import sys

# Hard-coded label/id columns
LABEL_COL = "decade"
ID_COL = "song_id"
CSV_DELIM = ","
NUM_COLS = 132400

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

        self.assignments = {train_utils.get_scope_name(i): [] for i in xrange(num_devices)}
        self.offsets = {train_utils.get_scope_name(i) : 0 for i in xrange(num_devices)}

        # Assign each batch to a device
        for i in xrange(num_batches):
            device_idx = i % num_devices
            scope = train_utils.get_scope_name(device_idx)
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
            f.readline()
            bytes_read = 0
            old_size = 0
            while bytes_read < self.batch_size:
                line = f.readline()
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
        # Get offset of next batch for current tower
        next_batch_offset = self.offsets[scope]

        # Look up index of next batch
        batch_idx = self.assignments[scope][next_batch_offset] 

        # Read the batch
        batch = self.read_batch(batch_idx)

        # Update reader (increment batch offset for current tower)
        num_batches_for_tower = len(self.assignments[scope])
        self.offsets[scope] = (self.offsets[scope] + 1) % num_batches_for_tower

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
            # TODO(smurching): Replace hard-coded header with code below
            # header = f.readline()
            # col_names = header.split(CSV_DELIM)
            col_names = [LABEL_COL] + map(str, range(NUM_COLS)) + [ID_COL]
            self.schema = {col_names[i] : i for i in xrange(len(col_names))}

    def drop(self, vals, *cols):
        result = []
        cols = set(cols)
        for i in xrange(len(vals)):
            if self.schema[i] not in cols:
                result.append(vals[i])
        return result

    def process(self, line):
        assert(self.schema is not None)
        vals = line.split(CSV_DELIM)
        return self.drop(vals, id_col, label_col)



if __name__ == "__main__":
    path = sys.argv[1]
    num_devices = int(sys.argv[2])

    processor = SongFeatureExtractor()
    reader = BatchReader(num_devices=num_devices, filename=path, batch_size=int(1e7), line_processor=processor)
    print reader.assignments
    print reader.offsets

