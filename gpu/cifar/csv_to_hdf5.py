import sys
import h5py
import numpy as np
import sys
import time
from hdf5_utils import *

# Hard-coded label/id columns
LABEL_COL = "decade"
ID_COL = "song_id"
CSV_DELIM = ","

class HDF5Converter:
    '''
    Used to convert CSV datasets to HDF5 datasets.
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


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python csv_to_hdf5.py path_to_csv num_subsamples num_coeffs prod_dataset")
        sys.exit(1)

    data_filename = sys.argv[1]
    data_filename_without_ext = ".".join(data_filename.split(".")[:-1])
    num_subsamples = int(sys.argv[2])
    num_coeffs = int(sys.argv[3])
    prod_dataset = int(sys.argv[4])

    result_filename = "%s.h5"%(data_filename_without_ext)
    batch_processor = HDF5Converter(filename=sys.argv[1], batch_size=None, 
        num_coeffs=num_coeffs, num_subsamples=num_subsamples, prod_dataset=prod_dataset)
    batch_processor.write_all(result_filename)

    with h5py.File(result_filename, 'r') as data_file:
        num_rows = data_file[NUM_ROWS_KEY][...]
        labels = data_file[LABELS_KEY][...]
        dims = data_file[DIMS_KEY][...]
        print("Reading back training example dims: %s"%dims)

        # for i in xrange(num_rows):
        #     s = time.time()
        #     row = data_file[get_key_for_row(i)][...]
        #     print("Read HDF5 row in %s sec"%(time.time() - s))
        #     label = labels[i]
        #     print("feature sum: %s, label: %s"%(np.sum(row.flatten()), label))
