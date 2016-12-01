import sys
import h5py
import numpy as np
import sys
import time
import train_utils
from train_utils import *

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python csv_to_hdf5.py path_to_csv path_to_hdf5_file")
        sys.exit(1)

    data_filename = sys.argv[1]
    result_filename = sys.argv[2]        
    batch_processor = BatchProcessor(filename=sys.argv[1], batch_size=None, 
        num_coeffs=100, num_subsamples=1324, prod_dataset=True)
    batch_processor.write_all(sys.argv[2])

    with h5py.File(result_filename, 'r') as data_file:
        num_rows = data_file[NUM_ROWS_KEY][...]
        labels = data_file[LABELS_KEY][...]
        dims = data_file[DIMS_KEY][...]
        print("Reading back dims: %s"%dims)

        for i in xrange(num_rows):
            s = time.time()
            row = data_file[train_utils._get_key_for_row(i)][...]
            print("Read HDF5 row in %s sec"%(time.time() - s))
            label = labels[i]
            print("feature sum: %s, label: %s"%(np.sum(row.flatten()), label))
