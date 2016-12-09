# Keys for HDF5 storage
LABELS_KEY = "labels"
NUM_ROWS_KEY = "num_rows"
ROW_KEY = "row"
DIMS_KEY = "dims"

def get_key_for_row(i):
    '''
    Return HDF5 key for row i.
    '''
    return "%s_%s"%(ROW_KEY, i)
