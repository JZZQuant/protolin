import numpy as np


def generate_batches( array, batch):
    row_indexes = np.arange(array.shape[0])
    while row_indexes.shape[0] > batch:
        sample_indexes = list(np.random.choice(np.arange(len(row_indexes)), batch, replace=False))
        val = np.take(row_indexes,sample_indexes)
        row_indexes=np.delete(row_indexes,sample_indexes)
        yield list(val)
