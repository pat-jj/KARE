import numpy as np
import h5py
import os

# Paths to your memory-mapped files
vectors_file = 'embeddings_test.dat'
ids_file = 'pmids_test.dat'

# Path to your new HDF5 file
hdf5_file = 'pubmed_embeddings.h5'

dtype_vectors = 'float32'
dtype_ids = 'int64'

# Load a small part of the memory-mapped file to infer the dimension
dimension = 768

print(f'dimension = {dimension}')

# Get the number of rows by dividing the file size by the size of one row
vectors_size = os.path.getsize(vectors_file)
ids_size = os.path.getsize(ids_file)
num_rows_vectors = vectors_size // (np.dtype(dtype_vectors).itemsize * dimension)
num_rows_ids = ids_size // np.dtype(dtype_ids).itemsize
assert num_rows_vectors == num_rows_ids, f"Number of rows in vectors and ids files must match {num_rows_vectors} != {num_rows_ids}"
num_rows = num_rows_vectors

print(f'num_rows = {num_rows} dimension = {dimension}')

# Load the memory-mapped arrays
doc_vectors_memmap = np.memmap(vectors_file, dtype=dtype_vectors, mode='r', shape=(num_rows, dimension))
doc_ids_memmap = np.memmap(ids_file, dtype=dtype_ids, mode='r', shape=(num_rows,))

# Create the HDF5 file and write the datasets
with h5py.File(hdf5_file, 'w') as f:
    f.create_dataset('doc_vectors', data=doc_vectors_memmap, dtype=dtype_vectors, chunks=True)
    f.create_dataset('doc_ids', data=doc_ids_memmap, dtype=dtype_ids, chunks=True)

print("Conversion to HDF5 completed successfully.")


