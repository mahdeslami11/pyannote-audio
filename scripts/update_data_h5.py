# update h5 files created by old versions of pyannote-speaker-embedding
# estimate mu/sigma and save it back to the file
# usage: update_data_h5.py /path/to/file.h5

import sys
import h5py
import numpy as np
from tqdm import tqdm

data_h5 = sys.argv[1]

with h5py.File(data_h5, mode='r') as fp:
    X = fp['X']
    weights, means, squared_means = zip(*(
        (len(x), np.mean(x, axis=0), np.mean(x**2, axis=0))
        for x in tqdm(X)))
    mu = np.average(means, weights=weights, axis=0)
    squared_mean = np.average(squared_means, weights=weights, axis=0)
    sigma = np.sqrt(squared_mean - mu ** 2)


with h5py.File(data_h5, mode='r+') as fp:
    X = fp['X']
    X.attrs['mu'] = mu
    X.attrs['sigma'] = sigma
