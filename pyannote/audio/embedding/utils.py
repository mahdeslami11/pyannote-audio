#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2018 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

import numpy as np
import scipy.spatial.distance
import scipy.cluster.hierarchy


def l2_normalize(fX):

    norm = np.sqrt(np.sum(fX ** 2, axis=1))
    norm[norm == 0] = 1.
    return (fX.T / norm).T


def get_range(metric='euclidean'):
    return {
        'angular': (0, np.pi),
        'euclidean': (0, 2),
        'sqeuclidean': (0, 4),
        'cosine': (-1.0, 1.0)
    }.get(metric, None)


def _pdist_func_1D(X, func):
    """Helper function for pdist"""

    X = X.squeeze()
    n_items, = X.shape

    distances = []

    for i in range(n_items - 1):
        distance = func(X[i], X[i+1:])
        distances.append(distance)

    return np.hstack(distances)


def pdist(fX, metric='euclidean', **kwargs):
    """Same as scipy.spatial.distance with support for additional metrics

    * 'angular': pairwise angular distance
    * 'equal':   pairwise equality check (only for 1-dimensional fX)
    * 'minimum': pairwise minimum (only for 1-dimensional fX)
    * 'maximum': pairwise maximum (only for 1-dimensional fX)
    * 'average': pairwise average (only for 1-dimensional fX)
    """

    if metric == 'angular':
        cosine = scipy.spatial.distance.pdist(
            fX, metric='cosine', **kwargs)
        return np.arccos(np.clip(1.0 - cosine, -1.0, 1.0))

    elif metric == 'equal':
        return _pdist_func_1D(fX, lambda x, X: x == X)

    elif metric == 'minimum':
        return _pdist_func_1D(fX, np.minimum)

    elif metric == 'maximum':
        return _pdist_func_1D(fX, np.maximum)

    elif metric == 'average':
        return _pdist_func_1D(fX, lambda x, X: .5 * (x + X))

    else:
        return scipy.spatial.distance.pdist(fX, metric=metric, **kwargs)


def _cdist_func_1D(X_trn, X_tst, func):
    """Helper function for cdist"""
    X_trn = X_trn.squeeze()
    X_tst = X_tst.squeeze()
    return np.vstack(func(x_trn, X_tst) for x_trn in iter(X_trn))


def cdist(fX_trn, fX_tst, metric='euclidean', **kwargs):
    """Same as scipy.spatial.distance.cdist with support for additional metrics

    * 'angular': pairwise angular distance
    * 'equal':   pairwise equality check (only for 1-dimensional fX)
    * 'minimum': pairwise minimum (only for 1-dimensional fX)
    * 'maximum': pairwise maximum (only for 1-dimensional fX)
    * 'average': pairwise average (only for 1-dimensional fX)
    """

    if metric == 'angular':
        cosine = scipy.spatial.distance.cdist(
            fX_trn, fX_tst, metric='cosine', **kwargs)
        return np.arccos(np.clip(1.0 - cosine, -1.0, 1.0))

    elif metric == 'equal':
        return _cdist_func_1D(fX_trn, fX_tst,
                              lambda x_trn, X_tst: x_trn == X_tst)

    elif metric == 'minimum':
        return _cdist_func_1D(fX_trn, fX_tst, np.minimum)

    elif metric == 'maximum':
        return _cdist_func_1D(fX_trn, fX_tst, np.maximum)

    elif metric == 'average':
        return _cdist_func_1D(fX_trn, fX_tst,
                              lambda x_trn, X_tst: .5 * (x_trn + X_tst))

    else:
        return scipy.spatial.distance.cdist(
            fX_trn, fX_tst, metric=metric, **kwargs)


def to_condensed(n, i, j):
    """Compute index in condensed pdist matrix

                V
        0 | . 0 1 2 3
     -> 1 | . . 4 5 6 <-   ==>   0 1 2 3 4 5 6 7 8 9
        2 | . . . 7 8                    ^
        3 | . . . . 9
        4 | . . . . .
           ----------
            0 1 2 3 4

    Parameters
    ----------
    n : int
        Number of inputs in squared pdist matrix
    i, j : `int` or `numpy.ndarray`
        Indices in squared pdist matrix

    Returns
    -------
    k : `int` or `numpy.ndarray`
        Index in condensed pdist matrix
    """
    i, j = np.array(i), np.array(j)
    if np.any(i == j):
        raise ValueError('i and j should be different.')
    i, j = np.minimum(i, j), np.maximum(i, j)
    return np.int64(i * n - i * i / 2 - 3 * i / 2 + j - 1)


def to_squared(n, k):
    """Compute indices in squared matrix

    Parameters
    ----------
    n : int
        Number of inputs in squared pdist matrix
    k : `int` or `numpy.ndarray`
        Index in condensed pdist matrix

    Returns
    -------
    i, j : `int` or `numpy.ndarray`
        Indices in squared pdist matrix

    """
    k = np.array(k)
    i = np.int64(n - np.sqrt(-8*k + 4*n**2 - 4*n + 1)/2 - 1/2)
    j = np.int64(i**2/2 - i*n + 3*i/2 + k + 1)
    return i, j


# for n in range(2, 10):
#     for k in range(int(n*(n-1)/2)):
#         assert to_condensed(n, *to_squared(n, k)) == k

def linkage(X, method='single', metric='euclidean'):
    """Same as scipy.cluster.hierarchy.linkage with more metrics and methods
    """

    if method == 'pool':
        return pool(X, metric=metric, pooling_func=None)

    # corner case when using non-euclidean distances with methods
    # designed for the euclidean distance
    if metric != 'euclidean' and method in ['centroid', 'median', 'ward']:
        # Those 3 methods only work with 'euclidean' distance.
        # Therefore, one has to unit-normalized embeddings before
        # comparison in case they were optimized for 'cosine' (or 'angular')
        # distance.
        X = l2_normalize(X)
        metric = 'euclidean'

    distance = pdist(X, metric=metric)
    return scipy.cluster.hierarchy.linkage(distance, method=method,
                                           metric=metric)

def pool(X, metric='euclidean', pooling_func=None):
    """'pool' linkage"""

    if pooling_func is None:
        def pooling_func(C_u, C_v, X_u, X_v):
            S_u = len(X_u)
            S_v = len(X_v)
            return (C_u * S_u + C_v * S_v) / (S_u + S_v)

    # obtain number of original observations
    n, dimension = X.shape

    # K[j] contains the index of the cluster to which
    # the jth observation is currently assigned
    K = np.arange(n)

    # S[k] contains the current size of kth cluster
    S = np.zeros(2 * n - 1, dtype=np.int16)
    S[:n] = 1

    # C[k] contains the centroid of kth cluster
    C = np.zeros((2 * n - 1, dimension))
    # at the beginning, each observation is assigned to its own cluster
    C[:n, :] = X

    # clustering tree (aka dendrogram)
    # Z[i, 0] and Z[i, 1] are merged at ith iteration
    # Z[i, 2] is the distance between Z[i, 0] and Z[i, 1]
    # Z[i, 3] is the total number of original observation in the newly formed cluster
    Z = np.zeros((n - 1, 4))

    # convert condensed pdist matrix for the `n` original observation to a
    # condensed pdist matrix for the `2n-1` clusters (including the `n`
    # original observations) that will exist at some point during the process.
    D = np.infty * np.ones((2 * n - 1) * (2 * n - 2) // 2)
    D[to_condensed(2 * n - 1, *to_squared(n, np.arange(n * (n - 1) // 2)))] = \
        pdist(X, metric=metric)

    for i in range(n - 1):

        # find two most similar clusters
        k = np.argmin(D)
        u, v = to_squared(2 * n - 1, k)

        # keep track of ...
        # ... which clusters are merged at this iteration
        Z[i, 0] = v if S[v] > S[u] else v
        Z[i, 1] = u if Z[i, 0] == v else v

        # ... their distance
        Z[i, 2] = D[k]

        # ... the size of the newly formed cluster
        Z[i, 3] = S[u] + S[v]
        S[n + i] = S[u] + S[v]

        # merged clusters are now empty...
        S[u] = 0
        S[v] = 0

        # compute centroid of newly formed cluster
        C[n + i] = pooling_func(C[u], C[v], X[K == u], X[K == v])

        # move observations of merged clusters into the newly formed cluster
        K[K == u] = n + i
        K[K == v] = n + i

        # distance to merged clusters u and v can no longer be computed
        D[to_condensed(2 * n - 1, u, np.arange(u))] = np.infty
        D[to_condensed(2 * n - 1, u, np.arange(u + 1, n + i + 1))] = np.infty
        D[to_condensed(2 * n - 1, v, np.arange(v))] = np.infty
        D[to_condensed(2 * n - 1, v, np.arange(v + 1, n + i + 1))] = np.infty

        # compute distance to newly formed cluster
        empty = S[:n + i] == 0
        k = to_condensed(2 * n - 1, n + i, np.arange(n + i)[~empty])
        D[k] = cdist(C[np.newaxis, n + i, :],
                     C[:n + i, :][~empty, :],
                     metric=metric)

        # is this really needed?
        k = to_condensed(2 * n - 1, n + i, np.arange(n + i)[empty])
        D[k] = np.infty

    return Z
