#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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

import itertools
import numpy as np
from sortedcollections import ValueSortedDict
from pyannote.algorithms.clustering.hac import \
    HierarchicalAgglomerativeClustering
from pyannote.algorithms.clustering.hac.model import HACModel
from pyannote.algorithms.clustering.hac.stop import DistanceThreshold
from scipy.spatial.distance import squareform
from pyannote.audio.embedding.utils import pdist, cdist, l2_normalize


class EmbeddingModel(HACModel):
    """

    Parameters
    ----------
    distance : str
        Defaults to 'angular'.
    mode : {'loose', 'strict'}

    """

    def __init__(self, distance='angular', mode='strict'):
        super(EmbeddingModel, self).__init__(is_symmetric=True)
        self.distance = distance
        self.mode = mode

    def compute_model(self, cluster, parent=None):

        # extract all embeddings for requested cluster
        support = parent.current_state.label_timeline(cluster).support()
        X = parent.features.crop(support, mode=self.mode)

        # average them all
        x = np.average(X, axis=0)
        n = len(X)

        return (x, n)

    def compute_merged_model(self, clusters, parent=None):

        # merge all embeddings by computing their weighted average
        X, N = zip(*[self[cluster] for cluster in clusters])
        x = np.average(X, axis=0, weights=N)
        n = np.sum(N)

        return (x, n)

    def compute_similarity_matrix(self, parent=None):

        clusters = list(self._models)
        n_clusters = len(clusters)

        X = np.vstack([self[cluster][0] for cluster in clusters])

        nX = l2_normalize(X)
        similarities = -squareform(pdist(nX, metric=self.distance))

        matrix = ValueSortedDict()
        for i, j in itertools.combinations(range(n_clusters), 2):
            matrix[clusters[i], clusters[j]] = similarities[i, j]
            matrix[clusters[j], clusters[i]] = similarities[j, i]

        return matrix

    def compute_similarities(self, cluster, clusters, parent=None):

        x = self[cluster][0].reshape((1, -1))
        X = np.vstack([self[c][0] for c in clusters])

        # L2 normalization
        nx = l2_normalize(x)
        nX = l2_normalize(X)

        similarities = -cdist(nx, nX, metric=self.distance)

        matrix = ValueSortedDict()
        for i, cluster_ in enumerate(clusters):
            matrix[cluster, cluster_] = similarities[0, i]
            matrix[cluster_, cluster] = similarities[0, i]

        return matrix

    def compute_similarity(self, cluster1, cluster2, parent=None):

        x1, _ = self[cluster1]
        x2, _ = self[cluster2]

        nx1 = l2_normalize(x1)
        nx2 = l2_normalize(x2)

        similarities = -cdist([nx1], [nx2], metric=self.distance)
        return similarities[0, 0]

# from pyannote.audio.features import Precomputed
# feature_extraction = Precomputed(
#     '/Users/bredin/Development/pyannote/pyannote-audio/experiments/precomputed/')
# from pyannote.audio.embedding.clustering import EmbeddingClustering
# clustering = EmbeddingClustering(force=True, distance='cosine', mode='strict')
#
# from pyannote.database import get_protocol
# protocol = get_protocol('Etape.SpeakerDiarization.Debug')
# current_file = next(protocol.train())
#
# starting_point = current_file['annotation'].anonymize_tracks()
# features = feature_extraction(current_file)
# result = clustering(starting_point, features=features)
#
# from pyannote.metrics.diarization import DiarizationPurity, DiarizationCoverage, GreedyDiarizationErrorRate
# Purity, Coverage, Error = DiarizationPurity(parallel=False), DiarizationCoverage(parallel=False), GreedyDiarizationErrorRate(parallel=False)
#
# reference, uem = current_file['annotation'], current_file['annotated']
# for i, hypothesis in enumerate(clustering.history):
#     threshold = -clustering.history.iterations[i].similarity
#     if abs(threshold - 0.2) > 0.1:
#         continue
#     purity = Purity(reference, hypothesis, uem=uem)
#     coverage = Coverage(reference, hypothesis, uem=uem)
#     error = Error(reference, hypothesis, uem=uem)
#     print(f'{i:04d} {threshold:.4f} {100*purity:.2f} {100*coverage:.2f} {100*error:.2f}')

class EmbeddingClustering(HierarchicalAgglomerativeClustering):
    """Audio sequence clustering based on embeddings

    Parameters
    ----------
    distance : str, optional
        Defaults to 'angular'.
    threshold : float, optional
        Defaults to 1.0.

    Usage
    -----
    >>> embedding = Precomputed(...)
    >>> clustering = EmbeddingClustering()
    >>> result = clustering(starting_point, features=embedding)

    """

    def __init__(self, threshold=1.0, force=False, distance='cosine',
                 mode='loose', constraint=None, logger=None):
        model = EmbeddingModel(distance=distance, mode=mode)
        stopping_criterion = DistanceThreshold(threshold=threshold,
                                               force=force)
        super(EmbeddingClustering, self).__init__(
            model,
            stopping_criterion=stopping_criterion,
            constraint=constraint,
            logger=logger)


class Clustering(object):
    def __init__(self, min_cluster_size=5, min_samples=None,
                 metric='euclidean'):
        super(Clustering, self).__init__()

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

    def apply(self, fX):
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size,
                            min_samples=self.min_samples,
                            metric='precomputed')
        distance_matrix = squareform(pdist(fX, metric=self.metric))

        # apply clustering
        cluster_labels = clusterer.fit_predict(distance_matrix)
        
        # cluster embedding
        n_clusters = np.max(cluster_labels) + 1

        if n_clusters < 2:
            return np.zeros(fX.shape[0], dtype=np.int)
        
        fC = l2_normalize(
            np.vstack([np.sum(fX[cluster_labels == k, :], axis=0)
                       for k in range(n_clusters)]))

        # tag each undefined embedding to closest cluster
        undefined = cluster_labels == -1
        closest_cluster = np.argmin(
            cdist(fC, fX[undefined, :], metric=self.metric), axis=0)
        cluster_labels[undefined] = closest_cluster        

        return cluster_labels
