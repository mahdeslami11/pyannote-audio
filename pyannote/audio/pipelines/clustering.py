# The MIT License (MIT)
#
# Copyright (c) 2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Clustering pipelines"""

from enum import Enum

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
from spectralcluster import EigenGapType, LaplacianType, SpectralClusterer

from pyannote.core.utils.distance import pdist
from pyannote.core.utils.hierarchy import linkage
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, Uniform


class ConstrainedAgglomerativeClustering(Pipeline):
    """Constrained  agglomerative clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".
    expects_num_clusters : bool, optional
        Whether the number of clusters should be provided.
        Defaults to False.

    Hyper-parameters
    ----------------
    threshold : float in range [0.0, 2.0]
    centroid_threshold : float in range [0.0, 2.0]

    Notes
    -----
    Embeddings are expected to be unit-normalized

    """

    def __init__(self, metric: str = "cosine", expects_num_clusters: bool = False):
        super().__init__()

        self.metric = metric
        if expects_num_clusters:
            raise ValueError()

        self.threshold = Uniform(0.0, 2.0)
        self.centroid_threshold = Uniform(0.0, 2.0)

    def __call__(
        self,
        embeddings: np.ndarray,
        num_clusters: int = None,  # not (yet) supported
        min_clusters: int = None,  # not (yet) supported
        max_clusters: int = None,  # not (yet) supported
        constraints: np.ndarray = None,
    ) -> np.ndarray:

        num_embeddings, _ = embeddings.shape

        min_clusters = max(1, num_clusters or min_clusters or 1)
        max_clusters = min(
            num_embeddings, num_clusters or max_clusters or num_embeddings
        )

        if constraints is None:
            constraints = np.zeros((num_embeddings, num_embeddings))

        # step 1: cluster embeddings with constraints
        dendrogram: np.ndarray = linkage(
            embeddings,
            method="pool",
            metric=self.metric,
            cannot_link=list(zip(*np.where(constraints < 0.0))),
            must_link=list(zip(*np.where(constraints > 0.0))),
        )

        threshold = self.threshold
        clusters = fcluster(dendrogram, threshold, criterion="distance") - 1

        num_centroids = np.max(clusters) + 1
        if num_centroids == 1:
            return clusters

        # step 2: cluster centroids without constraints
        centroids = np.vstack(
            [np.mean(embeddings[clusters == k], axis=0) for k in range(num_centroids)]
        )
        centroid_dendrogram = linkage(centroids, method="average", metric=self.metric)
        centroid_threshold = self.centroid_threshold
        centroid_clusters = (
            fcluster(centroid_dendrogram, centroid_threshold, criterion="distance") - 1
        )

        return centroid_clusters[clusters]


class AgglomerativeClustering(Pipeline):
    """Agglomerative clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".
    expects_num_clusters : bool, optional
        Whether the number of clusters should be provided.
        Defaults to False.

    Hyper-parameters
    ----------------
    method : {"average", "centroid", "complete", "median", "pool", "single", "ward"}
        Linkage method.
    threshold : float in range [0.0, 2.0]
        Clustering threshold. Only when `expects_num_clusters` is False.

    Notes
    -----
    Embeddings are expected to be unit-normalized.
    """

    def __init__(self, metric: str = "cosine", expects_num_clusters: bool = False):
        super().__init__()

        self.metric = metric

        self.expects_num_clusters = expects_num_clusters
        if not self.expects_num_clusters:
            self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings

        self.method = Categorical(
            ["average", "centroid", "complete", "median", "pool", "single", "ward"]
        )

    def __call__(
        self,
        embeddings: np.ndarray,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
        constraints: np.ndarray = None,  # not supported
    ) -> np.ndarray:
        """Apply agglomerative clustering

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) np.ndarray
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Defaults to 1.
            Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Defaults to `num_embeddings`.
            Has no effect when `num_clusters` is provided.

        Returns
        -------
        clusters : (num_embeddings, ) np.ndarray
        """

        num_embeddings, _ = embeddings.shape

        if self.expects_num_clusters:
            assert num_clusters is not None, "`num_clusters` must be provided."
            num_clusters = min_clusters = max_clusters = min(
                num_embeddings, max(1, num_clusters)
            )
        else:
            min_clusters = max(1, num_clusters or min_clusters or 1)
            max_clusters = min(
                num_embeddings, num_clusters or max_clusters or num_embeddings
            )

        dendrogram: np.ndarray = linkage(
            embeddings, method=self.method, metric=self.metric
        )

        max_threshold: float = dendrogram[-min_clusters, 2]
        min_threshold: float = (
            dendrogram[-max_clusters, 2] if max_clusters < num_embeddings else -np.inf
        )

        if num_clusters is None:
            threshold = min(max(self.threshold, min_threshold), max_threshold)

        else:
            threshold = (
                dendrogram[-num_clusters, 2]
                if num_clusters < num_embeddings
                else -np.inf
            )

        return fcluster(dendrogram, threshold, criterion="distance") - 1


class SpectralClustering(Pipeline):
    """Spectral clustering

    Parameters
    ----------
    metric : {"cosine", "euclidean", ...}, optional
        Distance metric to use. Defaults to "cosine".
    expects_num_clusters : bool, optional
        Whether the number of clusters should be provided.
        Defaults to False.

    Hyper-parameters
    ----------------
    laplacian : {"Affinity", "Unnormalized", "RandomWalk", "GraphCut"}
        Laplacian to use.
    eigengap : {"Ratio", "NormalizedDiff"}
        Eigengap approach to use.

    Notes
    -----
    Embeddings are expected to be unit-normalized.
    """

    def __init__(self, metric: str = "cosine", expects_num_clusters: bool = False):
        super().__init__()
        self.metric = metric
        self.expects_num_clusters = expects_num_clusters

        self.laplacian = Categorical(
            ["Affinity", "Unnormalized", "RandomWalk", "GraphCut"]
        )
        self.eigengap = Categorical(["Ratio", "NormalizedDiff"])

    def _affinity_function(self, embeddings: np.ndarray) -> np.ndarray:
        return squareform(1.0 - 0.5 * pdist(embeddings, metric=self.metric))

    def __call__(
        self,
        embeddings: np.ndarray,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
        constraints: np.ndarray = None,  # not (yet) supported
    ) -> np.ndarray:
        """Apply spectral clustering

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) np.ndarray
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Defaults to 1.
            Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Defaults to `num_embeddings`.
            Has no effect when `num_clusters` is provided.

        Returns
        -------
        clusters : (num_embeddings, ) np.ndarray
        """

        num_embeddings, _ = embeddings.shape

        if self.expects_num_clusters:
            assert num_clusters is not None, "`num_clusters` must be provided."
            num_clusters = min_clusters = max_clusters = min(
                num_embeddings, max(1, num_clusters)
            )
        else:
            min_clusters = max(1, num_clusters or min_clusters or 1)
            max_clusters = min(
                num_embeddings, num_clusters or max_clusters or num_embeddings
            )

        return SpectralClusterer(
            min_clusters=num_clusters,
            max_clusters=num_clusters,
            laplacian_type=LaplacianType[self.laplacian],
            eigengap_type=EigenGapType[self.eigengap],
            affinity_function=self._affinity_function,
        ).predict(embeddings)


class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    ConstrainedAgglomerativeClustering = ConstrainedAgglomerativeClustering
    SpectralClustering = SpectralClustering
