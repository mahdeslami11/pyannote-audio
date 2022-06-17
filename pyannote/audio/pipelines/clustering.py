# The MIT License (MIT)
#
# Copyright (c) 2021- CNRS
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


import math
from enum import Enum
from typing import Tuple

import numpy as np
from einops import rearrange
from hmmlearn.hmm import GaussianHMM
from pyannote.core import SlidingWindowFeature
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, Uniform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from spectralcluster import (
    AutoTune,
    EigenGapType,
    FallbackClustererType,
    FallbackOptions,
    LaplacianType,
    RefinementName,
    RefinementOptions,
    SingleClusterCondition,
    SpectralClusterer,
    SymmetrizeType,
    ThresholdType,
)

from pyannote.audio import Inference


class ClusteringMixin:
    def set_num_clusters(
        self,
        num_embeddings: int,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
    ):

        min_clusters = num_clusters or min_clusters or 1
        min_clusters = max(1, min(num_embeddings, min_clusters))
        max_clusters = num_clusters or max_clusters or num_embeddings
        max_clusters = max(1, min(num_embeddings, max_clusters))

        if min_clusters > max_clusters:
            raise ValueError(
                f"min_clusters must be smaller than (or equal to) max_clusters "
                f"(here: min_clusters={min_clusters:g} and max_clusters={max_clusters:g})."
            )

        if min_clusters == max_clusters:
            num_clusters = min_clusters

        if self.expects_num_clusters and num_clusters is None:
            raise ValueError("num_clusters must be provided.")

        return num_clusters, min_clusters, max_clusters

    def filter_embeddings(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature = None,
        target_overlap_ratio: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter NaN embeddings and downsample embeddings

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        target_overlap_ratio : float, optional
            Defaults to 0.5

        Returns
        -------
        filtered_embeddings : (num_embeddings, dimension) array
        chunk_idx : (num_embeddings, ) array
        speaker_idx : (num_embeddings, ) array
        """

        downsample_by = math.floor(
            target_overlap_ratio
            * segmentations.sliding_window.duration
            / segmentations.sliding_window.step
        )
        chunk_idx, speaker_idx = np.where(
            ~np.any(np.isnan(embeddings[::downsample_by]), axis=2)
        )
        chunk_idx *= downsample_by

        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx


class AgglomerativeClustering(ClusteringMixin, Pipeline):
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
    method : {"average", "centroid", "complete", "median", "single", "ward"}
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
            ["average", "centroid", "complete", "median", "single", "ward", "weighted"]
        )

    @staticmethod
    def cluster(
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = None,
        threshold: float = 1.0,
        method="average",
        metric="cosine",
        **kwargs,
    ):
        """

        Parameters
        ----------
        embeddings : (num_embeddings, dimension) array
            Embeddings
        min_clusters : int
            Minimum number of clusters
        max_clusters : int
            Maximum number of clusters
        num_clusters : int, optional
            Actual number of clusters. Default behavior is to estimate it based
            on values provided for `min_clusters`,  `max_clusters`, and `threshold`.
        threshold : float, optional
            Clustering threshold. Not used when `num_clusters` is provided.
        method : {"average", "centroid", "complete", "median", "single", "ward"}, optional
            Linkage method.
        metric : {"cosine", "euclidean", ...}, optional
            Distance metric to use. Defaults to "cosine".

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """

        num_embeddings, _ = embeddings.shape

        if metric == "cosine" and method in ["centroid", "median", "ward"]:
            # unit-normalize embeddings to somehow make them "euclidean"
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            dendrogram: np.ndarray = linkage(
                embeddings, method=method, metric="euclidean"
            )

        else:
            dendrogram: np.ndarray = linkage(embeddings, method=method, metric=metric)

        if num_clusters is None:

            max_threshold: float = (
                dendrogram[-min_clusters, 2]
                if min_clusters < num_embeddings
                else -np.inf
            )
            min_threshold: float = (
                dendrogram[-max_clusters, 2]
                if max_clusters < num_embeddings
                else -np.inf
            )

            threshold = min(max(threshold, min_threshold), max_threshold)

        else:

            threshold = (
                dendrogram[-num_clusters, 2]
                if num_clusters < num_embeddings
                else -np.inf
            )

        return fcluster(dendrogram, threshold, criterion="distance") - 1

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature = None,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
    ) -> np.ndarray:
        """Apply agglomerative clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Has no effect when `num_clusters` is provided.

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        """

        num_chunks, num_speakers, dimension = embeddings.shape

        valid_embeddings, chunk_idx, speaker_idx = self.filter_embeddings(
            embeddings, segmentations=segmentations
        )

        num_embeddings, _ = valid_embeddings.shape
        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        valid_clusters = self.cluster(
            valid_embeddings,
            min_clusters,
            max_clusters,
            num_clusters=num_clusters,
            threshold=None if self.expects_num_clusters else self.threshold,
            method=self.method,
            metric=self.metric,
        )

        num_clusters = np.max(valid_clusters) + 1

        centroids = np.vstack(
            [
                np.mean(valid_embeddings[valid_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        soft_clusters = -e2k_distance
        hard_clusters = np.argmax(soft_clusters, axis=2)
        hard_clusters[chunk_idx, speaker_idx] = valid_clusters

        return hard_clusters, soft_clusters


class SpectralClustering(ClusteringMixin, Pipeline):
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
    spectral_min_embeddings : int
        Fallback to agglomerative clustering when clustering less than that
        many embeddings.
    refinement_sequence : str
        Sequence of refinement operations (e.g. "CGTSDN").
        Each character represents one operation ("C" for CropDiagonal, "G" for GaussianBlur,
        "T" for RowWiseThreshold, "S" for Symmetrize, "D" for Diffuse, and "N" for RowWiseNormalize.
        Use empty string to not use any refinement.
    gaussian_blur_sigma : float
        Sigma value for the Gaussian blur operation
    symmetrize_type : {"Max", "Average"}
        How to symmetrize the matrix
    thresholding_with_binarization : boolean
        Set values larger than the threshold to 1.
    thresholding_preserve_diagonal : boolean
        In the row wise thresholding operation, set diagonals of the
        affinity matrix to 0 at the beginning, and back to 1 in the end
    thresholding_type : {"RowMax", "Percentile"}
        Type of thresholding operation.

    Notes
    -----
    Embeddings are expected to be unit-normalized.
    """

    def __init__(self, metric: str = "cosine", expects_num_clusters: bool = False):
        super().__init__()
        self.metric = metric
        self.expects_num_clusters = expects_num_clusters
        # TODO: reduce hyper-parameter search space when expects_num_clusters is True

        self.laplacian = Categorical(
            ["Affinity", "Unnormalized", "RandomWalk", "GraphCut"]
        )
        self.eigengap = Categorical(["Ratio", "NormalizedDiff"])

        # Hyperparameters for refinement operations.
        self.refinement_sequence = Categorical(["", "TS", "GTS", "CGTSDN"])
        self.gaussian_blur_sigma = Uniform(0, 3)
        self.symmetrize_type = Categorical(["Max", "Average"])
        self.thresholding_with_binarization = Categorical([True, False])
        self.thresholding_preserve_diagonal = Categorical([True, False])
        self.thresholding_type = Categorical(["RowMax", "Percentile"])

        # HACK https://github.com/wq2012/SpectralCluster/issues/39

        if not self.expects_num_clusters:
            # proportion of chunks with 1+ speaker that have 2+ speakers
            # according to the local segmentation model
            # (should be high for one-speaker audio but can also be high
            # for very audio with one dominant speaker)
            self.solo_speaker_ratio_threshold = Uniform(0.0, 1.0)
            self.solo_speaker_hac_threshold = Uniform(0.0, 1.0)

    def _affinity_function(self, embeddings: np.ndarray) -> np.ndarray:
        return squareform(1.0 - 0.5 * pdist(embeddings, metric=self.metric))

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature = None,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
    ) -> np.ndarray:
        """Apply spectral clustering

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Has no effect when `num_clusters` is provided.

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        """

        num_chunks, num_speakers, dimension = embeddings.shape

        valid_embeddings, chunk_idx, speaker_idx = self.filter_embeddings(
            embeddings, segmentations=segmentations
        )

        num_embeddings, _ = valid_embeddings.shape

        # may happen because of downsampling
        if num_embeddings < 2:
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.zeros((num_chunks, num_speakers, 1))
            return hard_clusters, soft_clusters

        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        # Fallback options.
        fallback_options = FallbackOptions(
            # TODO: remove SingleClusterCondition completely
            # HACK: deactivate single_cluster_condition by using AffinityGmmBic
            # but keeping the (all one) affinity diagonal
            single_cluster_condition=SingleClusterCondition.AffinityGmmBic,
            single_cluster_affinity_diagonal_offset=0,
            single_cluster_affinity_threshold=0.75,  # not used with AffinityGmmBic
            # HACK: hardcode spectral_min_embeddings to 5 to (almost) never
            # use the fallback clusterer.
            spectral_min_embeddings=5,
            fallback_clusterer_type=FallbackClustererType.Naive,
            naive_threshold=0.5,  # not used unless num_embeddings < 5
            naive_adaptation_threshold=None,  # not used unless num_embeddings < 5
        )

        # Autotune options.
        autotune = AutoTune(
            p_percentile_min=0.40,
            p_percentile_max=0.95,
            init_search_step=0.05,
            search_level=1,
        )

        # Sequence of refinement operations.
        refinement_sequence = []
        for refinement_char in self.refinement_sequence:
            refinement_char = refinement_char.upper()
            if refinement_char == "C":
                refinement_sequence.append(RefinementName.CropDiagonal)
            elif refinement_char == "G":
                refinement_sequence.append(RefinementName.GaussianBlur)
            elif refinement_char == "T":
                refinement_sequence.append(RefinementName.RowWiseThreshold)
            elif refinement_char == "S":
                refinement_sequence.append(RefinementName.Symmetrize)
            elif refinement_char == "D":
                refinement_sequence.append(RefinementName.Diffuse)
            elif refinement_char == "N":
                refinement_sequence.append(RefinementName.RowWiseNormalize)
            else:
                raise ValueError("Unsupported refinement: " + refinement_char)

        # Refinement options.
        refinement_options = RefinementOptions(
            gaussian_blur_sigma=self.gaussian_blur_sigma,
            thresholding_soft_multiplier=0.01,
            thresholding_type=ThresholdType[self.thresholding_type],
            thresholding_with_binarization=self.thresholding_with_binarization,
            thresholding_preserve_diagonal=self.thresholding_preserve_diagonal,
            symmetrize_type=SymmetrizeType[self.symmetrize_type],
            refinement_sequence=refinement_sequence,
        )

        valid_clusters = SpectralClusterer(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            refinement_options=refinement_options,
            autotune=autotune,
            fallback_options=fallback_options,
            laplacian_type=LaplacianType[self.laplacian],
            eigengap_type=EigenGapType[self.eigengap],
            affinity_function=self._affinity_function,
        ).predict(valid_embeddings)

        num_clusters = np.max(valid_clusters) + 1

        centroids = np.vstack(
            [
                np.mean(valid_embeddings[valid_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        # post-process spectral clustering (SC) output to handle the known
        # limitation that SC is not very good at handling one-cluster data
        # SEE https://github.com/wq2012/SpectralCluster/issues/39

        # compute the ratio of chunks with 1+ active speaker
        # in which there are actually 2+ active  speakers.
        # if this ratio is high enough, it might be a cue that
        # there is only one cluster. therefore we postprocess the
        # output of spectral clustering by applying an additional
        # hierarchical agglomerative clustering step on the centroids

        num_active_speaker = np.sum(np.any(segmentations.data > 0, axis=1), axis=1)
        solo_speaker_ratio = np.sum(num_active_speaker == 1) / np.sum(
            num_active_speaker > 0
        )

        if num_clusters > 1 and solo_speaker_ratio > self.solo_speaker_ratio_threshold:

            centroids_cluster = AgglomerativeClustering.cluster(
                centroids,
                min_clusters,
                max_clusters,
                num_clusters=num_clusters,
                threshold=self.solo_speaker_hac_threshold,
                method="average",
                metric=self.metric,
            )

            valid_clusters = centroids_cluster[valid_clusters]
            num_clusters = np.max(valid_clusters) + 1
            centroids = np.vstack(
                [
                    np.mean(valid_embeddings[valid_clusters == k], axis=0)
                    for k in range(num_clusters)
                ]
            )

        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        soft_clusters = -e2k_distance
        hard_clusters = np.argmax(soft_clusters, axis=2)
        hard_clusters[chunk_idx, speaker_idx] = valid_clusters

        return hard_clusters, soft_clusters


class GaussianHiddenMarkovModel(ClusteringMixin, Pipeline):
    """Hidden Markov Model with Gaussian states

    Parameters
    ----------
    n_trials :

    """

    def __init__(
        self,
        metric: str = "cosine",
        expects_num_clusters: bool = False,
        n_trials: int = 5,
    ):
        super().__init__()

        if metric not in ["euclidean", "cosine"]:
            raise ValueError("`metric` must be one of {'cosine', 'euclidean'}")

        self.metric = metric
        self.expects_num_clusters = expects_num_clusters
        self.n_trials = n_trials

        self.covariance_type = Categorical(["spherical", "diag", "full", "tied"])
        self.threshold = Uniform(0.0, 2.0)

    def get_training_sequence(
        self, embeddings: np.ndarray, segmentations: SlidingWindowFeature
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.

        Returns
        -------
        training_sequence : (num_steps, dimension) array
        chunk_idx : (num_steps, ) array
        speaker_idx : (num_steps, ) array

        """

        num_chunks, _, _ = embeddings.shape

        # focus on center of each chunk
        duration = segmentations.sliding_window.duration
        step = segmentations.sliding_window.step

        ratio = 0.5 * (duration - step) / duration
        center_segmentations = Inference.trim(segmentations, warm_up=(ratio, ratio))
        #   shape: num_chunks, num_center_frames, num_speakers

        # number of frames during which speakers are active
        # in the center of the chunk
        num_active_frames: np.ndarray = np.sum(center_segmentations.data, axis=1)
        #   shape: (num_chunks, num_speakers)

        priors = num_active_frames / (
            np.sum(num_active_frames, axis=1, keepdims=True) + 1e-8
        )
        #   shape: (num_chunks, local_num_speakers)

        speaker_idx = np.argmax(priors, axis=1)
        # (num_chunks, )

        training_sequence = embeddings[range(num_chunks), speaker_idx]
        # (num_chunks, dimension)

        chunk_idx = np.where(~np.any(np.isnan(training_sequence), axis=1))[0]
        # (num_chunks, )

        return (training_sequence[chunk_idx], chunk_idx, speaker_idx[chunk_idx])

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature = None,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
    ) -> np.ndarray:
        """

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension) array
            Sequence of embeddings.
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        num_clusters : int, optional
            Number of clusters, when known. Default behavior is to use
            internal threshold hyper-parameter to decide on the number
            of clusters.
        min_clusters : int, optional
            Minimum number of clusters. Has no effect when `num_clusters` is provided.
        max_clusters : int, optional
            Maximum number of clusters. Has no effect when `num_clusters` is provided.

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        """

        num_chunks, num_speakers, dimension = embeddings.shape

        if self.metric == "cosine":
            # unit-normalize embeddings to somehow make them "euclidean"
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

        training_sequence, chunk_idx, speaker_idx = self.get_training_sequence(
            embeddings,
            segmentations,
        )
        num_embeddings, _ = training_sequence.shape

        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        # FIXME
        if max_clusters == num_embeddings:
            max_clusters = 20

        # TODO: try to infer max_clusters automatically by looking at the evolution of selection criterion

        # estimate num_clusters by fitting an HMM with an increasing number of states
        debug = {"training_sequence": training_sequence, "hmm": dict()}
        if num_clusters is None:

            num_clusters = max_clusters
            for n_components in range(min_clusters, max_clusters + 1):

                hmm = GaussianHMM(
                    n_components=n_components,
                    covariance_type=self.covariance_type,
                    n_iter=100,
                    # random_state=random_state,
                    implementation="log",
                ).fit(training_sequence)

                debug["hmm"][n_components] = hmm

                if n_components > 1:
                    # as soon as two states get too close to each other, stop adding states
                    min_state_dist = np.min(pdist(hmm.means_, metric="euclidean"))

                    if min_state_dist < self.threshold:
                        num_clusters = max(min_clusters, n_components - 1)
                        break

        if num_clusters == 1:
            return np.zeros((num_chunks, num_speakers), dtype=np.int), np.zeros(
                (num_chunks, num_speakers, 1)
            )

        # once num_clusters is estimated, fit the HMM several times
        # and keep the one that best fits the data
        best_log_likelihood = -np.inf
        for random_state in range(self.n_trials):
            hmm = GaussianHMM(
                n_components=num_clusters,
                covariance_type=self.covariance_type,
                n_iter=100,
                random_state=random_state,
                implementation="log",
            )
            hmm.fit(training_sequence)

            try:
                log_likelihood = hmm.score(training_sequence)
            except ValueError:
                log_likelihood = -np.inf

            if log_likelihood >= best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_hmm = hmm

        debug["best_hmm"] = best_hmm
        self.debug_ = debug

        # compute distance between embeddings and clusters
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                best_hmm.means_,
                metric="cosine",
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        soft_clusters = 2 - e2k_distance

        # NOTE: using decoding instead of distance-to-state actually degraded performance
        # hard_clusters[chunk_idx, speaker_idx] = best_hmm.predict(training_sequence)

        hard_clusters = np.argmax(soft_clusters, axis=2)

        return hard_clusters, soft_clusters


class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    SpectralClustering = SpectralClustering
    GaussianHiddenMarkovModel = GaussianHiddenMarkovModel
