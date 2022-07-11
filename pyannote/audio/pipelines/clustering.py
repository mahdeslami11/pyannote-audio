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
import pyannote.core.utils.distance
from einops import rearrange
from hmmlearn.hmm import GaussianHMM
from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Categorical, ParamDict, Uniform
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.mixture import GaussianMixture
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
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import oracle_segmentation
from pyannote.audio.pipelines.utils.constraint import SegmentationConstraints
from pyannote.audio.utils.permutation import permutate


class BaseClustering(Pipeline):
    def __init__(
        self,
        metric: str = "cosine",
        expects_num_clusters: bool = False,
        constrained_assignment: bool = False,
    ):

        super().__init__()
        self.metric = metric
        self.expects_num_clusters = expects_num_clusters
        self.constrained_assignment = constrained_assignment

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

        if target_overlap_ratio is None:
            downsample_by = 1
        else:
            downsample_by = max(
                1,
                math.floor(
                    target_overlap_ratio
                    * segmentations.sliding_window.duration
                    / segmentations.sliding_window.step
                ),
            )

        chunk_idx, speaker_idx = np.where(
            ~np.any(np.isnan(embeddings[::downsample_by]), axis=2)
        )
        chunk_idx *= downsample_by

        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx

    def constrained_argmax(self, soft_clusters: np.ndarray) -> np.ndarray:

        soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))
        num_chunks, num_speakers, num_clusters = soft_clusters.shape
        # num_chunks, num_speakers, num_clusters

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

        for c, cost in enumerate(soft_clusters):
            speakers, clusters = linear_sum_assignment(cost, maximize=True)
            for s, k in zip(speakers, clusters):
                hard_clusters[c, s] = k

        return hard_clusters

    # def constrained_argmax(
    #     self, soft_clusters: np.ndarray, segmentations: SlidingWindowFeature
    # ) -> np.ndarray:
    #     """

    #     Parameters
    #     ----------
    #     soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
    #     segmentations : SlidingWindowFeature
    #         Binarized segmentation.

    #     Returns
    #     -------
    #     hard_clusters : (num_chunks, num_speakers)-shaped array
    #         Hard cluster assignment with

    #     """

    #     import cvxpy as cp

    #     num_chunks, num_speakers, num_clusters = soft_clusters.shape
    #     hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)

    #     for c, (scores, (chunk, segmentation)) in enumerate(
    #         zip(soft_clusters, segmentations)
    #     ):

    #         # scores : (num_speakers, num_clusters) array
    #         # segmentation : (num_frames, num_speakers) array

    #         assignment = cp.Variable(shape=(num_speakers, num_clusters), boolean=True)
    #         objective = cp.Maximize(cp.sum(cp.multiply(assignment, scores)))

    #         one_cluster_per_speaker_constraints = [
    #             cp.sum(assignment[i]) == 1 for i in range(num_speakers)
    #         ]

    #         # number of frames where both speakers are active
    #         co_occurrence: np.ndarray = segmentation.T @ segmentation
    #         np.fill_diagonal(co_occurrence, 0)
    #         cannot_link = set(
    #             tuple(sorted(x)) for x in zip(*np.where(co_occurrence > 0))
    #         )
    #         cannot_link_constraints = [
    #             assignment[i] + assignment[j] <= 1 for i, j in cannot_link
    #         ]

    #         problem = cp.Problem(
    #             objective, one_cluster_per_speaker_constraints + cannot_link_constraints
    #         )
    #         problem.solve()

    #         if problem.status == "optimal":
    #             hard_clusters[c] = np.argmax(assignment.value, axis=1)
    #         else:
    #             print(f"{co_occurrence=}")
    #             hard_clusters[c] = np.argmax(scores, axis=1)

    #     return hard_clusters

    def assign_embeddings(
        self,
        embeddings: np.ndarray,
        train_chunk_idx: np.ndarray,
        train_speaker_idx: np.ndarray,
        train_clusters: np.ndarray,
        constrained: bool = False,
    ):
        """Assign embeddings to the closest centroid

        Cluster centroids are computed as the average of the train embeddings
        previously assigned to them.

        Parameters
        ----------
        embeddings : (num_chunks, num_speakers, dimension)-shaped array
            Complete set of embeddings.
        train_chunk_idx : (num_embeddings,)-shaped array
        train_speaker_idx : (num_embeddings,)-shaped array
            Indices of subset of embeddings used for "training".
        train_clusters : (num_embedding,)-shaped array
            Clusters of the above subset
        constrained : bool, optional
            Use constrained_argmax, instead of (default) argmax.

        Returns
        -------
        soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
        hard_clusters : (num_chunks, num_speakers)-shaped array
        """

        num_clusters = np.max(train_clusters) + 1
        num_chunks, num_speakers, dimension = embeddings.shape

        train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]

        centroids = np.vstack(
            [
                np.mean(train_embeddings[train_clusters == k], axis=0)
                for k in range(num_clusters)
            ]
        )

        # compute distance between embeddings and clusters
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
        soft_clusters = 2 - e2k_distance

        # assign each embedding to the cluster with the most similar centroid
        if constrained:
            hard_clusters = self.constrained_argmax(soft_clusters)
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        # TODO: add a flag to revert argmax for training subset
        # hard_clusters[train_chunk_idx, train_speaker_idx] = train_clusters

        return hard_clusters, soft_clusters

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature = None,
        num_clusters: int = None,
        min_clusters: int = None,
        max_clusters: int = None,
        constraints: SegmentationConstraints = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply HMM clustering

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

        print(f"must = {np.sum(constraints > 0)}")
        print(f"cannot = {np.sum(constraints < 0)}")

        train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings(
            embeddings,
            segmentations=segmentations,
        )

        num_embeddings, _ = train_embeddings.shape
        num_clusters, min_clusters, max_clusters = self.set_num_clusters(
            num_embeddings,
            num_clusters=num_clusters,
            min_clusters=min_clusters,
            max_clusters=max_clusters,
        )

        if max_clusters < 2:
            train_clusters = np.zeros((num_embeddings,), dtype=np.int8)
        else:

            _, _, num_speakers = segmentations.data.shape
            idx = train_chunk_idx * num_speakers + train_speaker_idx
            constraints = squareform(squareform(constraints)[idx][:, idx])

            train_clusters = self.cluster(
                train_embeddings,
                min_clusters,
                max_clusters,
                num_clusters=num_clusters,
                constraints=constraints,
            )

        hard_clusters, soft_clusters = self.assign_embeddings(
            embeddings,
            train_chunk_idx,
            train_speaker_idx,
            train_clusters,
            constrained=self.constrained_assignment,
        )

        return hard_clusters, soft_clusters


class GaussianMixtureClustering(BaseClustering):
    def __init__(
        self,
        metric: str = "cosine",
        expects_num_clusters: bool = False,
        constrained_assignment: bool = False,
    ):

        if metric not in ["euclidean", "cosine"]:
            raise ValueError("`metric` must be one of {'cosine', 'euclidean'}")

        super().__init__(
            metric=metric,
            expects_num_clusters=expects_num_clusters,
            constrained_assignment=constrained_assignment,
        )

        self.covariance_type = Categorical(["spherical", "diag", "full", "tied"])
        if not self.expects_num_clusters:
            self.threshold = Uniform(0.0, 2.0)

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = None,
    ):

        # FIXME
        if max_clusters == len(embeddings):
            max_clusters = 20

        if self.metric == "cosine":
            # unit-normalize embeddings to somehow make them "euclidean"
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

        if num_clusters is None:

            num_clusters = max_clusters
            for n_components in range(min_clusters, max_clusters + 1):

                gmm = GaussianMixture(
                    n_components=n_components, covariance_type=self.covariance_type
                )
                gmm.fit(embeddings)

                bic = gmm.bic(embeddings)
                print(f"{n_components=} {bic=:.1f}")

                if n_components > 1:

                    # print(pdist(gmm.means_, metric="euclidean"))

                    # as soon as two states get too close to each other, stop adding states
                    min_state_dist = np.min(pdist(gmm.means_, metric="euclidean"))
                    print(f"{n_components=} {min_state_dist=}")
                    if min_state_dist < self.threshold:
                        num_clusters = max(min_clusters, n_components - 1)
                        break

        if num_clusters == 1:
            return np.zeros((len(embeddings),), dtype=np.int8)

        gmm = GaussianMixture(
            n_components=num_clusters, covariance_type=self.covariance_type
        )
        gmm.fit(embeddings)

        try:
            train_clusters = gmm.predict(embeddings)
        except ValueError:
            # ValueError: startprob_ must sum to 1 (got nan)
            train_clusters = np.zeros((len(embeddings),), dtype=np.int8)

        return train_clusters


class AgglomerativeClustering(BaseClustering):
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

    def __init__(
        self,
        metric: str = "cosine",
        expects_num_clusters: bool = False,
        constrained_assignment: bool = False,
    ):

        super().__init__(
            metric=metric,
            expects_num_clusters=expects_num_clusters,
            constrained_assignment=constrained_assignment,
        )

        if not self.expects_num_clusters:
            self.threshold = Uniform(0.0, 2.0)  # assume unit-normalized embeddings
        self.method = Categorical(
            ["average", "centroid", "complete", "median", "single", "ward", "weighted"]
        )

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = None,
        cannot_link: np.ndarray = None,
        must_link: np.ndarray = None,
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

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """

        num_embeddings, _ = embeddings.shape
        if num_embeddings == 1:
            return np.zeros((1,), dtype=np.uint8)

        if self.metric == "cosine" and self.method in ["centroid", "median", "ward"]:
            # unit-normalize embeddings to somehow make them "euclidean"
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric="euclidean"
            )

        else:
            dendrogram: np.ndarray = linkage(
                embeddings, method=self.method, metric=self.metric
            )

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

            threshold = min(max(self.threshold, min_threshold), max_threshold)

        else:

            threshold = (
                dendrogram[-num_clusters, 2]
                if num_clusters < num_embeddings
                else -np.inf
            )

        return fcluster(dendrogram, threshold, criterion="distance") - 1


class OracleClustering(BaseClustering):
    """Oracle clustering"""

    def __init__(self, metric: str = "cosine", expects_num_clusters: bool = False):
        super().__init__()

    def __call__(
        self,
        segmentations: SlidingWindowFeature = None,
        constraints: np.ndarray = None,
        file: AudioFile = None,
        frames: SlidingWindow = None,
        **kwargs,
    ) -> np.ndarray:
        """Apply agglomerative clustering

        Parameters
        ----------
        segmentations : (num_chunks, num_frames, num_speakers) array
            Binary segmentations.
        file : AudioFile
        frames : SlidingWindow

        Returns
        -------
        hard_clusters : (num_chunks, num_speakers) array
            Hard cluster assignment (hard_clusters[c, s] = k means that sth speaker
            of cth chunk is assigned to kth cluster)
        soft_clusters : (num_chunks, num_speakers, num_clusters) array
            Soft cluster assignment (the higher soft_clusters[c, s, k], the most likely
            the sth speaker of cth chunk belongs to kth cluster)
        """

        num_chunks, num_frames, num_speakers = segmentations.data.shape
        window = segmentations.sliding_window

        oracle_segmentations = oracle_segmentation(file, window, frames=frames)
        #   shape: (num_chunks, num_frames, true_num_speakers)

        file["oracle_segmentations"] = oracle_segmentations

        _, oracle_num_frames, num_clusters = oracle_segmentations.data.shape

        segmentations = segmentations.data[:, : min(num_frames, oracle_num_frames)]
        oracle_segmentations = oracle_segmentations.data[
            :, : min(num_frames, oracle_num_frames)
        ]

        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)
        soft_clusters = np.zeros((num_chunks, num_speakers, num_clusters))
        for c, (segmentation, oracle) in enumerate(
            zip(segmentations, oracle_segmentations)
        ):
            _, (permutation, *_) = permutate(oracle[np.newaxis], segmentation)
            for j, i in enumerate(permutation):
                if i is None:
                    continue
                hard_clusters[c, i] = j
                soft_clusters[c, i, j] = 1.0

        # when constraints is provided, report their quality
        if constraints is not None:
            must_link = constraints > 0
            num_must_link = np.sum(must_link)
            cannot_link = constraints < 0
            num_cannot_link = np.sum(cannot_link)
            same_cluster = pyannote.core.utils.distance.pdist(
                rearrange(hard_clusters, "c s -> (c s)"), metric="equal"
            )
            must_link_ratio = np.sum(must_link[same_cluster]) / np.sum(must_link)
            cannot_link_ratio = np.sum(cannot_link[~same_cluster]) / np.sum(cannot_link)

            criterion = math.sqrt(must_link_ratio * cannot_link_ratio)

            print(
                f"{num_must_link=} {num_cannot_link=} {must_link_ratio=:.2f} {cannot_link_ratio=:.2f} {criterion=:.2f}"
            )

        return hard_clusters, soft_clusters


class SpectralClustering(BaseClustering):
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

    def __init__(
        self,
        metric: str = "cosine",
        expects_num_clusters: bool = False,
        constrained_assignment: bool = False,
    ):

        super().__init__(
            metric=metric,
            expects_num_clusters=expects_num_clusters,
            constrained_assignment=constrained_assignment,
        )

        self.laplacian = Categorical(
            ["Affinity", "Unnormalized", "RandomWalk", "GraphCut"]
        )

        # Hyperparameters for refinement operations
        self.refinement = ParamDict(
            CropDiagonal=Categorical([True, False]),
            GaussianBlur=False,
            RowWiseThreshold=True,
            Symmetrize=Categorical([True, False]),
            Diffuse=Categorical([True, False]),
            RowWiseNormalize=Categorical([True, False]),
        )

        self.symmetrize_type = Categorical(["Max", "Average"])
        self.thresholding_type = Categorical(["RowMax", "Percentile"])
        self.thresholding_with_binarization = Categorical([True, False])
        self.thresholding_preserve_diagonal = Categorical([True, False])

        if not self.expects_num_clusters:
            self.eigengap = Categorical(["Ratio", "NormalizedDiff"])

            # HACK https://github.com/wq2012/SpectralCluster/issues/39
            # proportion of chunks with 1+ speaker that have 2+ speakers
            # according to the local segmentation model
            # (should be high for one-speaker audio but can also be high
            # for very audio with one dominant speaker)
            self.solo_speaker_ratio_threshold = Uniform(0.0, 1.0)
            self.solo_speaker_hac_threshold = Uniform(0.0, 1.0)

    def _affinity_function(self, embeddings: np.ndarray) -> np.ndarray:
        return squareform(1.0 - 0.5 * pdist(embeddings, metric=self.metric))

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = None,
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

        Returns
        -------
        clusters : (num_embeddings, ) array
            0-indexed cluster indices.
        """

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

        # Refinements
        refinement_sequence = [
            RefinementName[name] for name, active in self.refinement.items() if active
        ]
        refinement_options = RefinementOptions(
            refinement_sequence=refinement_sequence,
            gaussian_blur_sigma=0.0,
            thresholding_soft_multiplier=0.01,
            thresholding_type=ThresholdType[self.thresholding_type],
            thresholding_with_binarization=self.thresholding_with_binarization,
            thresholding_preserve_diagonal=self.thresholding_preserve_diagonal,
            symmetrize_type=SymmetrizeType[self.symmetrize_type],
        )

        # Laplacian
        laplacian_type = LaplacianType[self.laplacian]

        # Autotune
        if self.expects_num_clusters:
            autotune = None
            eigengap_type = EigenGapType.Ratio
        else:
            autotune = AutoTune(
                p_percentile_min=0.40,
                p_percentile_max=0.95,
                init_search_step=0.05,
                search_level=1,
            )
            eigengap_type = (EigenGapType[self.eigengap],)

        # Clustering
        clusters = SpectralClusterer(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            refinement_options=refinement_options,
            autotune=autotune,
            fallback_options=fallback_options,
            laplacian_type=laplacian_type,
            eigengap_type=eigengap_type,
            affinity_function=self._affinity_function,
        ).predict(embeddings)

        # if not self.expects_num_clusters:

        #     # post-process spectral clustering (SC) output to handle the known
        #     # limitation that SC is not very good at handling one-cluster data
        #     # SEE https://github.com/wq2012/SpectralCluster/issues/39

        #     # compute the ratio of chunks with 1+ active speaker
        #     # in which there are actually 2+ active  speakers.
        #     # if this ratio is high enough, it might be a cue that
        #     # there is only one cluster. therefore we postprocess the
        #     # output of spectral clustering by applying an additional
        #     # hierarchical agglomerative clustering step on the centroids

        #     num_active_speaker = np.sum(np.any(segmentations.data > 0, axis=1), axis=1)
        #     solo_speaker_ratio = np.sum(num_active_speaker == 1) / np.sum(
        #         num_active_speaker > 0
        #     )

        #     if (
        #         num_clusters > 1
        #         and solo_speaker_ratio > self.solo_speaker_ratio_threshold
        #     ):

        #         centroids_cluster = AgglomerativeClustering.cluster(
        #             centroids,
        #             min_clusters,
        #             max_clusters,
        #             num_clusters=num_clusters,
        #             threshold=self.solo_speaker_hac_threshold,
        #             method="average",
        #             metric=self.metric,
        #         )

        #         valid_clusters = centroids_cluster[valid_clusters]

        return clusters


class HiddenMarkovModelClustering(BaseClustering):
    """Hidden Markov Model with Gaussian states"""

    def __init__(
        self,
        metric: str = "cosine",
        expects_num_clusters: bool = False,
        constrained_assignment: bool = False,
    ):

        if metric not in ["euclidean", "cosine"]:
            raise ValueError("`metric` must be one of {'cosine', 'euclidean'}")

        super().__init__(
            metric=metric,
            expects_num_clusters=expects_num_clusters,
            constrained_assignment=constrained_assignment,
        )

        self.covariance_type = Categorical(["spherical", "diag", "full", "tied"])
        if not self.expects_num_clusters:
            self.threshold = Uniform(0.0, 2.0)

    def filter_embeddings(
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
        train_embeddings : (num_steps, dimension) array
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

        # TODO: generate alternative sequences that only differs from train_embeddings
        # in regions where there is overlap

        train_embeddings = embeddings[range(num_chunks), speaker_idx]
        # (num_chunks, dimension)

        # remove chunks with one of the following property:
        # * there is no active speaker in the center of the chunk
        # * embedding extraction has failed for the most active speaker in the center of the chunk
        center_is_non_speech = np.max(num_active_frames, axis=1) == 0.0
        embedding_is_invalid = np.any(np.isnan(train_embeddings), axis=1)
        chunk_idx = np.where(~(embedding_is_invalid | center_is_non_speech))[0]
        # (num_chunks, )

        return (train_embeddings[chunk_idx], chunk_idx, speaker_idx[chunk_idx])

    def fit_hmm(self, n_components, train_embeddings):

        hmm = GaussianHMM(
            n_components=n_components,
            covariance_type=self.covariance_type,
            n_iter=100,
            random_state=42,
            implementation="log",
            verbose=False,
        )
        hmm.fit(train_embeddings)

        return hmm

    def cluster(
        self,
        embeddings: np.ndarray,
        min_clusters: int,
        max_clusters: int,
        num_clusters: int = None,
        constraints: np.ndarray = None,
    ):

        must_link = constraints > 0
        cannot_link = constraints < 0

        print(f"must = {np.sum(must_link)}")
        print(f"cannot = {np.sum(cannot_link)}")

        # FIXME
        if max_clusters == len(embeddings):
            max_clusters = 20

        if self.metric == "cosine":
            # unit-normalize embeddings to somehow make them "euclidean"
            with np.errstate(divide="ignore", invalid="ignore"):
                embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

        if num_clusters is None:

            num_clusters = max_clusters
            for n_components in range(min_clusters, max_clusters + 1):

                hmm = self.fit_hmm(n_components, embeddings)
                train_clusters = hmm.predict(embeddings)

                same_cluster = pyannote.core.utils.distance.pdist(
                    train_clusters, metric="equal"
                )

                must_link_ratio = np.sum(must_link[same_cluster]) / np.sum(must_link)
                cannot_link_ratio = np.sum(cannot_link[~same_cluster]) / np.sum(
                    cannot_link
                )

                criterion = math.sqrt(must_link_ratio * cannot_link_ratio)

                print(
                    f"{must_link_ratio=:.2f} {cannot_link_ratio=:.2f} {criterion=:.2f}"
                )

                if n_components > 1:

                    # THIS IS A TERRIBLE CRITERION THAT NEEDS TO BE FIXED
                    # print(pdist(hmm.means_, metric="euclidean"))

                    # as soon as two states get too close to each other, stop adding states
                    min_state_dist = np.min(pdist(hmm.means_, metric="euclidean"))
                    # print(f"{n_components=} {min_state_dist=}")
                    if min_state_dist < self.threshold:
                        num_clusters = max(min_clusters, n_components - 1)
                        break

        if num_clusters == 1:
            return np.zeros((len(embeddings),), dtype=np.int8)

        # once num_clusters is estimated, fit the HMM several times
        # and keep the one that best fits the data
        hmm = self.fit_hmm(num_clusters, embeddings)

        try:
            train_clusters = hmm.predict(embeddings)
        except ValueError:
            # ValueError: startprob_ must sum to 1 (got nan)
            train_clusters = np.zeros((len(embeddings),), dtype=np.int8)

        return train_clusters


class Clustering(Enum):
    AgglomerativeClustering = AgglomerativeClustering
    SpectralClustering = SpectralClustering
    HiddenMarkovModelClustering = HiddenMarkovModelClustering
    GaussianMixtureClustering = GaussianMixtureClustering
    OracleClustering = OracleClustering
