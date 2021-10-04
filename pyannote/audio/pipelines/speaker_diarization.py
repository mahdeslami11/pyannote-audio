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

"""Speaker diarization pipelines"""

from typing import Mapping, Text

import einops
import numpy as np
import torch
from scipy.spatial.distance import cdist, squareform
from scipy.special import softmax
from sklearn.cluster import DBSCAN as SKLearnDBSCAN
from sklearn.cluster import OPTICS as SKLearnOPTICS
from sklearn.cluster import AffinityPropagation as SKLearnAffinityPropagation

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.distance import pdist
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline import Pipeline as BasePipeline
from pyannote.pipeline.parameter import Categorical, Integer, Uniform


class AffinityPropagation(BasePipeline):
    def __init__(self):
        super().__init__()
        self.damping = Uniform(0.5, 1.0)
        self.preference = Uniform(-50.0, 0.0)  # check what this interval should be

    def initialize(self):
        self._affinity_propagation = SKLearnAffinityPropagation(
            damping=self.damping,
            max_iter=200,
            convergence_iter=15,
            copy=True,
            preference=self.preference,
            affinity="precomputed",
            verbose=False,
            random_state=1337,  # for reproducibility
        )

    def __call__(self, affinity: np.ndarray) -> np.ndarray:
        return self._affinity_propagation.fit_predict(affinity)


class DBSCAN(BasePipeline):
    def __init__(self):
        super().__init__()
        self.eps = Uniform(0.0, 1.0)
        self.min_samples = Integer(2, 20)

    def initialize(self):
        self._dbscan = SKLearnDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
            algorithm="auto",
            leaf_size=30,
            n_jobs=None,
        )

    def __call__(self, affinity: np.ndarray) -> np.ndarray:
        return self._dbscan.fit_predict(1.0 - affinity)


class OPTICS(BasePipeline):
    def __init__(self):
        super().__init__()
        self.min_samples = Integer(2, 20)
        self.max_eps = Uniform(0.0, 1.0)
        self.xi = Uniform(0.0, 1.0)

    def initialize(self):
        self._optics = SKLearnOPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric="precomputed",
            cluster_method="xi",
            xi=self.xi,
            predecessor_correction=True,
            min_cluster_size=None,
            algorithm="auto",
            leaf_size=30,
            memory=None,
            n_jobs=None,
        )

    def __call__(self, affinity: np.ndarray) -> np.ndarray:
        return self._optics.fit_predict(1.0 - affinity)


class SpeakerDiarization(Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    segmentation : Inference or str, optional
        `Inference` instance used to extract raw segmentation scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "seg".
    embeddings : Inference or str, optional
        `Inference` instance used to extract speaker embeddings. When `str`,
        assumes that file already contains a corresponding key with precomputed
        embeddings. Defaults to "emb".
    optimize_with_expected_num_speakers : bool, optional
        Set to True to automatically pass the expected number of speakers when optimizing
        the pipeline (pipeline(file, expected_num_speakers=...)).
    clustering : {"AffinityPropagation", "DBSCAN", "OPTICS"}, optional
        Defaults to "AffinityPropagation".


    Hyper-parameters
    ----------------

    Usage
    -----
    >>> pipeline = SpeakerDiarization()
    >>> diarization = pipeline("/path/to/audio.wav")
    >>> diarization = pipeline("/path/to/audio.wav", expected_num_speakers=2)

    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        embedding: PipelineModel = "pyannote/embedding",
        clustering: Text = "AffinityPropagation",
        optimize_with_expected_num_speakers: bool = False,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.embedding = embedding
        self.optimize_with_expected_num_speakers = optimize_with_expected_num_speakers

        self.seg_model_: Model = get_model(segmentation)
        self.emb_model_: Model = get_model(embedding)
        self.emb_model_.eval()

        # send models to GPU (when GPUs are available and model is not already on GPU)
        cpu_models = [
            model
            for model in (self.seg_model_, self.emb_model_)
            if model.device.type == "cpu"
        ]
        for cpu_model, gpu_device in zip(
            cpu_models, get_devices(needs=len(cpu_models))
        ):
            cpu_model.to(gpu_device)

        self._segmentation_inference = Inference(self.seg_model_, skip_aggregation=True)

        # hyper-parameters
        self.onset = Uniform(0.05, 0.95)
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

        self.use_overlap_aware_embedding = Categorical([True, False])
        self.affinity_threshold_percentile = Uniform(0.0, 1.0)

        # Weights of constraints in final (constrained) affinity matrix.
        # Between 0 and 1, where alpha = 0.0 means no constraint.
        self.constraint_propagate = Uniform(0.0, 1.0)
        self.constraint_must_link = Uniform(0.0, 1.0)
        self.constraint_cannot_link = Uniform(0.0, 1.0)

        if clustering == "AffinityPropagation":
            self.clustering = AffinityPropagation()
        elif clustering == "DBSCAN":
            self.clustering = DBSCAN()
        elif clustering == "OPTICS":
            self.clustering = OPTICS()
        else:
            raise ValueError(
                "'clustering' must be one of {AffinityPropagation, DBSCAN, OPTICS}"
            )

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    @staticmethod
    def get_pooling_weights(segmentation: np.ndarray) -> np.ndarray:
        """Overlap-aware weights

        Parameters
        ----------
        segmentation: np.ndarray
            (num_frames, num_speakers) segmentation scores

        Returns
        -------
        weights: np.ndarray
            (num_frames, num_speakers) overlap-aware weights
        """

        power: int = 3
        scale: float = 10.0
        pow_segmentation = pow(segmentation, power)
        weights = pow_segmentation * pow(
            softmax(scale * pow_segmentation, axis=1), power
        )
        weights[weights < 1e-8] = 1e-8
        return weights

    @staticmethod
    def get_embedding(
        file: AudioFile,
        chunk: Segment,
        model: Model,
        pooling_weights: np.ndarray = None,
    ) -> np.ndarray:
        """Extract embedding from a chunk

        Parameters
        ----------
        file : AudioFile
        chunk : Segment
        model : Model
            Pretrained embedding model.
        pooling_weights : np.ndarray, optional
            (num_frames, num_speakers) pooling weights

        Returns
        -------
        embeddings : np.ndarray
            (1, dimension) if pooling_weights is None, else (num_speakers, dimension)
        """

        if pooling_weights is None:
            num_speakers = 1

        else:
            _, num_speakers = pooling_weights.shape
            pooling_weights = (
                torch.from_numpy(pooling_weights).float().T.to(model.device)
            )
            # (num_speakers, num_frames)

        waveforms = (
            model.audio.crop(file, chunk)[0]
            .unsqueeze(0)
            .expand(num_speakers, -1, -1)
            .to(model.device)
        )
        # (num_speakers, num_channels == 1, num_samples)

        with torch.no_grad():
            if pooling_weights is None:
                embeddings = model(waveforms)
            else:
                embeddings = model(waveforms, weights=pooling_weights)

        embeddings = embeddings.cpu().numpy()

        return embeddings

    def compute_constraints(self, segmentations: SlidingWindowFeature) -> np.ndarray:
        """

        Parameters
        ----------
        cannot_link : float, optional

        must_link : float, optional


        """

        num_chunks, num_frames, num_speakers = segmentations.data.shape

        # 1. intra-chunk "cannot link" constraints
        chunk_idx = np.broadcast_to(np.arange(num_chunks), (num_speakers, num_chunks))
        constraint = squareform(
            -self.constraint_cannot_link
            * pdist(einops.rearrange(chunk_idx, "s c -> (c s)"), metric="equal")
        )
        # (num_chunks x num_speakers, num_chunks x num_speakers)

        # 2. inter-chunk "must link" constraints
        # two speakers from two overlapping chunks are marked as "must-link"
        # if and only if the optimal permutation maps them and they are
        # both active in their common temporal support.
        chunks = segmentations.sliding_window

        # number of overlapping chunk
        num_overlapping_chunks = round(chunks.duration // chunks.step - 1.0)

        # loop on pairs of overlapping chunks
        np.fill_diagonal(constraint, 1.0)
        for C, (chunk, segmentation) in enumerate(segmentations):
            for c in range(max(0, C - num_overlapping_chunks), C):

                # extract common temporal support
                shift = round((C - c) * num_frames * chunks.step / chunks.duration)
                this_segmentation = segmentation[: num_frames - shift]
                past_segmentation = segmentations[c, shift:]

                # find the optimal one-to-one mapping
                _, (permutation,) = permutate(
                    this_segmentation[np.newaxis], past_segmentation
                )

                # check whether speakers are active on the common temporal support
                # otherwise there is no point trying to match them
                this_active = np.any(this_segmentation > self.onset, axis=0)
                past_active = np.any(past_segmentation > self.onset, axis=0)

                for this, past in enumerate(permutation):
                    if this_active[this] and past_active[past]:
                        constraint[
                            C * num_speakers + this, c * num_speakers + past
                        ] = self.constraint_must_link
                        # TODO: investigate weighting this by (num_frames - shift) / num_frames
                        # TODO: i.e. by the duration of the common temporal support

                        # make constraint matrix symmetric
                        constraint[
                            c * num_speakers + past, C * num_speakers + this
                        ] = constraint[C * num_speakers + this, c * num_speakers + past]

        return constraint

    def propagate_constraints(
        self, affinity: np.ndarray, constraint: np.ndarray
    ) -> np.ndarray:
        """Update affinity matrix by constraint propagation

        Stolen from
        https://github.com/wq2012/SpectralCluster/blob/34d155654dbbfcda61b808a4f61afa666476b3d2/spectralcluster/constraint.py

        Parameters
        ----------
        affinity : np.ndarray
            (N, N) affinity matrix with values in [0, 1].
            * affinity[i, j] = 1 indicates that i and j are very similar
            * affinity[i, j] = 0 indicates that i and j are very dissimilar
        constraint : np.ndarray
            (N, N) constraint matrix with values in [-1, 1].
            * constraint[i, j] > 0 indicates a must-link constraint
            * constraint[i, j] < 0 indicates a cannot-link constraint
            * constraint[i, j] = 0 indicates absence of constraint

        Returns
        -------
        constrained_affinity : np.ndarray
            Constrained affinity matrix.
        propagated_constraint : np.ndarray
            Propagated constraint matrix.

        Reference
        ---------
        Lu, Zhiwu, and IP, Horace HS.
        "Constrained spectral clustering via exhaustive and efficient constraint propagation."
        ECCV 2010
        """

        degree = np.diag(np.sum(affinity, axis=1))
        degree_norm = np.diag(1 / (np.sqrt(np.diag(degree)) + 1e-10))

        # Compute affinity_norm as D^(-1/2)AD^(-1/2)
        affinity_norm = degree_norm.dot(affinity).dot(degree_norm)

        # The closed form of the final converged constraint matrix is:
        # (1-alpha)^2 * (I-alpha*affinity_norm)^(-1) * constraint *
        # (I-alpha*affinity_norm)^(-1). We save (I-alpha*affinity_norm)^(-1) as a
        # `temp_value` for readibility
        temp_value = np.linalg.inv(
            np.eye(affinity.shape[0]) - (1 - self.constraint_propagate) * affinity_norm
        )
        propagated_constraint = self.constraint_propagate ** 2 * temp_value.dot(
            constraint
        ).dot(temp_value)

        # `is_positive` is a mask matrix where values of the propagated_constraint
        # are positive. The affinity matrix is adjusted by the final constraint
        # matrix using equation (4) in reference paper
        is_positive = propagated_constraint > 0
        affinity1 = 1 - (1 - propagated_constraint * is_positive) * (
            1 - affinity * is_positive
        )
        affinity2 = (1 + propagated_constraint * np.invert(is_positive)) * (
            affinity * np.invert(is_positive)
        )
        return affinity1 + affinity2, propagated_constraint

    CACHED_SEGMENTATION = "@diarization/segmentation/raw"
    CACHED_EMBEDDING = "@diarization/embedding/raw"

    def apply(
        self, file: AudioFile, expected_num_speakers: int = None, debug: bool = False
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        expected_num_speakers : int, optional
            Expected number of speakers. Defaults to estimate it automatically.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        # when optimizing with expected number of speakers, use reference annotation
        # to obtain the expected number of speakers
        if self.training and self.optimize_with_expected_num_speakers:
            expected_num_speakers = len(file["annotation"].labels())

        if expected_num_speakers is not None:
            raise NotImplementedError(
                "Speaker diarization with expected number of speakers is not supported yet"
            )

        # __ LOCAL SPEAKER SEGMENTATION ________________________________________________
        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, num_speakers)
        if (not self.training) or (
            self.training and self.CACHED_SEGMENTATION not in file
        ):
            file[self.CACHED_SEGMENTATION] = self._segmentation_inference(file)
        segmentations: SlidingWindowFeature = file[self.CACHED_SEGMENTATION]
        num_chunks, num_frames, num_speakers = segmentations.data.shape

        # __ LOCAL SPEAKER EMBEDDING ___________________________________________________
        # extract embeddings (only if needed)
        # output shape is (num_valid_chunks x num_speakers, embedding_dimension)
        if (not self.training) or (self.training and self.CACHED_EMBEDDING not in file):

            embeddings = []

            for c, (chunk, segmentation) in enumerate(segmentations):

                if self.use_overlap_aware_embedding:
                    pooling_weights: np.ndarray = self.get_pooling_weights(segmentation)
                    # (num_frames, num_speakers)
                else:
                    pooling_weights: np.ndarray = segmentation
                    # (num_frames, num_speakers)

                try:
                    chunk_embeddings: np.ndarray = self.get_embedding(
                        file, chunk, self.emb_model_, pooling_weights=pooling_weights
                    )
                    # (num_speakers, dimension)

                except ValueError:
                    if c + 1 == num_chunks:
                        # it might happen that one cannot extract embeddings from
                        # the very last chunk because of audio duration.
                        continue
                    else:
                        # however, if we fail in the middle of the file, something
                        # bad has happened and we should not go any further...
                        raise ValueError()

                embeddings.append(chunk_embeddings)

            embeddings = np.vstack(embeddings)
            # (num_valid_chunks x num_speakers, dimension)

            # unit-normalize embeddings
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

            # cache embeddings
            file[self.CACHED_EMBEDDING] = embeddings

        embeddings = file[self.CACHED_EMBEDDING]
        # update number of chunks (only those with embeddings)
        num_chunks = int(embeddings.shape[0] / num_speakers)
        segmentations.data = segmentations.data[:num_chunks]

        frames: SlidingWindow = self._segmentation_inference.model.introspection.frames
        # frame resolution (e.g. duration = step = 17ms)

        # __ LOCALLY ACTIVE SPEAKER DETECTION __________________________________________
        # active.data[c, k] indicates whether kth speaker is active in cth chunk
        active: np.ndarray = np.any(segmentations > self.onset, axis=1).data
        # (num_chunks, num_speakers)

        # TODO: use "pure_long_enough" instead of "active"...
        # TODO: ... and then assign "active and not pure_long_enough" in a postprocessing step

        # __ DEBUG [ORACLE CLUSTERING] _________________________________________________
        if debug and isinstance(file, Mapping) and "annotation" in file:

            reference = file["annotation"].discretize(
                duration=segmentations.sliding_window[len(segmentations) - 1].end,
                resolution=frames,
            )
            permutations = []

            for (
                c,
                (chunk, segmentation),
            ) in enumerate(segmentations):

                if np.sum(active[c]) == 0:
                    continue
                segmentation = segmentation[np.newaxis, :, active[c]]

                local_reference = reference.crop(chunk)
                _, (permutation,) = permutate(
                    segmentation,
                    local_reference[:num_frames],
                )
                active_reference = np.any(local_reference > 0, axis=0)
                permutations.extend(
                    [
                        i if ((i is not None) and (active_reference[i])) else None
                        for i in permutation
                    ]
                )

            file["@diarization/clusters/oracle"] = permutations

        active = einops.rearrange(active, "c s -> (c s)")

        # __ SEGMENTATION-BASED CLUSTERING CONSTRAINTS _________________________________
        # compute constraints based on segmentation

        # compute (soft) {must/cannot}-link constraints based on local segmentation
        constraint = self.compute_constraints(segmentations)
        # (num_valid_chunks x num_speakers, num_valid_chunks x num_speakers)

        # __ DEBUG [CONSTRAINTS] ___________________________________________________
        if debug:
            file["@diarization/constraint/raw"] = np.copy(constraint)

        constraint = constraint[active][:, active]
        # (num_active_speakers, num_active_speakers)

        # __ EMBEDDING-BASED AFFINITY MATRIX ___________________________________________
        # compute affinity matrix (num_active_speakers, num_active_speakers)-shaped
        affinity = squareform(1 - 0.5 * pdist(embeddings[active], metric="cosine"))

        # __ DEBUG [AFFINITY] __________________________________________________________
        if debug:
            file["@diarization/affinity/raw"] = np.copy(affinity)

        # __ AFFINITY MATRIX REFINEMENT BY NEAREST NEIGHBOR FILTERING __________________
        affinity = affinity * (
            affinity
            > np.percentile(affinity, 100 * self.affinity_threshold_percentile, axis=0)
        )
        # make the affinity matrix symmetric again
        affinity = 0.5 * (affinity + affinity.T)

        # __ DEBUG [AFFINITY] ______________________________________________________
        if debug:
            file["@diarization/affinity/refined"] = np.copy(affinity)

        # __ AFFINITY MATRIX REFINEMENT BY CONSTRAINT PROPAGATION ______________________
        affinity, constraint = self.propagate_constraints(affinity, constraint)

        # __ DEBUG [AFFINITY] ______________________________________________________
        if debug:
            file["@diarization/constraint/propagated"] = np.copy(constraint)
            file["@diarization/affinity/constrained"] = np.copy(affinity)

        # __ ACTIVE SPEAKER CLUSTERING _________________________________________________
        # clusters[chunk_id x num_speakers + speaker_id] = k
        # * k=-2                if speaker is inactive
        # * k=-1                if speaker is active but not assigned to any cluster
        # * k in {0, ... K - 1} if speaker is active and is assigned to cluster k
        clusters = -2 * np.ones(num_chunks * num_speakers, dtype=np.int)
        num_active = np.sum(active)
        if num_active < 2:
            clusters[active] = 0
            num_clusters = 1

        else:
            clusters[active] = self.clustering(affinity)
            num_clusters = np.max(clusters) + 1

        # __ DEBUG [CLUSTERING] ________________________________________________________
        if debug:
            file["@diarization/clusters/raw"] = np.copy(
                einops.rearrange(clusters, "(c s) -> c s", c=num_chunks, s=num_speakers)
            )

        # __ UNASSIGNED SPEAKER ASSIGNMENT _____________________________________________
        # assign not-yet-assigned speakers to closest centroid
        centroids = np.vstack(
            [np.mean(embeddings[clusters == k], axis=0) for k in range(num_clusters)]
        )
        distances = cdist(embeddings, centroids, metric="cosine")
        unassigned = clusters == -1
        clusters[unassigned] = np.argmin(distances[unassigned], axis=1)

        # TODO: make sure two speakers from the same chunk are not assigned to the same cluster
        # TODO: this can be done using linear_sum_assignment with a custom speaker-to-centroid
        # TODO: distance matrix crafted in such a way that previously assigned clusters are not
        # TODO: altered...

        # __ DEBUG [CLUSTERING] ________________________________________________________
        if debug:
            file["@diarization/clusters/centroids"] = np.copy(centroids)
            file["@diarization/clusters/assigned"] = np.copy(
                einops.rearrange(clusters, "(c s) -> c s", c=num_chunks, s=num_speakers)
            )

        # __ CLUSTERING-BASED SEGMENTATION AGGREGATION _________________________________
        # build final aggregated speaker activations
        clusters = einops.rearrange(
            clusters, "(c s) -> c s", c=num_chunks, s=num_speakers
        )
        clustered_segmentations = np.NAN * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )
        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(clusters, segmentations)
        ):
            # cluster is (num_speakers, )-shaped
            # segmentation is (num_frames, num_speakers)-shaped
            for s, k in enumerate(cluster):
                if k == -2:
                    continue
                clustered_segmentations[c, :, k] = segmentation[:, s]

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )

        # __ DEBUG [AGGREGATION] _______________________________________________________
        if debug:
            file[
                "@diarization/segmentation/clustered"
            ] = clustered_segmentations  # copy?

        speaker_activations = Inference.aggregate(clustered_segmentations, frames)

        # __ DEBUG [AGGREGATION] _______________________________________________________
        if debug:
            file["@diarization/segmentation/aggregated"] = speaker_activations

        # __ SPEAKER COUNTING __________________________________________________________
        # estimate instantaneous number of speakers
        active_speaker_count = Inference.aggregate(
            np.sum(segmentations > self.onset, axis=-1, keepdims=True),
            frames,
        )
        active_speaker_count.data = np.round(active_speaker_count)
        # TODO: improve speaker counting by using onset AND offset?

        # __ DEBUG [COUNTING] __________________________________________________________
        if debug:
            file["@diarization/speaker_count"] = active_speaker_count

        # __ FINAL BINARIZATION ________________________________________________________
        sorted_speakers = np.argsort(-speaker_activations, axis=-1)
        binarized = np.zeros_like(speaker_activations.data)
        for t, ((_, count), speakers) in enumerate(
            zip(active_speaker_count, sorted_speakers)
        ):
            # TODO: find a way to stop clustering early enough to avoid num_clusters < count
            count = min(num_clusters, int(count.item()))
            for i in range(count):
                binarized[t, speakers[i]] = 1.0

        diarization = self._binarize(SlidingWindowFeature(binarized, frames))
        diarization.uri = file["uri"]

        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
