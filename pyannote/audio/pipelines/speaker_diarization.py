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
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, squareform
from scipy.special import softmax

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.distance import pdist
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Categorical, Frozen, Uniform

from .clustering import Clustering

# TODO: automagically estimate HAC threshold based on local segmentation


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
    clustering : {"AffinityPropagation", "DBSCAN", "OPTICS", "AgglomerativeClustering"}, optional
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
    ):

        super().__init__()

        self.segmentation = segmentation
        self.embedding = embedding

        try:
            self.clustering = Clustering[clustering].value()
        except KeyError:
            raise ValueError(
                f'clustering must be one of [{", ".join(list(Clustering.__members__))}]'
            )

        self.seg_model_: Model = get_model(segmentation)

        # TODO: add support for SpeechBrain ECAPA-TDNN
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

        self.warm_up = Uniform(0.0, 0.2)

        # hyper-parameters
        self.onset = Uniform(0.05, 0.95)
        self.offset = Uniform(0.05, 0.95)
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

        # affinity
        self.use_overlap_aware_embedding = Categorical([True, False])
        self.affinity_threshold_percentile = Uniform(0.0, 1.0)

        # weights of constraints in final (constrained) affinity matrix.
        # Between 0 and 1, where alpha = 0.0 means no constraint.
        self.constraint_propagate = Uniform(0.0, 1.0)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    def trim_warmup(self, segmentations: SlidingWindowFeature) -> SlidingWindowFeature:

        _, num_frames, _ = segmentations.data.shape
        new_data = segmentations.data[
            :,
            round(num_frames * self.warm_up) : round(num_frames * (1.0 - self.warm_up)),
        ]

        chunks = segmentations.sliding_window
        new_chunks = SlidingWindow(
            start=chunks.start + self.warm_up * chunks.duration,
            step=chunks.step,
            duration=(1.0 - 2 * self.warm_up) * chunks.duration,
        )

        return SlidingWindowFeature(new_data, new_chunks)

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
        segmentations : SlidingWindowFeature
            (num_chunks, num_frames, num_speakers)-shaped segmentation.

        Returns
        -------
        constraints : np.ndarray
            (num_chunks x num_speakers, num_chunks x num_speakers)-shaped constraint matrix

        """

        num_chunks, num_frames, num_speakers = segmentations.data.shape

        # 1. intra-chunk "cannot link" constraints (upper triangle only)
        chunk_idx = np.broadcast_to(np.arange(num_chunks), (num_speakers, num_chunks))
        constraint = np.triu(
            squareform(
                -1.0
                * pdist(einops.rearrange(chunk_idx, "s c -> (c s)"), metric="equal")
            )
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
        # np.fill_diagonal(constraint, 1.0)
        for C, (_, segmentation) in enumerate(segmentations):
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
                            c * num_speakers + past,
                            C * num_speakers + this,
                        ] = 1.0
                        # TODO: investigate weighting this by (num_frames - shift) / num_frames
                        # TODO: i.e. by the duration of the common temporal support

        # propagate cannot link constraints by "transitivity": if c_ij = -1 and c_jk = 1 then c_ik = -1
        # (only when this new constraint is not conflicting with existing constraint, i.e. when c_ik = 1)

        # loop on (i, j) pairs such that c_ij is either 1 or -1
        for i, j in zip(*np.where(constraint != 0)):

            # find all k for which c_ij = - c_jk and mark c_ik as cannot-link
            # unless it has been marked as must-link (c_ik = 1) before
            constraint[
                i, (constraint[i] != 1.0) & (constraint[j] + constraint[i, j] == 0.0)
            ] = -1.0

        # make constraint matrix symmetric
        constraint = squareform(squareform(constraint, checks=False))
        np.fill_diagonal(constraint, 1.0)

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

    def apply(self, file: AudioFile, debug: bool = False) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        debug : bool, optional
            Set to True to add debugging keys into `file`.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        # __ LOCAL SPEAKER SEGMENTATION ________________________________________________
        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, num_speakers)
        if (not self.training) or (
            self.training and self.CACHED_SEGMENTATION not in file
        ):
            file[self.CACHED_SEGMENTATION] = self._segmentation_inference(file)
        segmentations: SlidingWindowFeature = file[self.CACHED_SEGMENTATION]

        # trim warm-up regions
        segmentations = self.trim_warmup(segmentations)
        num_chunks, num_frames, num_speakers = segmentations.data.shape

        # __ LOCAL SPEAKER EMBEDDING ___________________________________________________
        # extract embeddings (only if needed)
        # output shape is (num_valid_chunks x num_speakers, embedding_dimension)
        if (
            (not self.training)
            or (not isinstance(self.warm_up, Frozen))
            or (self.training and self.CACHED_EMBEDDING not in file)
        ):

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
        # actives[c, k] indicates whether kth speaker is active in cth chunk
        actives: np.ndarray = np.any(segmentations > self.onset, axis=1).data
        # (num_chunks, num_speakers)

        # __ DEBUG [ACTIVE] ____________________________________________________________
        if debug:
            file["@diarization/segmentation/active"] = np.copy(actives)

        # TODO: use "pure_long_enough" instead of "actives"...
        # TODO: ... and then assign "actives and not pure_long_enough" in a postprocessing step

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

                if np.sum(actives[c]) == 0:
                    continue
                segmentation = segmentation[np.newaxis, :, actives[c]]

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

        actives = einops.rearrange(actives, "c s -> (c s)")

        # __ SEGMENTATION-BASED CLUSTERING CONSTRAINTS _________________________________
        # compute constraints based on segmentation

        # compute (soft) {must/cannot}-link constraints based on local segmentation
        constraint = self.compute_constraints(segmentations)
        # (num_valid_chunks x num_speakers, num_valid_chunks x num_speakers)

        # __ DEBUG [CONSTRAINTS] ___________________________________________________
        if debug:
            file["@diarization/constraint/raw"] = np.copy(constraint)

        constraint = constraint[actives][:, actives]
        # (num_active_speakers, num_active_speakers)

        # __ EMBEDDING-BASED AFFINITY MATRIX ___________________________________________
        # compute affinity matrix (num_active_speakers, num_active_speakers)-shaped
        affinity = squareform(1 - 0.5 * pdist(embeddings[actives], metric="cosine"))

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
        num_active = np.sum(actives)
        if num_active < 2:
            clusters[actives] = 0
            num_clusters = 1

        else:
            clusters[actives] = self.clustering(affinity)
            num_clusters = np.max(clusters) + 1

            # corner case where affinity propagation fails to converge
            # and returns only -1 labels
            if num_clusters == 0:
                clusters[actives] = 0
                num_clusters = 1

        # __ DEBUG [CLUSTERING] ________________________________________________________
        if debug:
            file["@diarization/clusters/raw"] = np.copy(
                einops.rearrange(clusters, "(c s) -> c s", c=num_chunks, s=num_speakers)
            )

        # __ FINAL SPEAKER ASSIGNMENT ___________________________________________________

        # compute speaker-to-centroid distance matrix
        centroids = np.vstack(
            [np.mean(embeddings[clusters == k], axis=0) for k in range(num_clusters)]
        )
        distances = cdist(embeddings, centroids, metric="cosine")
        # (chunk_idx x num_speakers, num_clusters)

        # __ DEBUG [CLUSTERING] _________________________________________________________
        if debug:
            file["@diarization/clusters/centroids"] = np.copy(centroids)
            file["@diarization/distances/raw"] = np.copy(distances)

        # artificially decrease distance to actual cluster
        # (to force assignment to this cluster)
        assigned = clusters > -1
        distances[np.where(assigned)[0], clusters[assigned]] -= 1000.0

        # artificially increase distance for inactive speakers
        # (to prevent them from being assigned to any cluster)
        distances[np.where(~actives)] += 1000.0

        # reshape matrices
        distances = einops.rearrange(
            distances,
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
            k=num_clusters,
        )
        actives = einops.rearrange(
            actives, "(c s) -> c s", c=num_chunks, s=num_speakers
        )
        clusters = einops.rearrange(
            clusters, "(c s) -> c s", c=num_chunks, s=num_speakers
        )

        # find optimal bijective assignment between active speakers and clusters
        # (while preventing two speakers of the same chunk from being assigned
        # to the same cluster)
        for c, (distance, active) in enumerate(zip(distances, actives)):
            # distance is (num_speakers, num_clusters)-shaped
            # active is (num_speakers)-shaped
            for s, k in zip(*linear_sum_assignment(distance, maximize=False)):
                clusters[c, s] = k if active[s] else -2

        # __ DEBUG [CLUSTERING] ________________________________________________________
        if debug:
            file["@diarization/clusters/assigned"] = np.copy(clusters)

        # __ CLUSTERING-BASED SEGMENTATION AGGREGATION _________________________________
        # build final aggregated speaker activations
        # clusters = einops.rearrange(
        #     clusters, "(c s) -> c s", c=num_chunks, s=num_speakers
        # )
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
            file["@diarization/segmentation/clustered"] = clustered_segmentations

        activations = Inference.aggregate(clustered_segmentations, frames)

        # __ DEBUG [AGGREGATION] _______________________________________________________
        if debug:
            file["@diarization/segmentation/aggregated"] = activations

        # __ FINAL BINARIZATION ________________________________________________________

        diarization = self._binarize(activations)
        diarization.uri = file["uri"]

        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
