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

"""Speaker diarization pipelines"""

import itertools
import warnings
from typing import Callable, Optional

# import networkx as nx
import numpy as np

# import scipy.special
import torch
from einops import rearrange
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature

# from pyannote.core import Segment
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

# from pyannote.pipeline.parameter import Uniform, Categorical
from pyannote.pipeline.parameter import ParamDict, Uniform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_devices,
    get_model,
)
from pyannote.audio.utils.signal import multi_binarize

# try:
#     import cvxpy as cv
#     CVXPY_IS_AVAILABLE = True
# except ImportError:
#     CVXPY_IS_AVAILABLE = False


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    segmentation_step: float, optional
        Defaults to 0.1.
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 32.
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    embedding_batch_size : int, optional
        Batch size used for speaker embedding. Defaults to 32.
    clustering : str, optional
        Clustering algorithm. See pyannote.audio.pipelines.clustering.Clustering
        for available options. Defaults to "GaussianHiddenMarkovModel".
    expects_num_speakers : bool, optional
        Defaults to False.

    Hyper-parameters
    ----------------
    segmentation_onset : float
    embedding_onset : float
    clustering hyperparameters

    Usage
    -----
    >>> pipeline = SpeakerDiarization()
    >>> diarization = pipeline("/path/to/audio.wav")
    >>> diarization = pipeline("/path/to/audio.wav", num_speakers=2)

    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        segmentation_step: float = 0.1,
        segmentation_batch_size: int = 32,
        embedding: PipelineModel = "pyannote/embedding",
        embedding_batch_size: int = 32,
        clustering: str = "GaussianHiddenMarkovModel",
        expects_num_speakers: bool = False,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.segmentation_batch_size = segmentation_batch_size
        self.segmentation_step = segmentation_step

        self.embedding = embedding
        self.embedding_batch_size = embedding_batch_size

        self.klustering = clustering
        self.expects_num_speakers = expects_num_speakers

        seg_device, emb_device = get_devices(needs=2)

        model: Model = get_model(segmentation)
        model.to(seg_device)

        self._segmentation = Inference(
            model,
            duration=model.specifications.duration,
            step=self.segmentation_step * model.specifications.duration,
            skip_aggregation=True,
            batch_size=self.segmentation_batch_size,
        )
        self._frames: SlidingWindow = self._segmentation.model.introspection.frames

        self.segmentation_onset = ParamDict(
            main=Uniform(0.01, 0.99), left=Uniform(0.01, 0.99)
        )
        # self.segmentation_onset = Uniform(0.01, 0.99)

        if self.klustering == "OracleClustering":
            metric = "not_applicable"

        else:
            self._embedding = PretrainedSpeakerEmbedding(
                self.embedding, device=emb_device
            )
            self._audio = Audio(sample_rate=self._embedding.sample_rate, mono=True)
            metric = self._embedding.metric
            # self.embedding_onset = Uniform(0.1, 0.9)

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f'clustering must be one of [{", ".join(list(Clustering.__members__))}]'
            )
        self.clustering = Klustering.value(
            metric=metric,
            expects_num_clusters=self.expects_num_speakers,
        )

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

    @property
    def CACHED_SEGMENTATION(self):
        return "cache/segmentation"

    def get_segmentations(self, file) -> SlidingWindowFeature:
        """Apply segmentation model

        Parameter
        ---------
        file : AudioFile

        Returns
        -------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        """
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file)

        return segmentations

    def get_embeddings(self, file, segmentations: SlidingWindowFeature):
        """Extract embeddings for each (chunk, speaker) pair

        Parameters
        ----------
        file : AudioFile
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature

        Returns
        -------
        embeddings : (num_chunks, num_speakers, dimension) array
        """

        # TODO: extract both overlap aware and unaware embeddings

        def iter_waveform_and_mask():
            for chunk, masks in segmentations:
                # chunk: Segment(t, t + duration)
                # masks: (num_frames, local_num_speakers) np.ndarray

                waveform, _ = self._audio.crop(
                    file,
                    chunk,
                    duration=segmentations.sliding_window.duration,
                    mode="pad",
                )
                # waveform: (1, num_samples) torch.Tensor

                # mask may contain NaN (in case of partial stitching)
                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)

                for mask in masks.T:
                    # mask: (num_frames, ) np.ndarray

                    yield waveform[None], torch.from_numpy(mask)[None]
                    # w: (1, 1, num_samples) torch.Tensor
                    # m: (1, num_frames) torch.Tensor

        batches = batchify(
            iter_waveform_and_mask(),
            batch_size=self.embedding_batch_size,
            fillvalue=(None, None),
        )

        embedding_batches = []

        for batch in batches:
            waveforms, masks = zip(*filter(lambda b: b[0] is not None, batch))

            waveform_batch = torch.vstack(waveforms)
            # (batch_size, 1, num_samples) torch.Tensor

            mask_batch = torch.vstack(masks)
            # (batch_size, num_frames) torch.Tensor

            embedding_batch: np.ndarray = self._embedding(
                waveform_batch, masks=mask_batch
            )
            # (batch_size, dimension) np.ndarray

            embedding_batches.append(embedding_batch)

        return rearrange(
            np.vstack(embedding_batches), "(c s) d -> c s d", c=len(segmentations)
        )

    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build final discrete diarization out of clustered segmentation

        Parameters
        ----------
        segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Raw speaker segmentation.
        hard_clusters : (num_chunks, num_speakers) array
            Output of clustering step.
        count : (total_num_frames, 1) SlidingWindowFeature
            Instantaneous number of active speakers.

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

        num_clusters = np.max(hard_clusters) + 1
        clustered_segmentations = np.NAN * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )

        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(hard_clusters, segmentations)
        ):

            # cluster is (local_num_speakers, )-shaped
            # segmentation is (num_frames, local_num_speakers)-shaped
            for k in np.unique(cluster):
                if k == -2:
                    continue

                # TODO: can we do better than this max here?
                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )

        return self.to_diarization(clustered_segmentations, count)

    # def soft_stitching(
    #     self, soft_clusters: np.ndarray, stitching_graph: nx.Graph
    # ) -> np.ndarray:
    #     """WORK IN PROGRESS

    #     Parameters
    #     ----------
    #     soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
    #     stitiching_graph : nx.Graph

    #     Returns
    #     -------
    #     smoothed_soft_clusters : (num_chunks, num_speakers, num_clusters)-shaped array
    #     """

    #     num_chunks, num_speakers, num_clusters = soft_clusters.shape

    #     stitchable = np.zeros((num_chunks, num_speakers, num_chunks, num_speakers))
    #     for (c, s), (c_, s_), num_matching_frames in stitching_graph.edges(data="cost"):
    #         if c == c_:
    #             continue

    #         stitchable[c, s, c_, s_] = num_matching_frames
    #         stitchable[c_, s_, c, s] = num_matching_frames

    #     smoothed_soft_clusters = np.einsum(
    #         "ijkl,klm->ijm", stitchable, np.nan_to_num(soft_clusters, nan=0.0)
    #     ) / (np.einsum("ijkl->ij", stitchable)[:, :, None] + 1e-8)
    #     return smoothed_soft_clusters

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

    def apply(
        self,
        file: AudioFile,
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
        hook: Optional[Callable] = None,
    ) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        num_speakers : int, optional
            Number of speakers, when known.
        min_speakers : int, optional
            Minimum number of speakers. Has no effect when `num_speakers` is provided.
        max_speakers : int, optional
            Maximum number of speakers. Has no effect when `num_speakers` is provided.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        # if self.use_constrained_argmax and not CVXPY_IS_AVAILABLE:
        #     self.use_constrained_argmax = False
        #     warnings.warn()

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        num_speakers, min_speakers, max_speakers = self.set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        if self.expects_num_speakers and num_speakers is None:

            if "annotation" in file:
                num_speakers = len(file["annotation"].labels())

                if not self.training:
                    warnings.warn(
                        f"This pipeline expects the number of speakers (num_speakers) to be given. "
                        f"It has been automatically set to {num_speakers:d} based on reference annotation. "
                    )

            else:
                raise ValueError(
                    "This pipeline expects the number of speakers (num_speakers) to be given."
                )

        segmentations = self.get_segmentations(file)
        hook("segmentation", segmentations)
        #   shape: (num_chunks, num_frames, local_num_speakers)

        num_chunks, num_frames, _ = segmentations.data.shape

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            segmentations,
            frames=self._frames,
            **self.segmentation_onset,
        )
        hook("speaker_counting", count)
        #   shape: (num_frames, 1)
        #   dtype: int

        # if self.klustering == "OracleClustering":
        #     embedding_onset = self.segmentation_onset
        # else:
        #     embedding_onset = self.embedding_onset

        # binarize segmentation
        binarized_segmentations: SlidingWindowFeature = multi_binarize(
            segmentations,
            **self.segmentation_onset,
        )

        # keep track of inactive speakers
        # FIXME: inactive speakers should be obtained from segmentation_onset
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        if self.klustering == "OracleClustering":
            embeddings = None
        else:
            embeddings = self.get_embeddings(file, binarized_segmentations)
            hook("embeddings", embeddings)
            #   shape: (num_chunks, local_num_speakers, dimension)

        hard_clusters, soft_clusters = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
            file=file,  # <== for oracle clustering
            frames=self._frames,  # <== for oracle clustering
            hook=hook,
        )
        #   hard_clusters: (num_chunks, num_speakers)
        #   soft_clusters: (num_chunks, num_speakers, num_clusters)

        # reconstruct discrete diarization from raw hard clusters
        hard_clusters[inactive_speakers] = -2
        discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        hook("discrete_diarization", discrete_diarization)

        # if False:

        #     # turn soft cluster assignment into probabilities
        #     hook("soft_clusters/before_softmax", soft_clusters)
        #     soft_clusters = scipy.special.softmax(
        #         self.soft_temperature * soft_clusters, axis=2
        #     )
        #     hook("soft_clusters/after_softmax", soft_clusters)

        #     # compute stitching graph based on binarized segmentation
        #     stitching_graph = self.get_stitching_graph(binarized_segmentations)

        #     # smooth soft cluster assignment using stitching graph
        #     soft_clusters = self.soft_stitching(soft_clusters, stitching_graph)
        #     hook("soft_clusters/after_smoothing", soft_clusters)

        #     # hard_clusters = np.argmax(soft_clusters, axis=2)
        #     # hard_clusters[inactive_speakers] = -2
        #     # discrete_diarization = self.reconstruct(segmentations, hard_clusters, count)
        #     # hook("diarization/soft_stitching", discrete_diarization)

        # if self.use_constrained_argmax:
        #     hard_clusters = self.constrained_argmax(
        #         soft_clusters, binarized_segmentations
        #     )
        #     hard_clusters[inactive_speakers] = -2
        #     hook("diarization/hard_clusters/constrained", hard_clusters)
        #     discrete_diarization = self.reconstruct(segmentations, hard_clusters, count)
        #     hook("diarization/discrete/constrained", discrete_diarization)

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=0.0,
            min_duration_off=0.0,
        )
        diarization.uri = file["uri"]

        # when reference is available, use it to map hypothesized speakers
        # to reference speakers (this makes later error analysis easier
        # but does not modify the actual output of the diarization pipeline)
        if "annotation" in file:
            return self.optimal_mapping(file["annotation"], diarization)

        # when reference is not available, rename hypothesized speakers
        # to human-readable SPEAKER_00, SPEAKER_01, ...
        return diarization.rename_labels(
            {
                label: expected_label
                for label, expected_label in zip(diarization.labels(), self.classes())
            }
        )

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
