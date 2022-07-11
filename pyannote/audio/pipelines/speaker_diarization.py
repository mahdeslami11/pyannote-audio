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

import math
import random
import warnings
from itertools import combinations, zip_longest
from typing import Callable, List, Optional, Tuple

import networkx as nx
import numpy as np

# import scipy.special
import torch
from einops import rearrange
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.distance import to_condensed
from pyannote.metrics.diarization import GreedyDiarizationErrorRate

# from pyannote.pipeline.parameter import Uniform, Categorical
from pyannote.pipeline.parameter import Uniform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.clustering import Clustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SegmentationConstraints,
    SpeakerDiarizationMixin,
    get_devices,
    get_model,
)
from pyannote.audio.utils.signal import binarize

# try:
#     import cvxpy as cv
#     CVXPY_IS_AVAILABLE = True
# except ImportError:
#     CVXPY_IS_AVAILABLE = False


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return zip_longest(*args, fillvalue=fillvalue)


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
    embedding_exclude_overlap : bool, optional
        Exclude overlapping speech regions when extracting embeddings.
        Defaults (False) to use the whole speech.
    clustering : str, optional
        Clustering algorithm. See pyannote.audio.pipelines.clustering.Clustering
        for available options. Defaults to "HiddenMarkovModelClustering".
    expects_num_speakers : bool, optional
        Defaults to False.


    Hyper-parameters
    ----------------
    segmentation_onset : float
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
        embedding_exclude_overlap: bool = False,
        embedding_batch_size: int = 32,
        clustering: str = "HiddenMarkovModelClustering",
        expects_num_speakers: bool = False,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.segmentation_batch_size = segmentation_batch_size
        self.segmentation_step = segmentation_step

        self.embedding = embedding
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap

        self.klustering = clustering
        self.expects_num_speakers = expects_num_speakers

        seg_device, emb_device = get_devices(needs=2)

        self._model: Model = get_model(segmentation)
        self._model.to(seg_device)

        self._segmentation = Inference(
            self._model,
            duration=self._model.specifications.duration,
            step=self.segmentation_step * self._model.specifications.duration,
            skip_aggregation=True,
            batch_size=self.segmentation_batch_size,
        )
        self._frames: SlidingWindow = self._segmentation.model.introspection.frames
        self.segmentation_onset = Uniform(0.1, 0.9)

        if self.klustering == "OracleClustering":
            metric = "not_applicable"

        else:
            self._embedding = PretrainedSpeakerEmbedding(
                self.embedding, device=emb_device
            )
            self._audio = Audio(sample_rate=self._embedding.sample_rate, mono=True)
            metric = self._embedding.metric

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

        self.constraint_threshold = Uniform(0.0, 0.1)

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

    def get_constraints(
        self, file, binary_segmentations: SlidingWindowFeature
    ) -> nx.Graph:
        """

        Parameters
        ----------
        file : AudioFile
        binary_segmentations : (num_chunks, num_frames, num_speakers)-shaped SlidingWindowFeature
            Binary segmentation.

        Returns
        -------
        """

        # initial set of (good enough) constraints
        constraints = SegmentationConstraints.from_segmentation(
            binary_segmentations, identifier="original", cannot=True, must=True
        )
        constraints.remove_constraints(
            lambda n1, n2, data: data.get("permutation_cost", 0.0)
            > self.constraint_threshold
        )
        # TODO: should we propagate here?

        # initialize inference with half-overlapping chunks
        duration = self._model.specifications.duration
        segmentation = Inference(
            self._model,
            duration=duration,
            step=0.5 * duration,
            skip_aggregation=True,
            batch_size=self.segmentation_batch_size,
        )

        audio: Audio = self._model.audio
        file_duration: float = audio.get_duration(file)

        # generate sequence of non-overlapping chunks covering the whole file
        non_overlapping_chunks = SlidingWindow(
            start=0.0, duration=duration, step=duration
        )
        shuffling: List[Tuple[int, Segment]] = list(
            enumerate(non_overlapping_chunks(Segment(0, file_duration)))
        )
        # [(0, Segment(0, 5)), [1, Segment(5, 10), ...]] if duration == 5.

        for k in range(5):

            # shuffle non-overlapping chunks and keep track of order
            random.shuffle(shuffling)
            shuffled_indices, shuffled_chunks = zip(*shuffling)

            # map chunks of the shuffled audio to correspond chunk in the original audio
            half_overlapping_chunks = SlidingWindow(
                start=0.0, duration=duration, step=0.5 * duration
            )
            mapping = {
                non_overlapping_chunks[i]: half_overlapping_chunks[2 * j]
                for i, j in enumerate(shuffled_indices)
            }

            # build shuffled audio (as the concatenation of shuffled non-overlapping chunks)
            shuffled_waveform = torch.hstack(
                [
                    audio.crop(file, chunk, duration=duration)[0]
                    for chunk in shuffled_chunks
                ]
            )

            # apply segmentation on shuffled audio with half-overlapping chunks
            shuffled_segmentations = binarize(
                segmentation(
                    {"waveform": shuffled_waveform, "sample_rate": audio.sample_rate}
                ),
                onset=self.segmentation_onset,
            )

            # gather (good enough) must link constraints
            shuffled_constraints = SegmentationConstraints.from_segmentation(
                shuffled_segmentations,
                identifier=f"shuffled_#{k + 1:02d}",
                cannot=False,
                must=True,
            )
            shuffled_constraints.remove_constraints(
                lambda n1, n2, data: data.get("permutation_cost", 0.0)
                > self.constraint_threshold
            )

            # # propagate "cannot-link" constraints
            shuffled_constraints.propagate(cannot=False, must=True)
            # # TODO: investigate propagation of "must-link" constraints

            # remove nodes that do not exist in the original segmentation
            shuffled_constraints.remove_nodes_from(
                list(
                    filter(
                        lambda node: node[0] not in mapping,
                        shuffled_constraints.nodes(),
                    )
                )
            )

            # back to the original time reference
            unshuffled_constraints = nx.relabel_nodes(
                shuffled_constraints,
                {
                    (chunk, speaker_idx): (mapping[chunk], speaker_idx)
                    for chunk, speaker_idx in shuffled_constraints.nodes()
                },
                copy=True,
            )

            constraints = constraints.augment(unshuffled_constraints, copy=False)

        constraints = constraints.propagate(cannot=True, must=True, copy=False)

        # rename (chunk, speaker_idx) to (chunk_idx, speaker_idx)
        chunks = binary_segmentations.sliding_window
        num_chunks, _, num_speakers = binary_segmentations.data.shape
        mapping = {
            (chunk, speaker_idx): (
                round((chunk.start - chunks.start) / chunks.step),
                speaker_idx,
            )
            for chunk, speaker_idx in constraints.nodes()
        }
        nx.relabel_nodes(constraints, mapping, copy=False)

        n = num_chunks * num_speakers
        matrix = np.zeros((to_condensed(n, n, n - 1),), dtype=np.int8)
        for (c, s), (C, S), link in constraints.edges(data="link"):

            # TODO: fix this at the source...
            if (c, s) == (C, S):
                continue

            matrix[to_condensed(n, c * num_speakers + s, C * num_speakers + S)] = (
                1 if link == "must" else -1
            )

        return matrix

    def constraint_to_matrix(
        self,
        constraints: List[List[Tuple[Segment, int]]],
        binary_segmentations: SlidingWindowFeature,
    ) -> np.ndarray:
        """Convert constraint to condensed boolean matrix

        Parameters
        ----------
        constraints : list of list of (chunk, speaker_idx) tuple
            List of must link or cannot link constraints.
        segmentations : (num_chunks, num_frames, num_speakers)-shaped SlidingWindowFeature
            Binary segmentation

        Returns
        -------
        matrix :
            Condensed (à la pdist) constraint matrix.

        """

        num_chunks, _, num_speakers = binary_segmentations.data.shape
        chunks = binary_segmentations.sliding_window

        n = num_chunks * num_speakers
        matrix = np.zeros((to_condensed(n, n, n - 1),), dtype=bool)

        for constraint in constraints:
            for (chunk, speaker_idx), (other_chunk, other_speaker_idx) in combinations(
                constraint, r=2
            ):
                chunk_idx = round((chunk.start - chunks.start) / chunks.step)
                other_chunk_idx = round(
                    (other_chunk.start - chunks.start) / chunks.step
                )

                matrix[
                    to_condensed(
                        n,
                        chunk_idx * num_speakers + speaker_idx,
                        other_chunk_idx * num_speakers + other_speaker_idx,
                    )
                ] = True

        return matrix

    def get_embeddings(
        self,
        file,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
    ):
        """Extract embeddings for each (chunk, speaker) pair

        Parameters
        ----------
        file : AudioFile
        binary_segmentations : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
            Binarized segmentation.
        exclude_overlap : bool, optional
            Exclude overlapping speech regions when extracting embeddings.
            In case non-overlapping speech is too short, use the whole speech.

        Returns
        -------
        embeddings : (num_chunks, num_speakers, dimension) array
        """

        duration = binary_segmentations.sliding_window.duration
        num_chunks, num_frames, _ = binary_segmentations.data.shape

        if exclude_overlap:
            # minimum number of samples needed to extract an embedding
            # (a lower number of samples would result in an error)
            min_num_samples = self._embedding.min_num_samples

            # corresponding minimum number of frames
            num_samples = duration * self._embedding.sample_rate
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)

            # zero-out frames with overlapping speech
            clean_frames = 1.0 * (
                np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            )
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window,
            )

        else:
            min_num_frames = -1
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data, binary_segmentations.sliding_window
            )

        def iter_waveform_and_mask():
            for (chunk, masks), (_, clean_masks) in zip(
                binary_segmentations, clean_segmentations
            ):
                # chunk: Segment(t, t + duration)
                # masks: (num_frames, local_num_speakers) np.ndarray

                waveform, _ = self._audio.crop(
                    file,
                    chunk,
                    duration=duration,
                    mode="pad",
                )
                # waveform: (1, num_samples) torch.Tensor

                # mask may contain NaN (in case of partial stitching)
                masks = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                clean_masks = np.nan_to_num(clean_masks, nan=0.0).astype(np.float32)

                for mask, clean_mask in zip(masks.T, clean_masks.T):
                    # mask: (num_frames, ) np.ndarray

                    if np.sum(clean_mask) > min_num_frames:
                        used_mask = clean_mask
                    else:
                        used_mask = mask

                    yield waveform[None], torch.from_numpy(used_mask)[None]
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

        embedding_batches = np.vstack(embedding_batches)

        return rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)

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
            onset=self.segmentation_onset,
            frames=self._frames,
        )
        hook("speaker_counting", count)
        #   shape: (num_frames, 1)
        #   dtype: int

        # binarize segmentation
        binary_segmentations: SlidingWindowFeature = binarize(
            segmentations,
            onset=self.segmentation_onset,
            initial_state=False,
        )

        # derive {cannot-link, must-link} constraints from binary segmentation

        constraints = self.get_constraints(file, binary_segmentations)

        hook("constraints", constraints)

        if self.klustering == "OracleClustering":
            embeddings = None
        else:
            embeddings = self.get_embeddings(
                file,
                binary_segmentations,
                exclude_overlap=self.embedding_exclude_overlap,
            )
            hook("embeddings", embeddings)
            #   shape: (num_chunks, local_num_speakers, dimension)

        hard_clusters, soft_clusters = self.clustering(
            embeddings=embeddings,
            segmentations=binary_segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
            constraints=constraints,
            # cannot_link=cannot_link,
            # must_link=must_link,
            file=file,  # <== for oracle clustering
            frames=self._frames,  # <== for oracle clustering
        )
        #   hard_clusters: (num_chunks, num_speakers)
        #   soft_clusters: (num_chunks, num_speakers, num_clusters)

        # reconstruct discrete diarization from raw hard clusters

        # keep track of inactive speakers
        inactive_speakers = np.sum(binary_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        hard_clusters[inactive_speakers] = -2
        hook("hard_clusters", hard_clusters)

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
        #     stitching_graph = self.get_stitching_graph(binary_segmentations)

        #     # smooth soft cluster assignment using stitching graph
        #     soft_clusters = self.soft_stitching(soft_clusters, stitching_graph)
        #     hook("soft_clusters/after_smoothing", soft_clusters)

        #     # hard_clusters = np.argmax(soft_clusters, axis=2)
        #     # hard_clusters[inactive_speakers] = -2
        #     # discrete_diarization = self.reconstruct(segmentations, hard_clusters, count)
        #     # hook("diarization/soft_stitching", discrete_diarization)

        # if self.use_constrained_argmax:
        #     hard_clusters = self.constrained_argmax(
        #         soft_clusters, binary_segmentations
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
