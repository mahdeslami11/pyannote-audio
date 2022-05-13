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

import numpy as np
import torch
from einops import rearrange
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.clustering import GaussianHiddenMarkovModel
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_devices,
    get_model,
)
from pyannote.audio.utils.permutation import mae_cost_func
from pyannote.audio.utils.signal import binarize


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
    segmentation_stitching: bool, optional
        Whether to stitch local segmentation. Defaults to False.
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 32.
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    embedding_filtering: bool, optional
        Whether to filter out multi-speaker frames before extracting embeddings.
        Defaults to False.
    embedding_batch_size : int, optional
        Batch size used for speaker embedding. Defaults to 32.
    expects_num_speakers : bool, optional
        Defaults to False.

    Hyper-parameters
    ----------------

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
        segmentation_stitching: bool = False,
        segmentation_batch_size: int = 32,
        embedding: PipelineModel = "pyannote/embedding",
        embedding_filtering: bool = False,
        embedding_batch_size: int = 32,
        expects_num_speakers: bool = False,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.segmentation_step = segmentation_step
        self.segmentation_stitching = segmentation_stitching
        self.embedding = embedding
        self.embedding_filtering = embedding_filtering
        self.expects_num_speakers = expects_num_speakers
        self.segmentation_batch_size = segmentation_batch_size

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

        self._embedding = PretrainedSpeakerEmbedding(self.embedding, device=emb_device)
        self.embedding_batch_size = embedding_batch_size or len(
            model.specifications.classes
        )

        self._audio = Audio(sample_rate=self._embedding.sample_rate, mono=True)

        self.clustering = GaussianHiddenMarkovModel(
            metric=self._embedding.metric, expects_num_clusters=expects_num_speakers
        )

        # hyper-parameters

        # hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.01, 0.99)
        self.offset = Uniform(0.01, 0.99)

        if self.segmentation_stitching:
            self.stitching_threshold = Uniform(0.01, 0.99)

        # hyper-parameters used for post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

    @property
    def CACHED_SEGMENTATION(self):
        return "cache/segmentation"

    def get_segmentations(self, file):
        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, local_num_speakers)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file)

        return segmentations

    def stitch_match_func(
        self, this: np.ndarray, that: np.ndarray, cost: float
    ) -> bool:
        return (
            np.any(this > self.onset)
            and np.any(that > self.onset)
            and cost < self.stitching_threshold
        )

    @property
    def CACHED_EMBEDDINGS(self):
        if self.segmentation_stitching:
            if self.embedding_filtering:
                return "cache/embedding/stitched+filtered"
            else:
                return "cache/embedding/stitched"
        else:
            if self.embedding_filtering:
                return "cache/embedding/filtered"
            else:
                return "cache/embedding"

    def get_embeddings(self, file, segmentations: SlidingWindowFeature):
        # apply embedding model (only if needed)
        # output shape is (num_chunks, local_num_speakers, dimension)

        if self.training and self.CACHED_EMBEDDINGS in file:
            embeddings = file[self.CACHED_EMBEDDINGS]

        else:

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
                    masks = np.nan_to_num(masks, nan=0.0)

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

            embeddings = rearrange(
                np.vstack(embedding_batches), "(c s) d -> c s d", c=len(segmentations)
            )
            # (num_chunks, local_num_speakers, dimension)

        # cache embeddings when training
        if self.training:
            file[self.CACHED_EMBEDDINGS] = embeddings

        return embeddings

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

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            segmentations,
            onset=self.onset,
            offset=self.offset,
            frames=self._frames,
        )
        hook("speaker_counting", count)
        #   shape: (num_frames, 1)
        #   dtype: int

        if self.segmentation_stitching:
            segmentations = Inference.stitch(
                segmentations,
                frames=self._frames,
                cost_func=mae_cost_func,
                match_func=self.stitch_match_func,
            )
            hook("stitched_segmentation", segmentations)

        num_chunks, num_frames, local_num_speakers = segmentations.data.shape
        duration = segmentations.sliding_window.duration

        if self.embedding_filtering:
            # filter out overlapping speaker frames
            # fs[c, f, s] = 1 iff speaker s is the only active speaker at frame f of chunk c
            binarized_segmentations: SlidingWindowFeature = binarize(
                segmentations, onset=self.onset, offset=self.offset, initial_state=False
            )
            filtered_segmentations = SlidingWindowFeature(
                (
                    binarized_segmentations.data
                    * (
                        np.sum(binarized_segmentations.data, axis=-1, keepdims=True)
                        == 1.0
                    )
                ).astype(np.float32),
                binarized_segmentations.sliding_window,
            )
            embeddings = self.get_embeddings(file, filtered_segmentations)
            hook("filtered_segmentation", filtered_segmentations)
        else:
            embeddings = self.get_embeddings(file, segmentations)

        hook("embeddings", embeddings)
        #   shape: (num_chunks, local_num_speakers, dimension)

        # focus on center of each chunk
        step = segmentations.sliding_window.step
        ratio = 0.5 * (duration - step) / duration
        center_segmentations = Inference.trim(segmentations, warm_up=(ratio, ratio))

        # number of frames during which speakers are active
        # in the center of the chunk
        num_active_frames: np.ndarray = np.sum(
            center_segmentations.data > self.onset, axis=1
        )
        priors = num_active_frames / (
            np.sum(num_active_frames, axis=1, keepdims=True) + 1e-8
        )
        #   shape: (num_chunks, local_num_speakers)

        clusters = self.clustering(
            embeddings,
            priors=priors,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
        )

        if hasattr(self.clustering, "debug_"):
            hook("clustering.debug_", self.clustering.debug_)

        # mark inactive speakers as such (cluster = -2)
        num_active_frames: np.ndarray = np.sum(segmentations.data > self.onset, axis=1)
        clusters[num_active_frames == 0] = -2

        # build final aggregated speaker activations

        num_clusters = np.max(clusters) + 1
        clustered_segmentations = np.NAN * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )

        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(clusters, segmentations)
        ):

            # cluster is (local_num_speakers, )-shaped
            # segmentation is (num_frames, local_num_speakers)-shaped
            for k in np.unique(cluster):
                if k == -2:
                    continue

                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )

        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )
        hook("clustering", clustered_segmentations)

        discrete_diarization = self.to_diarization(clustered_segmentations, count)
        hook("diarization", discrete_diarization)

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

        diarization.uri = file["uri"]

        if "annotation" in file:
            return self.optimal_mapping(file["annotation"], diarization)

        return diarization.rename_labels(
            {
                label: expected_label
                for label, expected_label in zip(diarization.labels(), self.classes())
            }
        )

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)


class SpeakerDiarizationWithOracleSegmentation(SpeakerDiarization):
    """Speaker diarization pipeline with oracle segmentation"""

    def oracle_segmentation(self, file) -> SlidingWindowFeature:
        file_duration: float = self._audio.get_duration(file)

        reference: Annotation = file["annotation"]
        labels = reference.labels()
        if self._oracle_num_speakers > len(labels):
            num_missing = self._oracle_num_speakers - len(labels)
            labels += [
                f"FakeSpeakerForOracleSegmentationInference{i:d}"
                for i in range(num_missing)
            ]

        window = SlidingWindow(
            start=0.0, duration=self._oracle_duration, step=self._oracle_step
        )

        segmentations = []
        for chunk in window(Segment(0.0, file_duration)):
            chunk_segmentation: SlidingWindowFeature = reference.discretize(
                chunk,
                resolution=self._frames,
                labels=labels,
                duration=self._oracle_duration,
            )
            # keep `self._oracle_num_speakers`` most talkative speakers
            most_talkative_index = np.argsort(-np.sum(chunk_segmentation, axis=0))[
                : self._oracle_num_speakers
            ]

            segmentations.append(chunk_segmentation[:, most_talkative_index])

        return SlidingWindowFeature(np.float32(np.stack(segmentations)), window)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._oracle_duration: float = self._segmentation.duration
        self._oracle_step: float = self._segmentation.step
        self._oracle_num_speakers: int = len(
            self._segmentation.model.specifications.classes
        )
        self._segmentation = self.oracle_segmentation
