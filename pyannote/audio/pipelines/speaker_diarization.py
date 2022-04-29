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
from hmmlearn.hmm import GaussianHMM
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.distance import cdist
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform

from pyannote.audio import Audio, Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_devices,
    get_model,
)


def batchify(iterable, batch_size: int = 32, fillvalue=None):
    """Batchify iterable"""
    # batchify('ABCDEFG', 3) --> ['A', 'B', 'C']  ['D', 'E', 'F']  [G, ]
    args = [iter(iterable)] * batch_size
    return itertools.zip_longest(*args, fillvalue=fillvalue)


class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
    """Speaker diarization pipeline

    TODO: add local stitching (better embeddings because they are extracted on more data)
    TODO: add overlap-aware masks (better embeddings because they are extracted on clean data)

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    expects_num_speakers : bool, optional
        Defaults to False.
    segmentation_batch_size : int, optional
        Batch size used for speaker segmentation. Defaults to 32.
    embedding_batch_size : int, optional
        Batch size used for speaker embedding. Defaults to 32.

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
        embedding: PipelineModel = "pyannote/embedding",
        expects_num_speakers: bool = False,
        segmentation_batch_size: int = 32,
        embedding_batch_size: int = 32,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.embedding = embedding
        self.expects_num_speakers = expects_num_speakers
        self.segmentation_batch_size = segmentation_batch_size

        seg_device, emb_device = get_devices(needs=2)

        model: Model = get_model(segmentation)
        model.to(seg_device)
        self._segmentation = Inference(
            model, skip_aggregation=True, batch_size=self.segmentation_batch_size
        )
        self._frames: SlidingWindow = self._segmentation.model.introspection.frames

        self._embedding = PretrainedSpeakerEmbedding(self.embedding, device=emb_device)
        self.embedding_batch_size = embedding_batch_size or len(
            model.specifications.classes
        )

        self._audio = Audio(sample_rate=self._embedding.sample_rate, mono=True)

        # hyper-parameters

        # hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.01, 0.99)
        self.offset = Uniform(0.01, 0.99)

        # hyper-parameters used for post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

    CACHED_SEGMENTATION = "cache/segmentation/inference"
    CACHED_EMBEDDINGS = "cache/embedding/inference"

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

    def get_embeddings(self, file, segmentations):
        # apply embedding model (only if needed)
        # output shape is (num_chunks, local_num_speakers, dimension)

        if self.training and self.CACHED_EMBEDDINGS in file:
            embeddings = file[self.CACHED_EMBEDDINGS]

        else:

            def iter_waveform_and_mask():
                for chunk, masks in segmentations:
                    # chunk: Segment(t, t + duration)
                    # masks: (num_frames, local_num_speakers) np.ndarray

                    waveform, _ = self._audio.crop(file, chunk, mode="pad")
                    # waveform: (1, num_samples) torch.Tensor

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

        num_chunks, num_frames, local_num_speakers = segmentations.data.shape

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

        embeddings = self.get_embeddings(file, segmentations)
        hook("embeddings", embeddings)
        #   shape: (num_chunks, local_num_speakers, dimension)

        # unit-normalize embeddings
        with np.errstate(divide="ignore", invalid="ignore"):
            embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)

        # focus on step-long center of each chunk
        duration = segmentations.sliding_window.duration
        step = segmentations.sliding_window.step
        ratio = 0.5 * (duration - step) / duration
        center_segmentations = Inference.trim(segmentations, warm_up=(ratio, ratio))

        # number of frames during which speakers are active
        # in the center of the chunk
        num_active_frames: np.ndarray = np.sum(
            center_segmentations.data > self.onset, axis=1
        )
        # (num_chunks, local_num_speakers)

        # index of most active speaker in the center of the chunk
        most_active_speaker = np.argmax(num_active_frames, axis=1)
        # (num_chunks, )

        most_active_embedding = embeddings[range(num_chunks), most_active_speaker]
        # (num_chunks, dimension)

        # keep non-NANs most active embeddings
        valid_most_active_embedding = ~np.any(np.isnan(most_active_embedding), axis=1)

        X = most_active_embedding[valid_most_active_embedding]

        hmm = GaussianHMM(
            n_components=num_speakers,
            covariance_type="diag",
            n_iter=100,
            implementation="log",
        ).fit(X)

        distances_to_centroids = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"), hmm.means_, metric="cosine"
            ),
            "(c s) k -> c s k",
            c=num_chunks,
        )
        clusters = np.argmin(distances_to_centroids, axis=-1)
        # (num_chunks, local_num_speakers)

        # mark inactive speakers as such (cluster = -2)
        num_active_frames: np.ndarray = np.sum(segmentations.data > self.onset, axis=1)
        clusters[num_active_frames == 0] = -2

        # build final aggregated speaker activations

        num_clusters = num_speakers
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
