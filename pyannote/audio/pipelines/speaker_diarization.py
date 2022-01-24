# The MIT License (MIT)
#
# Copyright (c) 2021-2022 CNRS
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

import warnings
from typing import Callable, Optional, Text
from enum import IntEnum

import numpy as np
import torch
from scipy.spatial.distance import cdist

from pyannote.audio import Audio, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_devices,
)
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform

from .segmentation import SpeakerSegmentation
from .clustering import Clustering, NearestClusterAssignment
from .speaker_verification import PretrainedSpeakerEmbedding


class SpeakerStatus(IntEnum):

    # speaker speaks too little to extract embeddings
    LITTLE_SPEECH = 0

    # speaker speaks sufficiently to extract embeddings
    # but their embedding may be noisy due to overlap
    NOISY_SPEECH = 1

    # speaker speaks sufficiently to extract embeddings
    # from clean (i.e. single-speaker) speech
    CLEAN_SPEECH = 2


class SpeakerDiarization(SpeakerDiarizationMixin, Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    embedding : Model, str, or dict, optional
        Pretrained embedding model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    clustering : {"AgglomerativeClustering", "SpectralClustering"}, optional
        Defaults to "AgglomerativeClustering".
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
        embedding: PipelineModel = "pyannote/embedding",
        clustering: Text = "AgglomerativeClustering",
        expects_num_speakers: bool = False,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.embedding = embedding
        self.klustering = clustering
        self.expects_num_speakers = expects_num_speakers

        self.speaker_segmentation = SpeakerSegmentation(
            segmentation=segmentation, skip_conversion=True
        )
        self._frames: SlidingWindow = self.speaker_segmentation._frames

        (device,) = get_devices(needs=1)
        self._embedding = PretrainedSpeakerEmbedding(self.embedding, device=device)

        self._audio = Audio(sample_rate=self._embedding.sample_rate, mono=True)

        # minimum duration below which speaker embedding extraction will fail
        self._embedding_min_duration = (
            self._embedding.min_num_samples + 1
        ) / self._embedding.sample_rate

        try:
            Klustering = Clustering[clustering]
        except KeyError:
            raise ValueError(
                f'clustering must be one of [{", ".join(list(Clustering.__members__))}]'
            )
        self.clustering = Klustering.value(
            metric=self._embedding.metric,
            expects_num_clusters=self.expects_num_speakers,
        )
        self.nearest_cluster_assignment = NearestClusterAssignment(
            metric=self._embedding.metric, allow_reassignment=True
        )

        # minimum duration of clean speech to extract good enough speaker embedding
        duration = self.speaker_segmentation._segmentation.duration
        self.clean_embedding_min_duration = Uniform(
            self._embedding_min_duration, duration
        )

        # hyper-parameters used for post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    def default_parameters(self):
        raise NotImplementedError()

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

        hook = self.setup_hook(file, hook=hook)

        # validate provided num/min/max speakers
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

        # raw speaker segmentation (including overlapping speech)
        noisy_segmentation: SlidingWindow = self.speaker_segmentation(file)
        _, num_segmented_speakers = noisy_segmentation.data.shape

        # speaker segmentation where overlapping frames are zero'ed out
        clean_segmentation = SlidingWindowFeature(
            noisy_segmentation.data
            * (np.sum(noisy_segmentation.data, axis=-1, keepdims=True) == 1.0),
            noisy_segmentation.sliding_window,
        )

        # categorize segmented speakers based on the amount of (non-overlapping) speech:
        status = np.full(
            (num_segmented_speakers), SpeakerStatus.LITTLE_SPEECH, dtype=np.int
        )
        status[
            np.sum(noisy_segmentation.data, axis=0) * self._frames.step
            > self._embedding_min_duration
        ] = SpeakerStatus.NOISY_SPEECH
        status[
            np.sum(clean_segmentation.data, axis=0) * self._frames.step
            > self.clean_embedding_min_duration
        ] = SpeakerStatus.CLEAN_SPEECH

        # extract one embedding per speaker (unless they speak too little)
        clean_embeddings, noisy_embeddings = [], []
        speaker_support = Annotation(uri=file["uri"])

        for s, (clean_mask, noisy_mask) in enumerate(
            zip(clean_segmentation.data.T, noisy_segmentation.data.T,)
        ):

            if status[s] == SpeakerStatus.LITTLE_SPEECH:
                continue
            elif status[s] == SpeakerStatus.CLEAN_SPEECH:
                mask = clean_mask
                embeddings = clean_embeddings
            elif status[s] == SpeakerStatus.NOISY_SPEECH:
                mask = noisy_mask
                embeddings = noisy_embeddings

            # compute speaker temporal support
            first_frame, last_frame = np.nonzero(mask)[0][[0, -1]]
            support = Segment(
                self._frames[first_frame].start, self._frames[last_frame].end,
            )
            speaker_support[support, s] = s

            # extract waveform and binary mask
            waveform: torch.Tensor = Audio().crop(file, support, mode="pad")[0][None]
            mask: torch.Tensor = torch.from_numpy(
                mask[first_frame : last_frame + 1]
            ).float()[None]

            # compute embedding
            embeddings.append(self._embedding(waveform, masks=mask))

        # infer cannot link constraints from overlapping speaker support
        cannot_link = np.full(
            (num_segmented_speakers, num_segmented_speakers), 0, dtype=np.int
        )
        for (_, s), (_, t) in speaker_support.co_iter(speaker_support):
            cannot_link[s, t] = s != t

        # convert from list of (1, dimension) arrays to (num_speaker, dimension) array
        num_clean_embeddings = len(clean_embeddings)
        if num_clean_embeddings > 0:
            clean_embeddings = np.vstack(clean_embeddings)
        num_noisy_embeddings = len(noisy_embeddings)
        if num_noisy_embeddings > 0:
            noisy_embeddings = np.vstack(noisy_embeddings)

        clusters = np.full((num_segmented_speakers), -1, dtype=np.int)

        # cluster clean embeddings
        if num_clean_embeddings < 2:
            clusters[status == SpeakerStatus.CLEAN_SPEECH] = 0
            clusters[status == SpeakerStatus.NOISY_SPEECH] = 0
            num_clusters = 1
        else:
            clusters[status == SpeakerStatus.CLEAN_SPEECH] = self.clustering(
                clean_embeddings,
                num_clusters=num_speakers,
                min_clusters=min_speakers,
                max_clusters=max_speakers,
                # TODO: take cannot link constraint into account
            )

            num_clusters = np.max(clusters) + 1

        # assign embeddings to most similar cluster
        # (taking cannot link constraints into account)
        embeddings = np.full(
            (num_segmented_speakers, self._embedding.dimension), np.NAN
        )
        if num_clean_embeddings > 0:
            embeddings[status == SpeakerStatus.CLEAN_SPEECH] = clean_embeddings
        if num_noisy_embeddings > 0:
            embeddings[status == SpeakerStatus.NOISY_SPEECH] = noisy_embeddings
        clusters = self.nearest_cluster_assignment(
            embeddings, clusters, cannot_link=cannot_link
        )

        # build discrete diarization
        num_frames, _ = noisy_segmentation.data.shape
        discrete_diarization = np.zeros((num_frames, num_clusters + 1))
        for k in range(num_clusters):
            discrete_diarization[:, k] = np.sum(
                noisy_segmentation.data[:, clusters == k], axis=1
            )
        discrete_diarization[:, -1] = np.sum(
            noisy_segmentation.data[:, clusters == -1], axis=1
        )
        discrete_diarization = SlidingWindowFeature(
            discrete_diarization, noisy_segmentation.sliding_window
        )

        # convert to continuous diarization
        diarization = self.to_annotation(
            discrete_diarization,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )
        diarization.uri = file["uri"]

        # when reference is available, map cluster labels to actual speaker labels
        if "annotation" in file:
            return self.optimal_mapping(file["annotation"], diarization)

        # when reference is not available, sort speakers by decreasing speech duration
        # and rename them SPEAKER_0, SPEAKER_1, etc...
        return diarization.rename_labels(
            {
                label: expected_label
                for (label, _), expected_label in zip(
                    diarization.chart(), self.classes()
                )
            }
        )

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
