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

import numpy as np
import torch

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
from .clustering import Clustering
from .speaker_verification import PretrainedSpeakerEmbedding


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

        # __ HANDLE EXPECTED NUMBER OF SPEAKERS ________________________________________

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

        # speaker segmentation
        segmentation: SlidingWindow = self.speaker_segmentation(file)

        # mask overlapping speech regions
        mask = SlidingWindowFeature(
            segmentation.data
            * (np.sum(segmentation.data, axis=-1, keepdims=True) == 1.0),
            segmentation.sliding_window,
        )

        # extract one embedding per segmented speaker
        embeddings = []
        for s, active in enumerate(mask.data.T):

            # compute speaker temporal support
            first_active_frame, last_active_frame = np.nonzero(active)[0][[0, -1]]
            active_support = Segment(
                self._frames[first_active_frame].start,
                self._frames[last_active_frame].end,
            )

            # TODO. infer cannot-link constraints by intersecting speaker support

            # read waveform and mask
            speaker_waveform: torch.Tensor = Audio().crop(
                file, active_support, mode="pad"
            )[0][None]
            speaker_mask: torch.Tensor = torch.from_numpy(
                active[first_active_frame : last_active_frame + 1]
            ).float()[None]

            # compute embedding
            embeddings.append(self._embedding(speaker_waveform, masks=speaker_mask))

        # perform clustering
        clusters = self.clustering(
            np.vstack(embeddings),
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
        )

        # build discrete diarization
        num_clusters = np.max(clusters) + 1
        num_frames, _ = segmentation.data.shape
        discrete_diarization = np.zeros((num_frames, num_clusters))
        for k in range(num_clusters):
            discrete_diarization[:, k] = np.sum(
                segmentation.data[:, clusters == k], axis=1
            )
        discrete_diarization = SlidingWindowFeature(
            discrete_diarization, segmentation.sliding_window
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
