#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2019-2020 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

from typing import Optional
from typing import Text

import numpy as np
from pyannote.database import get_annotated
from pyannote.core import Segment
from pyannote.core import SlidingWindow
from pyannote.core import Timeline

from pyannote.core.utils.random import random_segment
from pyannote.core.utils.random import random_subsegment

from pyannote.audio.features import RawAudio

from .base import LabelingTask
from .base import LabelingTaskGenerator
from pyannote.audio.train.task import Task, TaskType, TaskOutput

from pyannote.audio.features.wrapper import Wrappable
from pyannote.database.protocol.protocol import Protocol
from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import Alignment


normalize = lambda wav: wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)


class OverlapDetectionGenerator(LabelingTaskGenerator):
    """Batch generator for training overlap detection

    Parameters
    ----------
    task : Task
        Task
    feature_extraction : Wrappable
        Describes how features should be obtained.
        See pyannote.audio.features.wrapper.Wrapper documentation for details.
    protocol : Protocol
    subset : {'train', 'development', 'test'}, optional
        Protocol and subset.
    resolution : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
        Defaults to `feature_extraction.sliding_window`
    alignment : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models that
        include the feature extraction step (e.g. SincNet) and therefore use a
        different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    snr_min, snr_max : float, optional
        Defines Signal-to-Overlap Ratio range in dB. Defaults to [0, 10].
    """

    def __init__(
        self,
        task: Task,
        feature_extraction: Wrappable,
        protocol: Protocol,
        subset: Text = "train",
        resolution: Optional[Resolution] = None,
        alignment: Optional[Alignment] = None,
        duration: float = 2.0,
        batch_size: int = 32,
        per_epoch: float = None,
        snr_min: float = 0,
        snr_max: float = 10,
    ):

        self.snr_min = snr_min
        self.snr_max = snr_max

        super().__init__(
            task,
            feature_extraction,
            protocol,
            subset=subset,
            resolution=resolution,
            alignment=alignment,
            duration=duration,
            batch_size=batch_size,
            per_epoch=per_epoch,
            waveform=True,
        )

    def samples(self):

        samples = super().samples()

        while True:

            sample = next(samples)
            if np.random.rand() < 0.5:
                pass

            else:

                # get random overlapping sequence
                overlap = next(samples)

                # select SNR at random
                snr = (
                    self.snr_max - self.snr_min
                ) * np.random.random_sample() + self.snr_min
                alpha = np.exp(-np.log(10) * snr / 20)

                sample["waveform"] += alpha * overlap["waveform"]

                # FIXME The call to hstack below is not correct because there
                # is a non-zero probability that the two samples contain speech
                # from a common speaker.
                # A solution would be to pass SlidingWindowFeature instances
                # instead of their (np.ndarray) "data" attribute and use their
                # newly added "labels" attribute to decide how to stack.
                # Sticking with SlidingWindowFeature instances all the way down
                # to the subsequent call to torch.tensor(...) would be even
                # better but for now this breaks (probably because of the
                # current behavior of SlidingWindowFeature.__iter__.
                sample["y"] = np.hstack([sample["y"], overlap["y"]])

            speaker_count = np.sum(sample["y"], axis=1, keepdims=True)
            sample["y"] = np.int64(speaker_count > 1)

            # run feature extraction (using sample["waveform"])
            sample["X"] = self.feature_extraction.crop(
                sample, Segment(0, self.duration), mode="center", fixed=self.duration
            )

            yield {"X": sample["X"], "y": sample["y"]}

    @property
    def specifications(self):
        return {
            "task": self.task,
            "X": {"dimension": self.feature_extraction.dimension},
            "y": {"classes": ["non_overlap", "overlap"]},
        }


class OverlapDetection(LabelingTask):
    """Train overlap detection

    Parameters
    ----------
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    """

    def get_batch_generator(
        self,
        feature_extraction: Wrappable,
        protocol: Protocol,
        subset: Text = "train",
        resolution: Optional[Resolution] = None,
        alignment: Optional[Resolution] = None,
    ) -> OverlapDetectionGenerator:
        """Get batch generator

        Parameters
        ----------
        feature_extraction : Wrappable
            Describes how features should be obtained.
            See pyannote.audio.features.wrapper.Wrapper documentation for details.
        protocol : Protocol
        subset : {'train', 'development', 'test'}, optional
            Protocol and subset.
        resolution : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        alignment : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.
        """
        return OverlapDetectionGenerator(
            self.task,
            feature_extraction,
            protocol,
            subset=subset,
            resolution=resolution,
            alignment=alignment,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
        )
