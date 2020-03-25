#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

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

"""Diarization"""

import torch
import torch.nn.functional as F
from typing import Optional
from typing import Type
from typing import Iterable
from typing import Dict
from typing import Text
import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from pyannote.core import Segment
from pyannote.core import SlidingWindowFeature
from pyannote.audio.train.task import Task, TaskType, TaskOutput
from pyannote.audio.features import FeatureExtraction
from pyannote.database.protocol.protocol import Protocol
from pyannote.audio.train.model import Resolution
from pyannote.audio.train.model import Alignment
from scipy.optimize import linear_sum_assignment


class DiarizationLoss:

    def __call__(self, input: torch.Tensor,
                       target: torch.Tensor,
                       weight: torch.Tensor = None,
                       mask: torch.Tensor = None):
        """

        Parameters
        ----------
        input : (batch_size, n_steps, n_speakers) torch.Tensor
        target : (batch_size, n_steps, n_speakers) torch.Tensor
        weight : (n_speakers, ) torch.Tensor, optional
        mask : (n_steps, ) torch.Tensor, optional

        Returns
        -------
        der : torch.Tensor

        """

        if weight is not None:
            msg = f'"weight" support is not implemented yet.'
            raise NotImplementedError(msg)

        if mask is not None:
            msg = f'"mask" support is not implemented yet.'
            raise NotImplementedError(msg)

        with torch.no_grad():
            # TODO add some kind of temperature factor
            K = input.transpose(2, 1) @ target
            mapped_target = []
            for k, y in zip(K, target):
                _, mapping = linear_sum_assignment(-k)
                mapped_target.append(y[:, mapping])
            mapped_target = torch.stack(mapped_target)

        return F.mse_loss(input, mapped_target, reduction='mean')

class DiarizationGenerator(LabelingTaskGenerator):
    """Batch generator for diarization

    Parameters
    ----------
    feature_extraction : FeatureExtraction
        Feature extraction
    protocol : Protocol
    subset : {'train', 'development', 'test'}, optional
        Protocol and subset.
    num_speakers : int, optional
        Maximum number of speakers in each audio chunk.
        Defaults to 4. TODO: estimate this automagically.
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
    exhaustive : bool, optional
        Ensure training files are covered exhaustively (useful in case of
        non-uniform label distribution).
    step : `float`, optional
        Ratio of audio chunk duration used as step between two consecutive
        audio chunks. Defaults to 0.1. Has not effect when exhaustive is False.
    mask : str, optional
        When provided, current_file[mask] is used by the loss function to weigh
        samples.

    """

    def __init__(self,
                 feature_extraction: FeatureExtraction,
                 protocol: Protocol,
                 num_speakers : int = None,
                 subset: Text = 'train',
                 resolution: Optional[Resolution] = None,
                 alignment: Optional[Alignment] = None,
                 duration: float = 2.0,
                 batch_size: int = 32,
                 per_epoch: float = None,
                 mask: Text = None):

        super().__init__(feature_extraction=feature_extraction,
                         protocol=protocol,
                         subset=subset,
                         resolution=resolution,
                         alignment=alignment,
                         duration=duration,
                         batch_size=batch_size,
                         per_epoch=per_epoch,
                         mask=mask)

        if num_speakers is None:
            num_speakers = 4
            # TODO. estimate this number automatically

        self.num_speakers = num_speakers

    def crop_y(self, y: SlidingWindowFeature,
                     segment: Segment) -> np.ndarray:
        """Extract y for specified segment

        Parameters
        ----------
        y : `pyannote.core.SlidingWindowFeature`
            Output of `initialize_y` above.
        segment : `pyannote.core.Segment`
            Segment for which to obtain y.

        Returns
        -------
        cropped_y : (n_samples, dim) `np.ndarray`
            y for specified `segment`
        """

        cropped =  y.crop(segment, mode=self.alignment, fixed=self.duration)

        # find indices corresponding to active speakers
        active = np.sum(cropped, axis=0) > 1
        num_active = np.sum(active)

        if num_active > self.num_speakers:
            msg = (
                f'Current chunk has {num_active} active speakers, while '
                f'maximum number of speakers is set to {self.num_speakers}'
            )
            raise ValueError(msg)

        # filter inactive speakers (i.e. only keep active ones)
        filtered = cropped[:, active]

        # in case there are less active speakers that maximum number of speakers,
        # add fake inactive speakers
        if num_active < self.num_speakers:
            num_inactive = self.num_speakers - num_active
            filtered = np.pad(filtered, ((0, 0), (0, num_inactive)))

        # randomize speakers
        np.random.shuffle(filtered.T)

        return filtered

    @property
    def specifications(self) -> Dict:
        """Task & sample specifications

        Returns
        -------
        specs : `dict`
            ['task'] (`pyannote.audio.train.Task`) : task
            ['X']['dimension'] (`int`) : features dimension
            ['y']['classes'] (`list`) : list of classes
        """

        specs = dict()
        specs['X'] = {'dimension': self.feature_extraction.dimension}
        specs['task'] = Task(type=TaskType.MULTI_LABEL_CLASSIFICATION,
                             output=TaskOutput.SEQUENCE)
        specs['y'] = {'classes': [f'speaker_{i+1:02d}'
                                  for i in range(self.num_speakers)]}

        return specs


class Diarization(LabelingTask):
    """Train diarization

    Parameters
    ----------
    duration : float, optional
        Duration of audio chunks. Defaults to 2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Force total audio duration per epoch, in days.
        Defaults to total duration of protocol subset.
    """

    def __init__(self, duration: float = 2.0,
                       batch_size: int = 32,
                       per_epoch: float = None,
                       num_speakers: int = None):

        super().__init__(duration=duration,
                         batch_size=batch_size,
                         per_epoch=per_epoch)

        self.num_speakers = num_speakers

    def get_batch_generator(self, feature_extraction: FeatureExtraction,
                                  protocol: Protocol,
                                  subset: Text = 'train',
                                  resolution: Optional[Resolution] = None,
                                  alignment: Optional[Alignment] = None) -> DiarizationGenerator:
        """
        Parameters
        ----------
        feature_extraction : FeatureExtraction
        protocol : Protocol
        subset : {'train', 'development'}, optional
            Defaults to 'train'.
        resolution : `pyannote.core.SlidingWindow`, optional
            Override `feature_extraction.sliding_window`. This is useful for
            models that include the feature extraction step (e.g. SincNet) and
            therefore output a lower sample rate than that of the input.
        alignment : {'center', 'loose', 'strict'}, optional
            Which mode to use when cropping labels. This is useful for models
            that include the feature extraction step (e.g. SincNet) and
            therefore use a different cropping mode. Defaults to 'center'.
        """
        return DiarizationGenerator(
            feature_extraction,
            protocol, subset=subset,
            resolution=resolution,
            alignment=alignment,
            duration=self.duration,
            per_epoch=self.per_epoch,
            batch_size=self.batch_size,
            num_speakers=self.num_speakers)

    def on_train_start(self):
        self.task_ = self.model_.task
        self.loss_func_ = DiarizationLoss()
