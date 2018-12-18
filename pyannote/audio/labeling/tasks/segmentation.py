#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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

"""Speaker change detection"""

import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from .base import TASK_MULTI_LABEL_CLASSIFICATION
import scipy.signal


class SegmentationGenerator(LabelingTaskGenerator):
    """Batch generator for training multi-task segmentation

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction
    speech : bool, optional
        Add speech dimension.
    overlap : bool, optional
        Add overlapping speech dimension
    change : bool, optional
        Add speaker change dimensions
    collar : float, optional
        Duration of (speaker change) neighborhood, in seconds. Default to 100ms (0.1).
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.

    Usage
    -----
    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/mfcc')

    # instantiate batch generator
    >>> batches = SegmentationGenerator(precomputed)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('Etape.SpeakerDiarization.TV')

    # iterate over training set
    >>> for batch in batches(protocol, subset='train'):
    >>>     # batch['X'] is a (batch_size, n_samples, n_features) numpy array
    >>>     # batch['y'] is a (batch_size, n_samples, n_tasks) numpy array
    >>>     pass
    """

    def __init__(self, feature_extraction, speech=True, overlap=True,
                 change=True, collar=0.100, **kwargs):

        super(SegmentationGenerator, self).__init__(
            feature_extraction, **kwargs)

        self.speech = speech
        self.overlap = overlap
        self.change = change
        self.collar = collar

        # number of samples in collar
        self.collar_ = 2 * \
            self.feature_extraction.sliding_window.durationToSamples(collar)

    def postprocess_y(self, Y):
        """Generate labels for multi-task segmentation

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by `pyannote.core.utils.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, 1) numpy.ndarray

        See also
        --------
        `pyannote.core.utils.numpy.one_hot_encoding`
        """

        n_samples, n_speakers = Y.shape

        # replace NaNs by 0s
        Y = np.nan_to_num(Y)

        y = []

        # Speech Activity Detection
        count = np.sum(Y, axis=1, keepdims=True)
        if self.speech:
            y_speech = 1 * (count > 0)
            y.append(y_speech)

        # Overlap Speech Detection
        if self.overlap:
            y_overlap = 1 * (count > 1)
            y.append(y_overlap)

        # Speaker Change Detection
        if self.change:

            # append (half collar) empty samples at the beginning/end
            expanded_Y = np.vstack([
                np.zeros(((self.collar_ + 1) // 2 , n_speakers),
                         dtype=Y.dtype),
                Y,
                np.zeros(((self.collar_ + 1) // 2 , n_speakers),
                         dtype=Y.dtype)])

            # stride trick. data[i] is now a sliding window of collar length
            # centered at time step i.
            data = np.lib.stride_tricks.as_strided(expanded_Y,
                shape=(n_samples, n_speakers, self.collar_),
                strides=(Y.strides[0], Y.strides[1], Y.strides[0]))

            # y_change[i] = 1 if more than one speaker are speaking in the
            # corresponding window. 0 otherwise
            y_change = 1 * (np.sum(np.sum(data, axis=2) > 0, axis=1) > 1)
            y_change = y_change.reshape(-1, 1)
            y.append(y_change)

        return np.hstack(y)


class Segmentation(LabelingTask):
    """Train multi-task segmentation

    Parameters
    ----------
    speech : bool, optional
        Add speech dimension.
    overlap : bool, optional
        Add overlapping speech dimension
    change : bool, optional
        Add speaker change dimensions
    collar : float, optional
        Duration of neighborhood, in seconds. Default to 100ms (0.1).
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    per_epoch : float, optional
        Total audio duration per epoch, in days.
        Defaults to one day (1).
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.

    Usage
    -----
    >>> task = Segmentation()

    # precomputed features
    >>> from pyannote.audio.features import Precomputed
    >>> precomputed = Precomputed('/path/to/features')

    # model architecture
    >>> from pyannote.audio.labeling.models import StackedRNN
    >>> model = StackedRNN(precomputed.dimension, task.n_classes)

    # evaluation protocol
    >>> from pyannote.database import get_protocol
    >>> protocol = get_protocol('Etape.SpeakerDiarization.TV')

    # train model using protocol training set
    >>> for epoch, model in task.fit_iter(model, precomputed, protocol):
    ...     pass

    """

    def __init__(self, speech=True, overlap=True, change=True, collar=0.100, **kwargs):
        super(Segmentation, self).__init__(**kwargs)
        self.speech = speech
        self.overlap = overlap
        self.change = change
        self.collar = collar

    def get_batch_generator(self, feature_extraction):
        return SegmentationGenerator(
            feature_extraction, speech=self.speech, overlap=self.overlap,
            change=self.change, collar=self.collar, duration=self.duration,
            batch_size=self.batch_size, per_epoch=self.per_epoch,
            parallel=self.parallel)

    @property
    def task_type(self):
        return TASK_MULTI_LABEL_CLASSIFICATION

    @property
    def n_classes(self):
        return sum([self.speech, self.overlap, self.change])

    @property
    def labels(self):
        labels = []
        if self.speech:
            labels.append('speech')
        if self.overlap:
            labels.append('overlap')
        if self.change:
            labels.append('change')
        return labels
