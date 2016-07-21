#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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

import yaml
import numpy as np
import scipy.signal

from pyannote.core import Segment, Annotation
from pyannote.core.util import pairwise

from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import TwinSlidingSegments

from ..embedding.models import SequenceEmbedding

from ..features.yaafe import YaafeMFCC

from ..generators.base import YaafeMixin
from ..features.utils import get_wav_duration




class EmbeddingSegmenter(YaafeMixin, FileBasedBatchGenerator):
    """

    Parameters
    ----------
    embedding : SequenceEmbedding
    feature_extractor : YaafeFeatureExtractor
    duration : float, optional
        Defaults to 3.
    normalize : boolean, optional
        Defaults to False.

    """

    PRECISION = 0.1
    MIN_DURATION = 1.0
    ALPHA = 1.0

    def __init__(self, embedding,
                 feature_extractor, duration=3.0, normalize=False,
                 precision=PRECISION, min_duration=MIN_DURATION, alpha=ALPHA):

        # initialize twin segments generator
        generator = TwinSlidingSegments(
            duration=duration,
            step=precision,
            gap=0.0)

        super(EmbeddingSegmenter, self).__init__(
            generator,
            batch_size=-1)

        # feature sequence
        self.feature_extractor = feature_extractor
        self.duration = duration
        self.normalize = normalize

        # embedding
        self.embedding = embedding

        # tunable hyper-parameters
        self.precision = precision
        self.min_duration = min_duration
        self.alpha = alpha

    @classmethod
    def from_disk(cls, embedding_dir,
                  precision=PRECISION,
                  min_duration=MIN_DURATION,
                  alpha=ALPHA):
        """
        Parameters
        ----------
        embedding_dir : str
        precision : float, optional
        min_duration : float, optional
        alpha : float, optional
        """

        # load best embedding
        architecture_yml = embedding_dir + '/architecture.yml'
        weights_h5 = embedding_dir + '/best.accuracy.h5'
        embedding = SequenceEmbedding.from_disk(
            architecture_yml, weights_h5)

        # feature sequence
        config_yml = embedding_dir + '/config.yml'
        with open(config_yml, 'r') as fp:
            config = yaml.load(fp)
        feature_extractor = YaafeMFCC(**config['feature_extraction']['mfcc'])
        duration = config['feature_extraction']['duration']
        normalize = config['feature_extraction']['normalize']

        return cls(embedding,
                   feature_extractor, duration=duration, normalize=normalize,
                   precision=precision, min_duration=min_duration, alpha=alpha)

    def signature(self):
        shape = self.get_shape()
        return (
            {'type': 'timestamp'},
            {'type': 'sequence', 'shape': shape},
            {'type': 'sequence', 'shape': shape}
        )

    def postprocess_sequence(self, batch):
        return self.embedding.transform(batch)

    def _diff(self, wav):
        """

        Parameter
        ---------
        wav : str
            Path to wav audio file

        Returns
        -------
        t : np.array
            Timestamps
        delta : np.array
            Left-to-right Euclidean distance at each timestamp
        """
        current_file = wav, None, None
        t, left, right = next(self.from_file(current_file))
        delta = np.sqrt(np.sum((left - right) ** 2, axis=-1))
        return t, delta

    def _find_peaks(self, delta, min_duration=None):
        """Find peaks indices

        Parameter
        ---------
        delta : np.array
            Left-to-right Euclidean distance
        min_duration : float, optional
            Minimum duration between two peaks, in seconds.
            Defaults to instance `min_duration` attribute.

        Returns
        -------
        indices : np.array
            Peaks indices
        """

        if min_duration is None:
            min_duration = self.min_duration

        order = int(np.rint(min_duration / self.precision))
        indices = scipy.signal.argrelmax(delta, order=order)[0]

        return indices

    def _filter_peaks(self, t, delta, indices, alpha=None):
        """Filter out smaller peaks

        Parameters
        ----------
        t : np.array
            Timestamps
        delta : np.array
            Left-to-right Euclidean distance
        indices : np.array
            Peaks indices
        alpha : float, optional
            Filter out peaks smaller than μ + α x 𝜎
            Defaults to instance `alpha` attribute

        Returns
        -------
        peaks : np.array
            Peaks timestamps
        """
        if alpha is None:
            alpha = self.alpha

        threshold = np.mean(delta) + alpha * np.std(delta)
        peaks = np.array([
            t[i] for i in indices if delta[i] > threshold
        ])

        return peaks

    def _build_annotation(self, wav, peaks):
        """

        Parameters
        ----------
        wav : str
            Path to wav audio file
        peaks : np.array
            Peaks timestamps

        Returns
        -------
        segmentation : pyannote.core.Annotation
            Temporal segmentation
        """
        boundaries = np.hstack([[0], peaks, [get_wav_duration(wav)]])
        segmentation = Annotation()
        for i, (start, end) in enumerate(pairwise(boundaries)):
            segment = Segment(start, end)
            segmentation[segment] = i
        return segmentation

    def apply(self, wav):

        t, delta = self._diff(wav)
        indices = self._find_peaks(delta)
        peaks = self._filter_peaks(t, delta, indices)
        segmentation = self._build_annotation(wav, peaks)

        return segmentation
