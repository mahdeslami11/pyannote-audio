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


import numpy as np
import itertools
import scipy.signal
from ..features.yaafe import YaafeFileBasedBatchGenerator
from ..features.utils import get_wav_duration
from pyannote.generators.fragment import SlidingSegments
from pyannote.core import Segment, Timeline
from ..embedding.models import SequenceEmbedding
from pyannote.core.util import pairwise


class Segmentation(object):

    def __init__(self, embedding,
                 feature_extractor, duration=3.0, normalize=False,
                 precision=0.05, min_duration=1.0, threshold=1.,
                 batch_size=32):
        super(Segmentation, self).__init__()
        self.embedding = embedding
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor

        self.duration = duration
        self.precision = precision
        self.normalize = normalize
        self.threshold = threshold

        self.min_duration = min_duration

        fragment_generator = SlidingSegments(duration=self.duration,
                                             step=self.precision)
        self.batch_generator_ = YaafeFileBasedBatchGenerator(
            self.feature_extractor, fragment_generator,
            batch_size=-1, normalize=self.normalize)

    def embed(self, wav, wav_duration):

        uem = Timeline(segments=[Segment(0, get_wav_duration(wav))])
        item = wav, uem, uem
        batch_sequences = next(self.batch_generator_.from_file(item))

        X = self.embedding.transform(
            batch_sequences, batch_size=self.batch_size, verbose=1)

        return X

    def diff(self, X):

        n = X.shape[0]
        # TODO check that precision does evenly divide duration
        k = int(self.duration / self.precision)

        d = np.array(
            [np.sum((X[i] - X[i + k])**2) for i in range(n - k)])
        t = np.array(
            [self.duration + self.precision * i for i in range(n - k)])

        return t, d

    def peaks(self, t, d, threshold=None):

        if threshold is None:
            threshold = self.threshold

        order = 1
        if self.min_duration >= self.precision:
            order = int(self.min_duration / self.precision)
        maxima = scipy.signal.argrelmax(d, order=order)

        t_ = t[maxima]
        d_ = d[maxima]

        # only keep high enough local maxima
        high_maxima = np.where(d_ > threshold)

        return t_[high_maxima]

    def _timeline(self, from_peaks, wav_duration):

        # create list of segment boundaries
        # do not forget very first and last boundaries
        boundaries = np.hstack([[0.0], from_peaks, [wav_duration]])

        # return list of segments
        return Timeline([Segment(*p) for p in pairwise(boundaries)])

    def iter_steps(self, wav):
        wav_duration = get_wav_duration(wav)
        yield wav_duration
        X = self.embed(wav, wav_duration)
        yield X
        t, d = self.diff(X)
        yield t, d
        peaks = self.peaks(t, d)
        yield peaks
        yield self._timeline(peaks, wav_duration)

    def __call__(self, wav):
        for result in self.iter_steps(wav):
            pass
        return result
