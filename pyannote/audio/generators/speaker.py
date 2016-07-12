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


from .base import YaafeMixin
from pyannote.core import SlidingWindowFeature
from pyannote.generators.fragment import SlidingLabeledSegments
from pyannote.generators.batch import FileBasedBatchGenerator
from scipy.stats import zscore


class SpeakerEmbeddingBatchGenerator(YaafeMixin, FileBasedBatchGenerator):

    def __init__(self, feature_extractor, duration=3.2, normalize=False,
                 step=0.8, batch_size=32):

        segment_generator = SlidingLabeledSegments(duration=duration, step=step)
        super(SpeakerEmbeddingBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step
        self.normalize = normalize

    def get_shape(self):
        n_samples = self.feature_extractor.sliding_window().samples(self.duration, mode='center')
        dimension = self.feature_extractor.dimension()
        return (n_samples, dimension)

    def signature(self):
        return [
            {'type': 'sequence', 'shape': self.get_shape()},
            {'type': 'label'}
        ]

    def preprocess(self, current_file, identifier=None):
        """Pre-compute file-wise X"""

        return self.yaafe_preprocess(
            current_file, identifier=identifier)

    # defaults to extracting frames centered on segment
    def process_segment(self, segment, signature=None, identifier=None):
        """Extract X subsequence"""

        duration = signature.get('duration', None)

        X = self.preprocessed_['X'][identifier].crop(
            segment, mode='center', fixed=duration)
        if self.normalize:
            X = zscore(X, axis=0)

        return X
