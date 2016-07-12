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
from pyannote.generators.fragment import SlidingSegments
from pyannote.generators.batch import FileBasedBatchGenerator
from scipy.stats import zscore
import numpy as np


class SpeechActivityDetectionBatchGenerator(YaafeMixin, FileBasedBatchGenerator):

    def __init__(self, feature_extractor, duration=3.2, normalize=False,
                 step=0.8, batch_size=32):

        segment_generator = SlidingSegments(duration=duration, step=step, source='uem')
        super(SpeechActivityDetectionBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step
        self.normalize = normalize

    def signature(self):

        shape = self.yaafe_get_shape()

        return [
            {'type': 'sequence', 'shape': shape},
            {'type': 'sequence', 'shape': (shape[0], 1)}
        ]

    def preprocess(self, current_file, identifier=None):
        """Pre-compute file-wise X and y"""

        current_file = self.yaafe_preprocess(
            current_file, identifier=identifier)

        if identifier in self.preprocessed_.setdefault('y', {}):
            return current_file

        X = self.preprocessed_['X'][identifier]
        sw = X.sliding_window
        n_samples = X.getNumber()

        y = -np.ones((n_samples + 1, 1), dtype=np.int8)
        # 1 => speech / 0 => non speech / -1 => unknown

        wav, uem, reference = current_file
        coverage = reference.get_timeline().coverage()

        for gap in coverage.gaps(uem):
            indices = sw.crop(gap, mode='loose')
            y[indices] = 0

        for segment in coverage:
            indices = sw.crop(segment, mode='loose')
            y[indices] = 1

        y = SlidingWindowFeature(y[:-1], sw)
        self.preprocessed_['y'][identifier] = y

        return current_file

    # defaults to extracting frames centered on segment
    def process_segment(self, segment, signature=None, identifier=None):
        """Extract X and y subsequences"""

        X = self.yaafe_process_segment(
            segment, signature=signature, identifier=identifier)

        duration = signature.get('duration', None)

        y = self.preprocessed_['y'][identifier].crop(
            segment, mode='center', fixed=duration)

        return [X, y]
