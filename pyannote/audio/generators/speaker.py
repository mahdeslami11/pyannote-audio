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
from pyannote.generators.fragment import SlidingLabeledSegments
from pyannote.generators.fragment import RandomSegmentPairs
from pyannote.generators.batch import FileBasedBatchGenerator


class SpeakerEmbeddingBatchGenerator(YaafeMixin, FileBasedBatchGenerator):

    def __init__(self, feature_extractor, duration=3.2, normalize=False,
                 step=0.8, batch_size=32):

        segment_generator = SlidingLabeledSegments(duration=duration,
                                                   step=step)
        super(SpeakerEmbeddingBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step
        self.normalize = normalize

    def signature(self):
        return [
            {'type': 'sequence', 'shape': self.get_shape()},
            {'type': 'label'}
        ]


class SpeakerPairsBatchGenerator(YaafeMixin, FileBasedBatchGenerator):

    def __init__(self, feature_extractor, duration=3.2, normalize=False,
                 per_label=40, batch_size=32):

        pair_generator = RandomSegmentPairs(duration=duration,
                                            per_label=per_label,
                                            yield_label=False)
        super(SpeakerPairsBatchGenerator, self).__init__(
            pair_generator, batch_size=batch_size)

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.normalize = normalize
        self.per_label = per_label

    def signature(self):
        shape = self.get_shape()
        return [
            (
                {'type': 'sequence', 'shape': shape},
                {'type': 'sequence', 'shape': shape},
            ),
            {'type': 'boolean'},
        ]
