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

from pyannote.generators.batch import BaseBatchGenerator
from pyannote.generators.fragment import RandomSegmentTriplets
from ..features.yaafe import YaafeMFCC, YaafeFrame
import numpy as np


class TripletLossBatchGenerator(BaseBatchGenerator):
    """
    Parameters
    ----------
    batch_size: int, optional
    duration: float, optional
    per_label: int, optional
    embedding: SequenceEmbedding, optional
    """

    def __init__(self, duration=3.2, batch_size=1000, per_label=40,
                 embedding=None):

        # fragment generation
        fragment_generator = RandomSegmentTriplets(duration=duration,
                                                   per_label=per_label,
                                                   yield_label=False)
        super(TripletLossBatchGenerator, self).__init__(fragment_generator,
                                                        batch_size=batch_size)

        # feature extraction
        self.feature_extractor = YaafeMFCC(e=False, De=False, DDe=False, coefs=11, D=False, DD=False)
        self.fe_frame = YaafeFrame(blockSize=self.feature_extractor.block_size,
                                   stepSize=self.feature_extractor.step_size,
                                   sampleRate=self.feature_extractor.sample_rate)
        self.fe_n = self.fe_frame.durationToSamples(duration)
        self.X_ = {}

        # triplet selection
        self.embedding = embedding

    def get_shape(self):
        return (self.fe_n, self.feature_extractor.dimension())

    # defaults to features pre-computing
    def preprocess(self, protocol_item, identifier=None):
        wav, _, _ = protocol_item
        if not identifier in self.X_:
            self.X_[identifier] = self.feature_extractor(wav)
        return protocol_item

    def process(self, fragment, signature=None, identifier=None):
        if signature['type'] == 'segment':
            i0, _ = self.fe_frame.segmentToRange(fragment)
            return self.X_[identifier][i0:i0+self.fe_n]

        if signature['type'] == 'boolean':
            return fragment

    def postprocess(self, batch, signature=None):
        if self.embedding is not None:
            batch = self.triplet_selection(batch)

        return (batch, np.ones((batch[0].shape[0], 1)))

    def triplet_selection(self, batch):

        anchor_batch, positive_batch, negative_batch = batch

        # extract embeddings
        anchor_embedding = self.embedding.transform(anchor_batch)
        positive_embedding = self.embedding.transform(positive_batch)
        negative_embedding = self.embedding.transform(negative_batch)

        # compute positive and negative distance
        positive_distance = np.sum((anchor_embedding - positive_embedding) ** 2, axis=-1)
        negative_distance = np.sum((anchor_embedding - negative_embedding) ** 2, axis=-1)

        # only keep triplets with semi-hard negative sample
        semi_hard_negative = np.where(
            (positive_distance < negative_distance) *
            (negative_distance < positive_distance + self.embedding.alpha))

        return [anchor_batch[semi_hard_negative],
                positive_batch[semi_hard_negative],
                negative_batch[semi_hard_negative]]
