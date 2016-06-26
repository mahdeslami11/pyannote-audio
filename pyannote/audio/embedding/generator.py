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

import itertools
import numpy as np
import scipy.spatial.distance
from pyannote.generators.batch import BaseBatchGenerator
from pyannote.generators.fragment import RandomSegmentsPerLabel
from ..features.yaafe import YaafeBatchGenerator


class _YaafeTripletGenerator(object):

    def __init__(self, yaafe_feature_extractor, embedding, duration=3.2, per_label=40):
        super(_YaafeTripletGenerator, self).__init__()

        generator = RandomSegmentsPerLabel(
            duration=duration,
            per_label=per_label,
            yield_label=True)

        self.batch_generator = YaafeBatchGenerator(
            yaafe_feature_extractor,
            generator,
            batch_size=-1)

        self.embedding = embedding

    def get_shape(self):
        return self.batch_generator.get_shape()

    def signature(self):
        shape = self.batch_generator.get_shape()
        return [
            {'type': 'sequence', 'shape': shape},
            {'type': 'sequence', 'shape': shape},
            {'type': 'sequence', 'shape': shape}
        ]

    def from_protocol_item(self, protocol_item):

        for batch_sequences, batch_labels in self.batch_generator.from_protocol_item(protocol_item):

            batch_embeddings = self.embedding.transform(batch_sequences)
            batch_distances = scipy.spatial.distance.pdist(batch_embeddings, metric='euclidean')

            labels, unique_inverse = np.unique(batch_labels, return_inverse=True)

            for i, label in enumerate(labels):
                positives = np.where(unique_inverse == i)[0]
                negatives = np.where(unique_inverse != i)[0]

                # loop over all (anchor, positive) pairs for current label
                for anchor, positive in itertools.combinations(positives):

                    # find all negatives within the margin
                    d = batch_distances[anchor, positive]
                    within_margin = np.where(
                        batch_distances[anchor, negatives] < d + alpha)[0]

                    # choose one at random (if at least one exists)
                    if not within_margin:
                        continue
                    negative = negatives[np.random.choice(within_margin)]
                    yield batch_embeddings[anchor], batch_embeddings[positive], batch_embeddings[negative]


class YaafeTripletBatchGenerator(BaseBatchGenerator):
    """
    Parameters
    ----------
    extractor : YaafeFeatureExtractor
    embedding : TripletLossSequenceEmbedding
    """
    def __init__(self, extractor, embedding, duration=3.2, per_label=40, batch_size=32):
        generator = _YaafeTripletGenerator(extractor, embedding, duration=duration, per_label=per_label)
        super(YaafeTripletBatchGenerator, self).__init__(generator, batch_size=batch_size)

    def get_shape(self):
        return self.generator.get_shape()
