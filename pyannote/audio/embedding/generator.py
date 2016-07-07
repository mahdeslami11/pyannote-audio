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
from scipy.spatial.distance import pdist, squareform
from pyannote.generators.batch import BaseBatchGenerator
from pyannote.generators.fragment import SlidingLabeledSegments
from ..features.yaafe import YaafeFileBasedBatchGenerator


class YaafeTripletLossGenerator(object):
    """Yaafe sequence generator for triplet loss

    Parameter
    ---------
    extractor: YaafeFeatureExtractor
        Yaafe feature extraction (e.g. YaafeMFCC())
    file_generator: iterable
        File generator.
    embedding: TripletLossSequenceEmbedding
        Triplet loss sequence embedding.
    duration: float, optional
        Sequence duration. Defaults to 3 seconds.
    overlap: float, optional
        Sequence overlap ratio. Defaults to 0 (no overlap).
    normalize: boolean, optional
        When True, normalize sequence (z-score). Defaults to False.
    n_labels: int, optional
        Number of labels per group. Defaults to 40.
    per_label: int, optional
        Number of samples per label. Defaults to 40.
    embedding_batch_size: int, optional
        Batch size to use for embedding. Defaults to 32.
    """

    def __init__(self, extractor, file_generator, embedding,
                 duration=3.0, overlap=0.0, normalize=False,
                 n_labels=40, per_label=40, embedding_batch_size=32):
        super(YaafeTripletLossGenerator, self).__init__()

        self.extractor = extractor
        self.file_generator = file_generator
        self.embedding = embedding
        self.duration = duration
        self.overlap = overlap
        self.normalize = normalize
        self.n_labels = n_labels
        self.per_label = per_label
        self.embedding_batch_size = embedding_batch_size

        fragment_generator = SlidingLabeledSegments(
            duration=self.duration,
            step=(1 - self.overlap) * self.duration)

        self.batch_sequence_generator_ = YaafeFileBasedBatchGenerator(
            self.extractor,
            fragment_generator,
            batch_size=-1,
            normalize=self.normalize)

        self.triplet_generator_ = self.iter_triplets()

    def iter_triplets(self):

        # generate all sequences first
        X, y = [], []
        for batch_sequences, batch_labels in self.batch_sequence_generator_(self.file_generator):
            X.append(batch_sequences)
            y.append(batch_labels)
        X = np.vstack(X)
        y = np.hstack(y)

        # unique labels
        unique, y, counts = np.unique(y, return_inverse=True, return_counts=True)
        n = len(unique)

        # infinite loop
        while True:

            # shuffle labels
            shuffled_labels = np.random.choice(n, size=n, replace=False)

            # take them n_labels per n_labels
            for k in range(n / self.n_labels):
                from_label = k * self.n_labels
                to_label = (k+1) * self.n_labels
                labels = shuffled_labels[from_label: to_label]

                # select min(per_label, count) sequences
                # at random for each label
                indices = []
                per_label = []
                for label in labels:
                    per_label.append(min(self.per_label, counts[label]))
                    i = np.random.choice(
                        np.where(y == label)[0],
                        size=per_label[-1],
                        replace=True)
                    indices.append(i)
                indices = np.hstack(indices)
                per_label = np.hstack([[0], np.cumsum(per_label)])

                # pre-compute distances
                sequences = X[indices]
                embeddings = self.embedding.transform(
                    sequences, batch_size=self.embedding_batch_size)
                distances = squareform(pdist(embeddings, metric='euclidean'))

                for i in range(self.n_labels):

                    positives = list(range(per_label[i], per_label[i+1]))
                    negatives = list(range(per_label[i])) + list(range(per_label[i+1], per_label[-1]))

                    # loop over all (anchor, positive) pairs for current label
                    for anchor, positive in itertools.combinations(positives, 2):

                        # find all negatives within the margin
                        d = distances[anchor, positive]
                        within_margin = np.where(
                            distances[anchor, negatives] < d + self.embedding.margin)[0]

                        # choose one at random (if at least one exists)
                        if len(within_margin) < 1:
                            continue
                        negative = negatives[np.random.choice(within_margin)]

                        yield [sequences[anchor], sequences[positive], sequences[negative]], 1

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.triplet_generator_)

    def get_shape(self):
        return self.batch_sequence_generator_.get_shape()

    def signature(self):
        shape = self.batch_sequence_generator_.get_shape()
        return (
            [
                {'type': 'sequence', 'shape': shape},
                {'type': 'sequence', 'shape': shape},
                {'type': 'sequence', 'shape': shape}
            ],
            {'type': 'boolean'}
         )


class YaafeTripletLossBatchGenerator(BaseBatchGenerator):
    """Yaafe batch generator for triplet loss

    Parameter
    ---------
    extractor: YaafeFeatureExtractor
        Yaafe feature extraction (e.g. YaafeMFCC())
    embedding: TripletLossSequenceEmbedding
        Triplet loss sequence embedding.
    duration: float, optional
        Sequence duration. Defaults to 3 seconds.
    overlap: float, optional
        Sequence overlap ratio. Defaults to 0 (no overlap).
    normalize: boolean, optional
        When True, normalize sequence (z-score). Defaults to False.
    n_labels: int, optional
        Number of labels per batch. Defaults to 40.
    per_label: int, optional
        Number of samples per label. Defaults to 40.
    batch_size: int, optional
        Number of sequences per batch. Defaults to 32
    """

    def __init__(self, file_generator, extractor, embedding,
                 duration=3.0, overlap=0.0, normalize=False,
                 n_labels=40, per_label=40,
                 batch_size=32):

        self.triplet_generator_ = YaafeTripletLossGenerator(
            extractor, file_generator, embedding,
            duration=duration, overlap=overlap, normalize=normalize,
            n_labels=n_labels, per_label=per_label,
            embedding_batch_size=batch_size)

        super(YaafeTripletLossBatchGenerator, self).__init__(
            self.triplet_generator_, batch_size=batch_size)

    def get_shape(self):
        return self.triplet_generator_.get_shape()
