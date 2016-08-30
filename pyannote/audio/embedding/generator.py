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
from ..generators.yaafe import YaafeMixin
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import SlidingLabeledSegments


class LabeledSequencesBatchGenerator(YaafeMixin, FileBasedBatchGenerator):
    """(X_batch, y_batch) batch generator

    Yields batches made of sequences obtained using a sliding window over the
    coverage of the reference. Heterogeneous segments (i.e. containing more
    than one label) are skipped.

    Parameters
    ----------
    feature_extractor : YaafeFeatureExtractor
    duration: float, optional
    step: float, optional
        Duration and step of sliding window (in seconds).
        Default to 3.2 and 0.8.
    normalize: boolean, optional
        Normalize (zscore) feature sequences

    Returns
    -------
    X_batch : (batch_size, n_samples, n_features) numpy array
        Batch of feature sequences
    y_batch : (batch_size, ) numpy array
        Batch of corresponding labels

    Usage
    -----
    >>> batch_generator = LabeledSequencesBatchGenerator(feature_extractor)
    >>> for X_batch, y_batch in batch_generator.from_file(current_file):
    ...     # do something with
    """

    def __init__(self, feature_extractor, duration=3.0, normalize=False,
                 step=0.75, batch_size=32):

        self.feature_extractor = feature_extractor
        self.duration = duration
        self.step = step
        self.normalize = normalize

        segment_generator = SlidingLabeledSegments(duration=duration,
                                                   step=step)
        super(LabeledSequencesBatchGenerator, self).__init__(
            segment_generator, batch_size=batch_size)

    def signature(self):
        return (
            {'type': 'sequence', 'shape': self.get_shape()},
            {'type': 'label'}
        )


class TripletGenerator(object):
    """Triplet generator for triplet loss sequence embedding

    Generates ([Xa, Xp, Xn], 1) tuples where
      * Xa is the anchor sequence (e.g. by speaker S)
      * Xp is the positive sequence (also uttered by speaker S)
      * Xn is the negative sequence (uttered by a different speaker)

    and such that d(f(Xa), f(Xn)) < d(f(Xa), f(Xp)) + margin where
      * f is the current state of the embedding network (being optimized)
      * d is the euclidean distance
      * margin is the triplet loss margin (e.g. 0.2, typically)

    Parameters
    ----------
    extractor: YaafeFeatureExtractor
        Yaafe feature extraction (e.g. YaafeMFCC instance)
    file_generator: iterable
        File generator (the training set, typically)
    embedding: TripletLossSequenceEmbedding
        Triplet loss sequence embedding (currently being optimized)
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
    batch_size: int, optional
    forward_batch_size: int, optional
        Batch size to use for embedding. Defaults to 32.
    """

    def __init__(self, extractor, file_generator, embedding,
                 duration=3.0, overlap=0.0, normalize=False,
                 n_labels=40, per_label=40,
                 batch_size=32, forward_batch_size=32):

        self.extractor = extractor
        self.file_generator = file_generator
        self.embedding = embedding
        self.duration = duration
        self.overlap = overlap
        self.normalize = normalize
        self.n_labels = n_labels
        self.per_label = per_label
        self.forward_batch_size = forward_batch_size

        self.batch_sequence_generator_ = LabeledSequencesBatchGenerator(
            self.extractor,
            duration=self.duration,
            normalize=self.normalize,
            step=(1 - self.overlap) * self.duration,
            batch_size=-1)

        self.triplet_generator_ = self.iter_triplets()

        super(TripletGenerator, self).__init__()


    def iter_triplets(self):

        # pre-generate all labeled sequences (from the whole training set)
        # this might be huge in memory
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

                # per_label[i] contains the actual number of examples
                # available for ith label -- as it may actually be smaller
                # than the requested self.per_label for small classes (e.g.
                # for speakers without only a few seconds of speech)
                per_label = []

                # indices contains the list of indices of all sequences
                # to be used for later triplet selection
                indices = []

                for label in labels:

                    # number of available sequences for current label
                    per_label.append(min(self.per_label, counts[label]))

                    # randomly choose this many sequences
                    # from the set of available sequences
                    i = np.random.choice(
                        np.where(y == label)[0],
                        size=per_label[-1],
                        replace=True)

                    # append indices of selected sequences
                    indices.append(i)


                # after this line, per_label[i] will contain the position of
                # the first sequence of ith label so that the range
                # per_label[i]: per_label[i+1] points to the indices
                # corresponding to all sequences from ith label
                per_label = np.hstack([[0], np.cumsum(per_label)])

                # turn indices into a 1-dimensional numpy array.
                # combined with (above) per_label, it can be used
                # to get all indices of sequences from a given label
                indices = np.hstack(indices)

                # pre-compute pairwise distances d(f(X), f(X')) between every
                # pair (X, X') of selected sequences, where f is the current
                # state of the embedding being optimized, and d is the
                # euclidean distance

                # selected sequences
                sequences = X[indices]
                # their embeddings (using current state of embedding network)
                embeddings = self.embedding.transform(
                    sequences, batch_size=self.forward_batch_size)
                # pairwise euclidean distances
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
                        # TODO / add an option to choose the most difficult one
                        negative = negatives[np.random.choice(within_margin)]

                        yield [sequences[anchor], sequences[positive], sequences[negative]], 1

                        # FIXME -- exit this loop when an epoch has ended
                    # FIXME -- exit this loop when an epoch has ended
                # FIXME -- exit this loop when an epoch has ended

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        return next(self.triplet_generator_)

    def get_shape(self):
        return self.batch_sequence_generator_.get_shape()

    def signature(self):
        shape = self.get_shape()
        return (
            [
                {'type': 'sequence', 'shape': shape},
                {'type': 'sequence', 'shape': shape},
                {'type': 'sequence', 'shape': shape}
            ],
            {'type': 'boolean'}
        )


class TripletBatchGenerator(BaseBatchGenerator):
    """Pack batches out of TripletGenerator

    Parameters
    ----------
    batch_size: int, optional
        Batch size

    Returns
    -------
    X_batch : (batch_size, n_samples, n_features) numpy array
        Batch of feature sequences
    y_batch : (batch_size, ) numpy array
        Batch of corresponding labels

    See also
    --------
    TripletGenerator
    """

    def __init__(self, extractor, file_generator, embedding,
                 duration=3.0, overlap=0.0, normalize=False,
                 n_labels=40, per_label=40,
                 batch_size=32, forward_batch_size=32):

        self.triplet_generator_ = TripletGenerator(
            extractor, file_generator, embedding,
            duration=duration, overlap=overlap, normalize=normalize,
            n_labels=n_labels, per_label=per_label,
            forward_batch_size=forward_batch_size)

        super(TripletBatchGenerator, self).__init__(
            self.triplet_generator_, batch_size=batch_size)

    def signature(self):
        return self.triplet_generator_.signature()

    def get_shape(self):
        return self.triplet_generator_.get_shape()
