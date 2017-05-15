#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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

import warnings
import numpy as np

from pyannote.core import SlidingWindow, SlidingWindowFeature
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import TwinSlidingSegments
from pyannote.audio.generators.periodic import PeriodicFeaturesMixin
from pyannote.audio.embedding.utils import cdist


class Segmentation(PeriodicFeaturesMixin, FileBasedBatchGenerator):
    """Segmentation based on sequence embedding

    Computes the euclidean distance between the embeddings of two
    (left and right) sliding windows.

    Parameters
    ----------
    sequence_embedding : SequenceEmbedding
        Pre-trained sequence embedding.
    feature_extractor : YaafeFeatureExtractor
        Yaafe feature extractor
    duration : float, optional
    step : float, optional
        Sliding window duration and step (in seconds).
        Defaults to 1 second window with 100ms step.

    Usage
    -----
    >>> sequence_embedding = SequenceEmbedding.from_disk('architecture_yml', 'weights.h5')
    >>> feature_extractor = YaafeFeatureExtractor(...)
    >>> segmentation = Segmentation(sequence_embedding, feature_extractor)
    >>> predictions = segmentation.apply(current_file)
    >>> segmentation = Peak().apply(predictions)

    See also
    --------
    pyannote.audio.embedding.models.SequenceEmbedding
    pyannote.audio.signal.Peak

    """
    def __init__(self, sequence_embedding, feature_extractor,
                 duration=1.000, step=0.100, distance='angular'):

        # feature sequence
        self.feature_extractor = feature_extractor

        # sequence embedding
        self.sequence_embedding = sequence_embedding

        # (left vs. right) sliding windows
        self.duration = duration
        self.step = step
        generator = TwinSlidingSegments(duration=duration, step=step)

        self.distance = distance

        super(Segmentation, self).__init__(generator, batch_size=32)

    def signature(self):
        shape = self.shape
        return (
            {'type': 'scalar'},
            {'type': 'ndarray', 'shape': shape},
            {'type': 'ndarray', 'shape': shape}
        )

    def postprocess_sequence(self, batch):
        return self.sequence_embedding.transform(batch)

    @classmethod
    def apply_precomputed(cls, embedding, window=None, metric='angular'):
        """

        Parameters
        ----------
        embedding : SlidingWindowFeature
        window : SlidingWindow, optional
            When provided, aggregate embedding over this window
        metric : str, optional

        """

        # if window is provided, aggregate embeddings using this very window
        if window is not None:
            window = SlidingWindow(start=window.start,
                                   duration=window.duration,
                                   step=window.step,
                                   end=embedding.getExtent().end)
            fX = np.vstack([np.mean(embedding.crop(segment), axis=0)
                            for segment in window])
            embedding = SlidingWindowFeature(fX, window)

        # if not, assume that embeddings are already aggregated
        else:
            window = embedding.sliding_window

        # make sure window duration is a multiple of window step
        n = window.duration / window.step
        if n != int(round(n)):
            warnings.warn('duration / step is not integer. rounding.')
        n = int(round(n))

        # number of windows
        N = embedding.data.shape[0]

        # pairwise distances
        y = np.array([
            cdist(embedding[i, np.newaxis], embedding[i + n, np.newaxis],
                  metric=metric)[0, 0] for i in range(N - n)])

        window = SlidingWindow(duration=2 * window.duration,
                               step=window.step,
                               start=window.start)

        return SlidingWindowFeature(y, window)

    def apply(self, current_file):
        """Computes distance between sliding windows embeddings

        Parameter
        ---------
        current_file : dict

        Returns
        -------
        predictions : SlidingWindowFeature
        """

        warnings.warn('Segmentation.apply is probably broken')

        # apply sequence labeling to the whole file
        y = []
        for t_batch, left_batch, right_batch in self.from_file(current_file):
            y_batch = np.diag(
                cdist(left_batch, right_batch, metric=self.distance))
            y.append(y_batch)

        y = np.hstack(y)

        window = SlidingWindow(duration=2 * self.duration,
                               step=self.step, start=0.)
        return SlidingWindowFeature(y, window)
