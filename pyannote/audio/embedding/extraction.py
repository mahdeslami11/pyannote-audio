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
from pyannote.generators.fragment import SlidingSegments
from pyannote.audio.generators.periodic import PeriodicFeaturesMixin


class Extraction(PeriodicFeaturesMixin, FileBasedBatchGenerator):
    """Embedding extraction

    Parameters
    ----------
    embedding : keras.Model
        Pre-trained embedding.
    feature_extraction : callable
        Feature extractor
    duration : float
        Subsequence duration, in seconds.
    step : float, optional
        Subsequence step, in seconds. Defaults to 25% of `duration`.

    Usage
    -----
    >>> epoch = 1000
    >>> embedding = SequenceEmbeddingAutograd.load(train_dir, epoch)
    >>> feature_extraction = YaafeMFCC(...)
    >>> duration = 3.2
    >>> extraction = Extraction(embedding, feature_extraction, duration)

    """
    def __init__(self, embedding, feature_extraction, duration,
                 step=None, batch_size=32):

        self.embedding = embedding
        self.feature_extraction = feature_extraction
        self.duration = duration

        generator = SlidingSegments(duration=duration, step=step, source='wav')
        self.step = generator.step if step is None else step

        # build function that takes batch of sequences as input
        # and returns their internal embedding
        K_func = K.function(
            [self.embedding.get_layer(name='input').input, K.learning_phase()],
            [self.embedding.get_layer(name='internal').output])
        def embed(batch):
            return K_func([batch, 0])[0]
        self.embed_ = embed

        super(Extraction, self).__init__(generator, batch_size=self.batch_size)

    @property
    def dimension(self):
        return self.embedding.output_shape[-1]

    @property
    def sliding_window(self):
        return self.feature_extraction.sliding_window()

    def signature(self):
        shape = self.shape
        return {'type': 'ndarray', 'shape': shape}

    def postprocess_ndarray(self, X):
        """Embed sequences

        Parameters
        ----------
        X : (batch_size, n_samples, n_features) numpy array
            Batch of input sequences

        Returns
        -------
        fX : (batch_size, n_samples, n_dimensions) numpy array
            Batch of sequence embeddings.

        """
        return self.embed_(X)

    def apply(self, current_file):
        """Compute embeddings on a sliding window

        Parameter
        ---------
        current_file : dict

        Returns
        -------
        embedding : SlidingWindowFeature
        """

        # compute embedding on sliding window
        # over the whole duration of the file
        fX = np.vstack(
            [batch for batch in self.from_file(current_file,
                                               incomplete=True)])

        # get total number of frames
        identifier = get_unique_identifier(current_file)
        n_frames = self.preprocessed_['X'][identifier].shape[0]

        # data[i] is the sum of all embeddings for frame #i
        data = np.zeros((n_frames, self.dimension), dtype=np.float32)

        # k[i] is the number of sequences that overlap with frame #i
        k = np.zeros((n_frames, 1), dtype=np.int8)

        # frame and sub-sequence sliding windows
        frames = self.feature_extraction.sliding_window()
        subsequences = SlidingWindow(duration=self.duration, step=self.step)

        for subsequence, fX_ in zip(subsequences, fX):

            # indices of frames overlapped by subsequence
            indices = frames.crop(subsequence,
                                  mode='center',
                                  fixed=self.duration)

            # accumulate their embedding
            data[indices] += fX_

            # keep track of the number of overlapping sequence
            k[indices] += 1

        # compute average embedding of each frame
        data = data / np.maximum(k, 1)

        return SlidingWindowFeature(data, frames)
