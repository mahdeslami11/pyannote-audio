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

import numpy as np
import keras.backend as K
from pyannote.core import SlidingWindow, SlidingWindowFeature
from ..labeling.base import SequenceLabeling

import torch.nn as nn
from torch.autograd import Variable
import torch


class SequenceEmbedding(SequenceLabeling):
    """Sequence embedding

    Parameters
    ----------
    model : keras.Model or nn.Module
        Pre-trained sequence embedding model.
    feature_extraction : callable
        Feature extractor
    duration : float
        Subsequence duration, in seconds.
    step : float, optional
        Subsequence step, in seconds. Defaults to 50% of `duration`.
    internal : bool, optional
        Extract internal representation.
    batch_size : int, optional
        Defaults to 32.

    Usage
    -----
    >>> from pyannote.audio.keras_utils import load_model
    >>> model = load_model('/path/to/model.h5')
    >>> feature_extraction = YaafeMFCC(...)
    >>> duration = 3.2
    >>> sequence_embedding = SequenceEmbedding(model, feature_extraction, duration)
    >>> sequence_embedding.apply(current_file)
    """

    def __init__(self, model, feature_extraction, duration,
                 step=None, batch_size=32, internal=False,
                 source='audio'):

        super(SequenceEmbedding, self).__init__(model, feature_extraction, duration,
                                        step=step, batch_size=batch_size,
                                        source=source)

        self.internal = internal

        # build function that takes batch of sequences as input
        # and returns their (internal) embedding

        if isinstance(self.model, nn.Module):
            self.model.internal = self.internal
            def embed(X):
                X = Variable(torch.from_numpy(np.rollaxis(np.array(X, dtype=np.float32), 0, 2)))
                emb = self.model(X)
                if self.internal:
                    return np.rollaxis(emb.data.numpy(), 1, 0)
                else:
                    return emb.data.numpy() 
            self.embed_ = embed

        else:

            if self.internal:
                # TODO add support for any internal layer
                output_layer = self.model.get_layer(name='internal')
                if output_layer is None:
                    raise ValueError(
                        'Model does not support extraction of internal embedding.')
            else:
                output_layer = self.model.get_layer(index=-1)

            input_layer = self.model.get_layer(name='input')
            K_func = K.function(
                [input_layer.input, K.learning_phase()], [output_layer.output])
            def embed(batch):
                return K_func([batch, 0])[0]
            self.embed_ = embed

    @property
    def sliding_window(self):
        if self.internal:
            return self.feature_extractor.sliding_window()
        else:
            return SlidingWindow(duration=self.duration, step=self.step)

    def postprocess_ndarray(self, X):
        """Embed sequences

        Parameters
        ----------
        X : (batch_size, n_samples, n_features) numpy array
            Batch of input sequences

        Returns
        -------
        fX : numpy array
            Batch of sequence embeddings.
            (batch_size, n_samples, n_dimensions) if internal
            (batch_size, n_dimensions) if not internal
        """
        return self.embed_(X)

    def apply(self, current_file):
        """Extract embeddings

        Can process either pyannote.database protocol items (as dict) or
        batch of precomputed feature sequences (as numpy array).

        Parameter
        ---------
        current_file : dict or numpy array
            File (from pyannote.database protocol) or batch of precomputed
            feature sequences.

        Returns
        -------
        embedding : SlidingWindowFeature or numpy array
        """

        if isinstance(current_file, np.ndarray):
            return self.embed_(current_file)

        if self.internal:
            return super(SequenceEmbedding, self).apply(current_file)

        # compute embedding on sliding window
        # over the whole duration of the file
        fX = np.vstack(
            [batch for batch in self.from_file(current_file,
                                               incomplete=True)])

        subsequences = SlidingWindow(duration=self.duration, step=self.step)

        return SlidingWindowFeature(fX, subsequences)
