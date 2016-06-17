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


import keras.backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Layer
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Input
from keras.layers import merge


class L2Normalize(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        return super(L2Normalize, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return K.l2_normalize(x, axis=self.axis)

    def get_config(self):
        base_config = super(L2Normalize, self).get_config()
        base_config.update(dict(axis=self.axis))
        return base_config


class TripletLossSequenceEmbedding(object):
    """

    Parameters
    ----------
    output_dim: int
        Embedding dimension.
    lstm: list
        List of output dimension of stacked LSTMs.
        Defaults to [12, ] (i.e. one LSTM with output dimension 12)
    dense: list
        List of output dimension of additionnal stacked dense layers.
        Defaults to [] (i.e. do not add any dense layer)

    """
    def __init__(self, output_dim, lstm=[12], dense=[]):
        super(TripletLossSequenceEmbedding, self).__init__()
        self.output_dim = output_dim
        self.lstm = lstm
        self.dense = dense

    def _embedding(self, input_shape):

        model = Sequential(name="embedding")

        # stack LSTM layers
        n_lstm = len(self.lstm)
        for i, output_dim in enumerate(self.lstm):
            return_sequences = i+1 < n_lstm
            layer = LSTM(input_shape=input_shape if i==0 else None,
                         output_dim=output_dim,
                         return_sequences=return_sequences,
                         activation='tanh')
            model.add(layer)

        # stack dense layers
        for i, output_dim in enumerate(self.dense):
            layer = Dense(output_dim, activation='tanh')
            model.add(layer)

        # stack final dense layer
        layer = Dense(self.output_dim, activation='tanh')
        model.add(layer)

        # stack L2 normalization layer
        model.add(L2Normalize())

        return model

    @staticmethod
    def _triplet_loss(inputs, alpha=0.2):
        p = K.sum(K.square(inputs[0] - inputs[1]), axis=-1)
        n = K.sum(K.square(inputs[0] - inputs[2]), axis=-1)
        return K.maximum(0, p + alpha - n)

    @staticmethod
    def _output_shape(input_shapes):
        return (input_shapes[0][0], 1)

    @staticmethod
    def _identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)

    def get_model(self, input_shape):
        """
        Parameters
        ----------
        input_shape: (n_samples, n_features) tuple
            Shape of input sequences.
        """

        anchor = Input(shape=input_shape, name="anchor")
        positive = Input(shape=input_shape, name="positive")
        negative = Input(shape=input_shape, name="negative")

        embed = self._embedding(input_shape)
        embedded_anchor = embed(anchor)
        embedded_positive = embed(positive)
        embedded_negative = embed(negative)

        distance = merge(
            [embedded_anchor, embedded_positive, embedded_negative],
            mode=self._triplet_loss, output_shape=self._output_shape)

        model = Model(input=[anchor, positive, negative], output=distance)

        model.compile(optimizer='rmsprop', loss=self._identity_loss)

        return model

    def get_embedding(self, from_model):
        return from_model.get_layer(name="embedding")
