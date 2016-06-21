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

import os.path

import keras.backend as K
from keras.models import Sequential
from keras.models import Model

from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import merge

from keras.callbacks import ModelCheckpoint
from keras.models import model_from_yaml


class SequenceEmbedding(object):
    """

    Parameters
    ----------
    checkpoint: str, optional

    """
    def __init__(self, checkpoint='weights.{epoch:03d}-{loss:.2f}.hdf5'):
        super(SequenceEmbedding, self).__init__()
        self.checkpoint = checkpoint

    def _get_embedding(self, from_model):
        return from_model.get_layer(name="embedding")

    @classmethod
    def from_disk(cls, architecture, weights):
        self = SequenceEmbedding()

        with open(architecture, 'r') as fp:
            yaml_string = fp.read()
        self.embedding_ = model_from_yaml(yaml_string)
        self.embedding_.load_weights(weights)
        return self

    def to_disk(self, architecture, weights=None, overwrite=False):

        if os.path.isfile(architecture) and not overwrite:
            raise ValueError("File '{architecture}' already exists.".format(architecture=architecture))

        if weights and os.path.isfile(weights) and not overwrite:
            raise ValueError("File '{weights}' already exists.".format(weights=weights))

        yaml_string = self.embedding_.to_yaml()
        with open(architecture, 'w') as fp:
            fp.write(yaml_string)

        if weights:
            self.embedding_.save_weights(weights, overwrite=overwrite)

    def fit(self, input_shape, generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=[], validation_data=None,
            nb_val_samples=None, class_weight={}, max_q_size=10):

        if self.checkpoint:
            callbacks.append(ModelCheckpoint(
                self.checkpoint, monitor='loss', verbose=0,
                save_best_only=False, mode='auto'))

        self.model_ = self._get_model(input_shape)
        self.model_.fit_generator(
            generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=callbacks, validation_data=validation_data,
            nb_val_samples=nb_val_samples, class_weight=class_weight,
            max_q_size=max_q_size)
        self.embedding_ = self._get_embedding(self.model_)

    def transform(self, sequence, batch_size=32, verbose=0):
        return self.embedding_.predict(
            sequence, batch_size=batch_size, verbose=verbose)


class TripletLossSequenceEmbedding(SequenceEmbedding):
    """Triplet loss sequence embedding

    Parameters
    ----------
    output_dim: int
        Embedding dimension.
    alpha: float, optional
        Defaults to 0.2
    lstm: list
        List of output dimension of stacked LSTMs.
        Defaults to [12, ] (i.e. one LSTM with output dimension 12)
    dense: list
        List of output dimension of additionnal stacked dense layers.
        Defaults to [] (i.e. do not add any dense layer)
    checkpoint: str
        Defaults to 'weights.{epoch:03d}.hdf5'
    """
    def __init__(self, output_dim, alpha=0.2, lstm=[12], dense=[],
                 checkpoint='weights.{epoch:03d}.hdf5'):
        super(TripletLossSequenceEmbedding, self).__init__(
            checkpoint=checkpoint)
        self.output_dim = output_dim
        self.alpha = alpha
        self.lstm = lstm
        self.dense = dense

    def _embedding(self, input_shape):

        model = Sequential(name="embedding")

        # stack LSTM layers
        n_lstm = len(self.lstm)
        for i, output_dim in enumerate(self.lstm):
            return_sequences = i+1 < n_lstm
            if i:
                layer = LSTM(output_dim=output_dim,
                             return_sequences=return_sequences,
                             activation='tanh')
            else:
                layer = LSTM(input_shape=input_shape,
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
        model.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))

        return model

    def _triplet_loss(self, inputs):
        p = K.sum(K.square(inputs[0] - inputs[1]), axis=-1, keepdims=True)
        n = K.sum(K.square(inputs[0] - inputs[2]), axis=-1, keepdims=True)
        return K.maximum(0, p + self.alpha - n)

    @staticmethod
    def _output_shape(input_shapes):
        return (input_shapes[0][0], 1)

    @staticmethod
    def _identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)

    def _get_model(self, input_shape):
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
