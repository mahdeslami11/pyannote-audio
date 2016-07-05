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

from keras.callbacks import Callback
from keras.models import model_from_yaml


class EmbeddingCheckpoint(Callback):

    def __init__(self, sequence_embedding,
                       checkpoint='weights.{epoch:03d}-{loss:.2f}.hdf5'):
        super(EmbeddingCheckpoint, self).__init__()
        self.sequence_embedding = sequence_embedding
        self.checkpoint = checkpoint

    def on_epoch_end(self, epoch, logs={}):
        weights = self.checkpoint.format(epoch=epoch, **logs)
        self.sequence_embedding.to_disk(
            weights=weights, overwrite=True, model=self.model)


class SequenceEmbedding(object):
    """

    Parameters
    ----------
    checkpoint: str, optional

    """
    def __init__(self, checkpoint='weights.{epoch:03d}-{loss:.2f}.hdf5'):
        super(SequenceEmbedding, self).__init__()
        self.checkpoint = checkpoint

    @classmethod
    def from_disk(cls, architecture, weights):
        self = SequenceEmbedding()

        with open(architecture, 'r') as fp:
            yaml_string = fp.read()
        self.embedding_ = model_from_yaml(yaml_string)
        self.embedding_.load_weights(weights)
        return self

    def to_disk(self, architecture=None, weights=None, overwrite=False, input_shape=None, model=None):

        if architecture and os.path.isfile(architecture) and not overwrite:
            raise ValueError("File '{architecture}' already exists.".format(architecture=architecture))

        if weights and os.path.isfile(weights) and not overwrite:
            raise ValueError("File '{weights}' already exists.".format(weights=weights))

        if model is not None:
            embedding = self.get_embedding(model)

        elif hasattr(self, 'embedding_'):
            embedding = self._embedding

        elif input_shape is None:
            raise ValueError('Cannot save embedding to disk because input_shape is missing.')

        else:
            model = self.design_model(input_shape)
            embedding = self.get_embedding(model)

        if architecture:
            yaml_string = embedding.to_yaml()
            with open(architecture, 'w') as fp:
                fp.write(yaml_string)

        if weights:
            embedding.save_weights(weights, overwrite=overwrite)

    def fit(self, input_shape, generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=[], validation_data=None,
            nb_val_samples=None, class_weight={}, max_q_size=10):

        if self.checkpoint:
            callbacks.append(EmbeddingCheckpoint(self, checkpoint=self.checkpoint))

        self.model_ = self.design_model(input_shape)
        self.embedding_ = self.get_embedding(self.model_)
        self.model_.fit_generator(
            generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=callbacks, validation_data=validation_data,
            nb_val_samples=nb_val_samples, class_weight=class_weight,
            max_q_size=max_q_size)

    def transform(self, sequence, batch_size=32, verbose=0):
        return self.embedding_.predict(
            sequence, batch_size=batch_size, verbose=verbose)


class TripletLossSequenceEmbedding(SequenceEmbedding):
    """Triplet loss sequence embedding

    Parameters
    ----------
    output_dim: int
        Embedding dimension.
    margin: float, optional
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
    def __init__(self, output_dim, margin=0.2, lstm=[12], dense=[],
                 bidirectional=False, optimizer='rmsprop',
                 checkpoint='weights.{epoch:03d}.hdf5'):
        super(TripletLossSequenceEmbedding, self).__init__(
            checkpoint=checkpoint)
        self.output_dim = output_dim
        self.margin = margin
        self.lstm = lstm
        self.dense = dense
        self.bidirectional = bidirectional
        self.optimizer = optimizer

    def design_embedding(self, input_shape):

        inputs = Input(shape=input_shape,
                       name="embedding_input")
        x = inputs

        # stack LSTM layers
        n_lstm = len(self.lstm)
        for i, output_dim in enumerate(self.lstm):

            # last LSTM should not return a sequence
            return_sequences = i+1 < n_lstm
            if i:
                # all but first LSTM
                forward = LSTM(output_dim=output_dim,
                               return_sequences=return_sequences,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(forward)
                if self.bidirectional:
                    backward = LSTM(output_dim=output_dim,
                                    return_sequences=return_sequences,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(backward)
            else:
                # first LSTM
                forward = LSTM(input_shape=input_shape,
                               output_dim=output_dim,
                               return_sequences=return_sequences,
                               activation='tanh',
                               dropout_W=0.0,
                               dropout_U=0.0)(x)
                if self.bidirectional:
                    backward = LSTM(go_backwards=True,
                                    input_shape=input_shape,
                                    output_dim=output_dim,
                                    return_sequences=return_sequences,
                                    activation='tanh',
                                    dropout_W=0.0,
                                    dropout_U=0.0)(x)

        # concatenate forward and backward
        if self.bidirectional:
            x = merge([forward, backward], mode='concat', concat_axis=1)
        else:
            x = forward

        # stack dense layers
        for i, output_dim in enumerate(self.dense):
            x = Dense(output_dim, activation='tanh')(x)

        # stack final dense layer
        x = Dense(self.output_dim, activation='tanh')(x)

        # stack L2 normalization layer
        embeddings = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                            name="embedding_output")(x)

        return Model(input=inputs, output=embeddings)

    def _triplet_loss(self, inputs):
        p = K.sum(K.square(inputs[0] - inputs[1]), axis=-1, keepdims=True)
        n = K.sum(K.square(inputs[0] - inputs[2]), axis=-1, keepdims=True)
        return K.maximum(0, p + self.margin - n)

    @staticmethod
    def _output_shape(input_shapes):
        return (input_shapes[0][0], 1)

    @staticmethod
    def _identity_loss(y_true, y_pred):
        return K.mean(y_pred - 0 * y_true)

    def get_embedding(self, model):
        return model.layers_by_depth[1][0]

    def design_model(self, input_shape):
        """
        Parameters
        ----------
        input_shape: (n_samples, n_features) tuple
            Shape of input sequences.
        """

        anchor = Input(shape=input_shape, name="anchor")
        positive = Input(shape=input_shape, name="positive")
        negative = Input(shape=input_shape, name="negative")

        embedding = self.design_embedding(input_shape)
        embedded_anchor = embedding(anchor)
        embedded_positive = embedding(positive)
        embedded_negative = embedding(negative)

        distance = merge(
            [embedded_anchor, embedded_positive, embedded_negative],
            mode=self._triplet_loss, output_shape=self._output_shape)

        model = Model(input=[anchor, positive, negative], output=distance)

        model.compile(optimizer=self.optimizer, loss=self._identity_loss)

        return model
