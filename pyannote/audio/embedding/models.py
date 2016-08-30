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
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import merge

from pyannote.audio.callback import LoggingCallback
from keras.models import model_from_yaml


class SequenceEmbedding(object):
    """Base class for sequence embedding

    Parameters
    ----------
    log_dir: str, optional
        When provided, log status after each epoch into this directory. This
        will create several files, including loss plots and weights files.
    """
    def __init__(self, log_dir=None):
        super(SequenceEmbedding, self).__init__()
        self.log_dir = log_dir

    @classmethod
    def from_disk(cls, architecture, weights):
        """Load pre-trained sequence embedding from disk

        Parameters
        ----------
        architecture : str
            Path to architecture file (e.g. created by `to_disk` method)
        weights : str
            Path to pre-trained weight file (e.g. created by `to_disk` method)

        Returns
        -------
        sequence_embedding : SequenceEmbedding
            Pre-trained sequence embedding model.
        """
        self = SequenceEmbedding()

        with open(architecture, 'r') as fp:
            yaml_string = fp.read()
        self.model_ = model_from_yaml(yaml_string)
        self.model_.load_weights(weights)
        return self

    def to_disk(self, architecture=None, weights=None, overwrite=False, input_shape=None, model=None):
        """Save trained sequence embedding to disk

        Parameters
        ----------
        architecture : str, optional
            When provided, path where to save architecture.
        weights : str, optional
            When provided, path where to save weights
        overwrite : boolean, optional
            Overwrite (architecture or weights) file in case they exist.
        """

        if not hasattr(self, 'model_'):
            raise AttributeError('Model must be trained first.')

        if architecture and os.path.isfile(architecture) and not overwrite:
            raise ValueError("File '{architecture}' already exists.".format(architecture=architecture))

        if weights and os.path.isfile(weights) and not overwrite:
            raise ValueError("File '{weights}' already exists.".format(weights=weights))

        if architecture:
            yaml_string = self.model_.to_yaml()
            with open(architecture, 'w') as fp:
                fp.write(yaml_string)

        if weights:
            self.model_.save_weights(weights, overwrite=overwrite)

    def loss(self, y_true, y_pred):
        raise NotImplementedError('')

    def fit(self, input_shape, generator,
            samples_per_epoch, nb_epoch, callbacks=[]):
        """Train model

        Parameters
        ----------
        input_shape :
        generator :
        samples_per_epoch :
        np_epoch :
        callbacks :
        """

        if not callbacks and self.log_dir:
            default_callback = LoggingCallback(self, log_dir=self.log_dir)
            callbacks = [default_callback]

        self.model_ = self.design_model(input_shape)
        self.model_.compile(optimizer=self.optimizer,
                            loss=self.loss)

        self.model_.fit_generator(
            generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=callbacks)

    def transform(self, sequence, batch_size=32, verbose=0):
        if not hasattr(self, 'embedding_'):
            self.embedding_ = self.get_embedding()

        return self.embedding_.predict(
            sequence, batch_size=batch_size, verbose=verbose)


class BiLSTMSequenceEmbedding(SequenceEmbedding):
        """Bi-directional LSTM sequence embedding

        Parameters
        ----------
        output_dim: int
            Embedding dimension.
        lstm: list, optional
            List of output dimension of stacked LSTMs.
            Defaults to [12, ] (i.e. one LSTM with output dimension 12)
        dense: list, optional
            List of output dimension of additionnal stacked dense layers.
            Defaults to [] (i.e. do not add any dense layer)
        bidirectional: boolean, optional
            When True, use bi-directional LSTMs
        space: {'sphere', 'quadrant'}, optional
            When 'sphere' (resp. 'quadrant'), use 'tanh' (resp. 'sigmoid') as
            final activation. Defaults to 'sphere'.
        optimizer: str, optional
            Keras optimizer. Defaults to 'rmsprop'.
        log_dir: str, optional
            When provided, log status after each epoch into this directory. This
            will create several files, including loss plots and weights files.
        """
        def __init__(self, output_dim, lstm=[12], dense=[],
                     bidirectional=False, space='sphere',
                     margin=0.2, optimizer='rmsprop', log_dir=None):

            self.output_dim = output_dim
            self.lstm = lstm
            self.dense = dense
            self.bidirectional = bidirectional
            self.space = space
            self.optimizer = optimizer

            super(BiLSTMSequenceEmbedding, self).__init__(
                log_dir=log_dir)

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
            if self.space == 'sphere':
                activation = 'tanh'
            elif self.space == 'quadrant':
                activation = 'sigmoid'
            x = Dense(self.output_dim, activation=activation)(x)

            # stack L2 normalization layer
            embeddings = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                                name="embedding_output")(x)

            return Model(input=inputs, output=embeddings)


class TripletLossBiLSTMSequenceEmbedding(BiLSTMSequenceEmbedding):
    """Triplet loss Bi-directional LSTM sequence embedding

    Parameters
    ----------
    output_dim: int
        Embedding dimension.
    margin: float, optional
        Defaults to 0.2
    lstm: list, optional
        List of output dimension of stacked LSTMs.
        Defaults to [12, ] (i.e. one LSTM with output dimension 12)
    dense: list, optional
        List of output dimension of additionnal stacked dense layers.
        Defaults to [] (i.e. do not add any dense layer)
    bidirectional: boolean, optional
        When True, use bi-directional LSTMs
    space: {'sphere', 'quadrant'}, optional
        When 'sphere' (resp. 'quadrant'), use 'tanh' (resp. 'sigmoid') as
        final activation. Defaults to 'sphere'.
    optimizer: str, optional
        Keras optimizer. Defaults to 'rmsprop'.
    log_dir: str, optional
        When provided, log status after each epoch into this directory. This
        will create several files, including loss plots and weights files.
    """
    def __init__(self, output_dim, lstm=[12], dense=[],
                 bidirectional=False, space='sphere',
                 margin=0.2, optimizer='rmsprop', log_dir=None):

        self.margin = margin

        super(TripletLossBiLSTMSequenceEmbedding, self).__init__(
            output_dim,
            lstm=lstm,
            dense=dense,
            bidirectional=bidirectional,
            space=space,
            optimizer=optimizer,
            log_dir=log_dir)

        self.optimizer = optimizer

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

    def loss(self, y_true, y_pred):
        return self._identity_loss(y_true, y_pred)

    def get_embedding(self):
        """Extract embedding from Keras model (a posteriori)"""
        return self.model_.layers_by_depth[1][0]

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

        return model
