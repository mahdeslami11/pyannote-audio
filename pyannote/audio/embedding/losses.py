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
from keras.models import Model

from keras.layers import Input
from keras.layers import merge


class Loss(object):
    """Loss function for sequence embedding training

    Parameters
    ----------
    design_embedding : function or callable
        This function should take input_shape as input and return a Keras model
        should take a sequence as input, and return the embedding as output.

    See also
    --------
    An example of `design_embedding` can be found in
    pyannote.audio.embedding.models.TristouNet.__call__
    """
    def __init__(self, design_embedding, **kwargs):
        super(Loss, self).__init__()
        self.design_embedding = design_embedding

    def __call__(self, y_true, y_pred):
        raise NotImplementedError('')

    def design_model(self, input_shape):
        """Design the model for which the loss is optimized

        This method can (and should!) make use of `design_embedding` attribute

        Parameters
        ----------
        input_shape: (n_samples, n_features) tuple
            Shape of input sequences.

        Returns
        -------
        model : Keras model

        See also
        --------
        An example of such a method can be found in `TripletLoss` class
        """
        raise NotImplementedError('')

    def get_embedding(self, from_model):
        """Extract embedding Keras model

        Parameters
        ----------
        from_model : Keras model
            Current state of the model

        Returns
        -------
        embedding : Keras model
        """
        raise NotImplementedError('')


class TripletLoss(Loss):
    """Triplet loss for sequence embedding

            anchor        |-----------|     |---------|
            input    -->  | embedding | --> |         |
            sequence      |-----------|     |         |
                                            |         |
            positive      |-----------|     | triplet |
            input    -->  | embedding | --> |         | --> loss value
            sequence      |-----------|     |  loss   |
                                            |         |
            negative      |-----------|     |         |
            input    -->  | embedding | --> |         |
            sequence      |-----------|     |---------|

    Parameters
    ----------
    design_embedding : callable, or func
        This function should take input_shape as input and return the embedding
        as a Keras model
    margin : float, optional
        Triplet loss margin. Defaults to 0.2.

    Reference
    ---------
    Hervé Bredin, "TristouNet: Triplet Loss for Speaker Turn Embedding"
    Submitted to ICASSP 2017. https://arxiv.org/abs/1609.04301

    See also
    --------
    An example of `design_embedding` can be found in
    pyannote.audio.embedding.models.TristouNet.__call__
    """
    def __init__(self, design_embedding, margin=0.2):
        super(TripletLoss, self).__init__(design_embedding)
        self.margin = margin
        # HACK https://github.com/fchollet/keras/issues/3833
        self.__name__ = 'TripletLoss'

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

    def __call__(self, y_true, y_pred):
        return self._identity_loss(y_true, y_pred)

    def get_embedding(self, from_model):
        return from_model.layers_by_depth[1][0]

    def design_model(self, input_shape):
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
