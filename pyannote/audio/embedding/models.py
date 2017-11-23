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
# Grégory GELLY

import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.models import Model

from keras.layers import Input
from keras.layers import Masking
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import Permute
from keras.layers import RepeatVector
from keras.layers import Concatenate
from keras.layers import Multiply
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import TimeDistributed
import numpy as np

from pyannote.audio.keras_utils import register_custom_object


class EmbeddingAveragePooling(Layer):

    def __init__(self, **kwargs):
        super(EmbeddingAveragePooling, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]
        self.supports_masking = True

    def call(self, x, mask=None):
        # thanks to L2 normalization, mask actually has no effect
        return K.l2_normalize(K.sum(x, axis=1), axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def compute_mask(self, input, input_mask=None):
        return None


# register user-defined Keras layer
register_custom_object('EmbeddingAveragePooling', EmbeddingAveragePooling)


class TristouNet(object):
    """TristouNet sequence embedding

    RNN ( » ... » RNN ) » pooling › ( MLP › ... › ) MLP › normalize

    Reference
    ---------
    Hervé Bredin, "TristouNet: Triplet Loss for Speaker Turn Embedding"
    Submitted to ICASSP 2017.
    https://arxiv.org/abs/1609.04301

    Parameters
    ----------
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    implementation : {0, 1, 2}, optional
        If set to 0, the RNN will use an implementation that uses fewer,
        larger matrix products, thus running faster on CPU but consuming more
        memory. If set to 1, the RNN will use more matrix products, but smaller
        ones, thus running slower (may actually be faster on GPU) while
        consuming less memory. If set to 2 (LSTM/GRU only), the RNN will combine
        the input gate, the forget gate and the output gate into a single
        matrix, enabling more time-efficient parallelization on the GPU.
    mask : bool, optional
        Set to True to support variable length sequences through masking.
    recurrent: list, optional
        List of output dimension of stacked RNNs.
        Defaults to [16, ] (i.e. one RNN with output dimension 16)
    bidirectional: {False, 'ave', 'concat'}, optional
        Defines how the output of forward and backward RNNs are merged.
        'ave' stands for 'average', 'concat' (default) for concatenation.
        See keras.layers.wrappers.Bidirectional for more information.
        Use False to only use forward RNNs.
    mlp: list, optional
        Number of units in additionnal stacked dense MLP layers.
        Defaults to [16, 16] (i.e. two dense MLP layers with 16 units)
    """

    def __init__(self, rnn='LSTM', implementation=0, mask=False,
                 recurrent=[16,], bidirectional='concat', mlp=[16, 16]):

        super(TristouNet, self).__init__()
        self.rnn = rnn
        self.implementation = implementation
        self.mask = mask
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.mlp = mlp

        rnns = __import__('keras.layers.recurrent', fromlist=[self.rnn])
        self.RNN_ = getattr(rnns, self.rnn)


    def __call__(self, input_shape):
        """Design embedding

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence.

        Returns
        -------
        model : Keras model
        """

        inputs = Input(shape=input_shape,
                       name="input")

        if self.mask:
            masking = Masking(mask_value=0.)
            x = masking(inputs)
        else:
            x = inputs

        # stack RNN layers
        for i, output_dim in enumerate(self.recurrent):

            params = {
                'name': 'rnn_{i:d}'.format(i=i),
                'return_sequences': True,
                # 'go_backwards': False,
                # 'stateful': False,
                # 'unroll': False,
                'implementation': self.implementation,
                'activation': 'tanh',
                # 'recurrent_activation': 'hard_sigmoid',
                # 'use_bias': True,
                # 'kernel_initializer': 'glorot_uniform',
                # 'recurrent_initializer': 'orthogonal',
                # 'bias_initializer': 'zeros',
                # 'unit_forget_bias': True,
                # 'kernel_regularizer': None,
                # 'recurrent_regularizer': None,
                # 'bias_regularizer': None,
                # 'activity_regularizer': None,
                # 'kernel_constraint': None,
                # 'recurrent_constraint': None,
                # 'bias_constraint': None,
                # 'dropout': 0.0,
                # 'recurrent_dropout': 0.0,
            }

            # first RNN needs to be given the input shape
            if i == 0:
                params['input_shape'] = input_shape

            recurrent = self.RNN_(output_dim, **params)

            if self.bidirectional:
                recurrent = Bidirectional(recurrent, merge_mode=self.bidirectional)

            x = recurrent(x)

        pooling = EmbeddingAveragePooling(name='pooling')
        x = pooling(x)

        # stack dense MLP layers
        for i, output_dim in enumerate(self.mlp):

            mlp = Dense(output_dim,
                        activation='tanh',
                        name='mlp_{i:d}'.format(i=i))
            x = mlp(x)

        normalize = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                           output_shape=(output_dim, ),
                           name='normalize')
        embeddings = normalize(x)

        return Model(inputs=inputs, outputs=embeddings)

    @property
    def output_dim(self):
        return self.mlp[-1]


class TrottiNet(object):
    """TrottiNet sequence embedding

    RNN ( » ... » RNN ) » ( MLP » ... » ) MLP » pooling › normalize

    Parameters
    ----------
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    implementation : {0, 1, 2}, optional
        If set to 0, the RNN will use an implementation that uses fewer,
        larger matrix products, thus running faster on CPU but consuming more
        memory. If set to 1, the RNN will use more matrix products, but smaller
        ones, thus running slower (may actually be faster on GPU) while
        consuming less memory. If set to 2 (LSTM/GRU only), the RNN will combine
        the input gate, the forget gate and the output gate into a single
        matrix, enabling more time-efficient parallelization on the GPU.
    mask : bool, optional
        Set to True to support variable length sequences through masking.
    recurrent: list, optional
        List of output dimension of stacked RNNs.
        Defaults to [16, ] (i.e. one RNN with output dimension 16)
    bidirectional: {False, 'ave', 'concat'}, optional
        Defines how the output of forward and backward RNNs are merged.
        'ave' (default) stands for 'average', 'concat' for concatenation.
        See keras.layers.wrappers.Bidirectional for more information.
        Use False to only use forward RNNs.
    mlp: list, optional
        Number of units of additionnal stacked dense MLP layers.
        Defaults to [16, 16] (i.e. add one dense MLP layer with 16 units)
    """

    def __init__(self, rnn='LSTM', implementation=0, mask=False,
                 recurrent=[16,], bidirectional='ave', mlp=[16, 16]):

        super(TrottiNet, self).__init__()
        self.rnn = rnn
        self.implementation = implementation
        self.mask = mask
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.mlp = mlp

        rnns = __import__('keras.layers.recurrent', fromlist=[self.rnn])
        self.RNN_ = getattr(rnns, self.rnn)

    def __call__(self, input_shape):
        """Design embedding

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence.

        Returns
        -------
        model : Keras model
        """

        inputs = Input(shape=input_shape,
                       name="input")

        if self.mask:
            masking = Masking(mask_value=0.)
            x = masking(inputs)
        else:
            x = inputs

        # stack (bidirectional) RNN layers
        for i, output_dim in enumerate(self.recurrent):

            internal_layer = not self.mlp and i + 1 == len(self.recurrent)

            params = {
                'name': 'rnn_{i:d}'.format(i=i),
                'return_sequences': True,
                # 'go_backwards': False,
                # 'stateful': False,
                # 'unroll': False,
                'implementation': self.implementation,
                'activation': 'tanh',
                # 'recurrent_activation': 'hard_sigmoid',
                # 'use_bias': True,
                # 'kernel_initializer': 'glorot_uniform',
                # 'recurrent_initializer': 'orthogonal',
                # 'bias_initializer': 'zeros',
                # 'unit_forget_bias': True,
                # 'kernel_regularizer': None,
                # 'recurrent_regularizer': None,
                # 'bias_regularizer': None,
                # 'activity_regularizer': None,
                # 'kernel_constraint': None,
                # 'recurrent_constraint': None,
                # 'bias_constraint': None,
                # 'dropout': 0.0,
                # 'recurrent_dropout': 0.0,
            }

            # first RNN needs to be given the input shape
            if i == 0:
                params['input_shape'] = input_shape

            if internal_layer and not self.bidirectional:
                params['name'] = 'internal'

            recurrent = self.RNN_(output_dim, **params)

            if self.bidirectional:
                name = 'internal' if internal_layer else None
                recurrent = Bidirectional(recurrent,
                                     merge_mode=self.bidirectional,
                                     name=name)

            x = recurrent(x)

        # stack dense MLP layers
        for i, output_dim in enumerate(self.mlp):

            internal_layer = i + 1 == len(self.mlp)

            mlp = Dense(output_dim,
                        activation='tanh',
                        name='mlp_{i:d}'.format(i=i))

            name = 'internal' if internal_layer else None
            x = TimeDistributed(mlp, name=name)(x)

        # average pooling and L2 normalization
        pooling = EmbeddingAveragePooling(name='pooling')
        embeddings = pooling(x)

        return Model(inputs=inputs, outputs=embeddings)

    @property
    def output_dim(self):
        return self.mlp[-1]


class ClopiNet(object):
    """ClopiNet sequence embedding

    RNN          ⎤
      » RNN      ⎥ ( » MLP » ... » MLP ) » pooling › normalize
           » RNN ⎦

    Parameters
    ----------
    rnn: {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    implementation : {0, 1, 2}, optional
        If set to 0, the RNN will use an implementation that uses fewer,
        larger matrix products, thus running faster on CPU but consuming more
        memory. If set to 1, the RNN will use more matrix products, but smaller
        ones, thus running slower (may actually be faster on GPU) while
        consuming less memory. If set to 2 (LSTM/GRU only), the RNN will combine
        the input gate, the forget gate and the output gate into a single
        matrix, enabling more time-efficient parallelization on the GPU.
    mask : bool, optional
        Set to True to support variable length sequences through masking.
    recurrent: list, optional
        List of output dimension of stacked RNNs.
        Defaults to [16, ] (i.e. one RNN with output dimension 16)
    bidirectional: {False, 'ave', 'concat'}, optional
        Defines how the output of forward and backward RNNs are merged.
        'ave' (default) stands for 'average', 'concat' for concatenation.
        See keras.layers.wrappers.Bidirectional for more information.
        Use False to only use forward RNNs.
    mlp: list, optional
        Number of units in additionnal stacked dense MLP layers.
        Defaults to [] (i.e. do not stack any dense MLP layer)
    linear: bool, optional
        Make final dense layer use linear activation. Has no effect
        when `mlp` is empty.
    """

    def __init__(self, rnn='LSTM', implementation=0, mask=False,
                 recurrent=[16, 8, 8], bidirectional='ave', mlp=[],
                 linear=False, attention=False):
        super(ClopiNet, self).__init__()
        self.rnn = rnn
        self.implementation = implementation
        self.mask = mask
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.mlp = mlp
        self.linear = linear
        self.attention = attention

        rnns = __import__('keras.layers.recurrent', fromlist=[self.rnn])
        self.RNN_ = getattr(rnns, self.rnn)

    def __call__(self, input_shape):
        """Design embedding

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence.

        Returns
        -------
        model : Keras model
        """

        inputs = Input(shape=input_shape,
                       name="input")

        if self.mask:
            masking = Masking(mask_value=0.)
            x = masking(inputs)
        else:
            x = inputs

        # stack (bidirectional) recurrent layers
        for i, output_dim in enumerate(self.recurrent):

            # is it the sole layer in the network?
            sole_layer = not self.mlp and len(self.recurrent) == 1

            # is it the last layer before pooling?
            internal_layer = not self.mlp and i + 1 == len(self.recurrent)

            # the sole purpose of these two booleans is to determine
            # which layer in the last one before pooling so that it
            # can be given the name "internal".

            params = {
                'name': 'rnn_{i:d}'.format(i=i),
                'return_sequences': True,
                # 'go_backwards': False,
                # 'stateful': False,
                # 'unroll': False,
                'implementation': self.implementation,
                'activation': 'tanh',
                # 'recurrent_activation': 'hard_sigmoid',
                # 'use_bias': True,
                # 'kernel_initializer': 'glorot_uniform',
                # 'recurrent_initializer': 'orthogonal',
                # 'bias_initializer': 'zeros',
                # 'unit_forget_bias': True,
                # 'kernel_regularizer': None,
                # 'recurrent_regularizer': None,
                # 'bias_regularizer': None,
                # 'activity_regularizer': None,
                # 'kernel_constraint': None,
                # 'recurrent_constraint': None,
                # 'bias_constraint': None,
                # 'dropout': 0.0,
                # 'recurrent_dropout': 0.0,
            }

            # first RNN needs to be given the input shape
            if i == 0:
                params['input_shape'] = input_shape

            if sole_layer and not self.bidirectional and not self.attention:
                params['name'] = 'internal'

            recurrent = self.RNN_(output_dim, **params)

            # bi-directional RNN
            if self.bidirectional:
                name = 'internal' if (sole_layer and not self.attention) \
                       else None
                recurrent = Bidirectional(recurrent,
                                          merge_mode=self.bidirectional,
                                          name=name)

            # (actually) stack RNN
            x = recurrent(x)

            # concatenate output of all levels
            if i > 0:
                name = 'internal' if (internal_layer and not self.attention) \
                       else None
                concat_x = Concatenate(axis=-1, name=name)([concat_x, x])
            else:
                # corner case for 1st level (i=0)
                # as concat_x does not yet exist
                concat_x = x

        # just rename the concatenated output variable to x
        x = concat_x

        # (optionally) stack dense MLP layers
        for i, output_dim in enumerate(self.mlp):

            internal_layer = i + 1 == len(self.mlp)

            activation = 'tanh'
            use_bias = True
            if internal_layer and self.linear:
                activation = 'linear'
                use_bias = True  # TODO - test with use_bias = False

            mlp = Dense(output_dim,
                        name='mlp_{i:d}'.format(i=i),
                        activation=activation,
                        use_bias=use_bias)

            name = 'internal' if (internal_layer and not self.attention) \
                   else None
            x = TimeDistributed(mlp, name=name)(x)

        if self.attention:
            # https://github.com/fchollet/keras/issues/1472
            attention = Dense(input_shape[0],
                              activation='softmax',
                              name='attention')
            a = attention(Flatten()(x))
            a = RepeatVector(self.output_dim)(a)
            a = Permute((2, 1))(a)
            x = Multiply(name='internal')([x, a])

        # average pooling and L2 normalization
        pooling = EmbeddingAveragePooling(name='pooling')
        embeddings = pooling(x)

        return Model(inputs=[inputs], outputs=embeddings)

    @property
    def output_dim(self):
        if self.mlp:
            return self.mlp[-1]
        return np.sum(self.recurrent)
