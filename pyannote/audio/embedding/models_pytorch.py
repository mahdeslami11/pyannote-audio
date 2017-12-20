#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

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


import torch
from torch.autograd import Variable
import torch.nn as nn


class ClopiNet(nn.Module):
    """ClopiNet sequence embedding

    RNN          ⎤
      » RNN      ⎥ » MLP » Weight » temporal pooling › normalize
           » RNN ⎦

    Parameters
    ----------
    n_features : int
        Input feature dimension.
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent : list, optional
        List of hidden dimensions of stacked recurrent layers. Defaults to
        [16, ], i.e. one recurrent layer with hidden dimension of 16.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, ], i.e.
        one linear layer with hidden dimension of 16.
    weighted : bool, optional
        Add dimension-wise trainable weights. Defaults to False.

    Usage
    -----
    >>> model = ClopiNet(n_features)
    >>> final, internal = model(sequence)
    """

    def __init__(self, n_features,
                 rnn='LSTM', recurrent=[16,], bidirectional=False,
                 linear=[16, ], weighted=False):

        super(ClopiNet, self).__init__()

        self.n_features = n_features
        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.linear = linear
        self.weighted = weighted

        self.num_directions_ = 2 if self.bidirectional else 1

        # create list of recurrent layers
        self.recurrent_layers_ = []
        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(input_dim, hidden_dim,
                                          bidirectional=self.bidirectional)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(input_dim, hidden_dim,
                                         bidirectional=self.bidirectional)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim

        # the output of recurrent layers are concatenated so the input
        # dimension of subsequent linear layers is the sum of their output
        # dimension
        input_dim = sum(self.recurrent)

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        # define post-linear activation
        self.tanh_ = nn.Tanh()

        if self.weighted:
            self.alphas_ = nn.Parameter(torch.ones(input_dim))

    @property
    def output_dim(self):
        if self.linear:
            return self.linear[-1]
        return sum(self.recurrent)

    def forward(self, sequence):

        # check input feature dimension
        n_samples, batch_size, n_features = sequence.size()
        if n_features != self.n_features:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features))

        output = sequence

        outputs = []
        # stack recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                hidden = (
                    Variable(torch.zeros(
                        self.num_directions_, batch_size, hidden_dim),
                        requires_grad=False),
                    Variable(torch.zeros(
                        self.num_directions_, batch_size, hidden_dim),
                        requires_grad=False)
                )

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = Variable(torch.zeros(
                    self.num_directions_, batch_size, hidden_dim),
                    requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

            # average both directions in case of bidirectional layers
            if self.bidirectional:
                output = .5 * (output[:, :, :hidden_dim] + \
                               output[:, :, hidden_dim:])

            outputs.append(output)

        # concatenate outputs
        output = torch.cat(outputs, dim=2)

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = self.tanh_(output)

        if self.weighted:
            output = output * self.alphas_

        internal = output

        # average temporal pooling
        final = internal.sum(dim=0)

        # L2 normalization
        final = final / torch.norm(final, 2, 0)

        return final, internal
