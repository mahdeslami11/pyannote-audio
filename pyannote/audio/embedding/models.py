#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence


class TristouNet(nn.Module):
    """TristouNet sequence embedding

    RNN ( » ... » RNN ) » temporal pooling › ( MLP › ... › ) MLP › normalize

    Parameters
    ----------
    n_features : int
        Input feature dimension
    rnn : {'LSTM', 'GRU'}, optional
        Defaults to 'LSTM'.
    recurrent: list, optional
        List of output dimension of stacked RNNs.
        Defaults to [16, ] (i.e. one RNN with output dimension 16)
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False.
    pooling : {'sum', 'max'}
        Temporal pooling strategy. Defaults to 'sum'.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to [16, 16].

    Reference
    ---------
    Hervé Bredin. "TristouNet: Triplet Loss for Speaker Turn Embedding."
    ICASSP 2017 (https://arxiv.org/abs/1609.04301)
    """

    def __init__(self, n_features,
                 rnn='LSTM', recurrent=[16], bidirectional=False,
                 pooling='sum', linear=[16, 16]):

        super(TristouNet, self).__init__()

        self.n_features = n_features
        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.linear = [] if linear is None else linear

        self.num_directions_ = 2 if self.bidirectional else 1

        if self.pooling not in {'sum', 'max'}:
            raise ValueError('"pooling" must be one of {"sum", "max"}')

        # create list of recurrent layers
        self.recurrent_layers_ = []
        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(input_dim, hidden_dim,
                                          bidirectional=self.bidirectional,
                                          batch_first=True)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(input_dim, hidden_dim,
                                         bidirectional=self.bidirectional,
                                         batch_first=True)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim * (2 if self.bidirectional else 1)

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

    @property
    def output_dim(self):
        if self.linear:
            return self.linear[-1]
        return self.recurrent[-1] * (2 if self.bidirectional else 1)

    def forward(self, sequence):
        """

        Parameters
        ----------
        sequence : (batch_size, n_samples, n_features) torch.Tensor

        """

        packed_sequences = isinstance(sequence, PackedSequence)

        if packed_sequences:
            _, n_features = sequence.data.size()
            batch_size = sequence.batch_sizes[0].item()
            device = sequence.data.device
        else:
            # check input feature dimension
            batch_size, _, n_features = sequence.size()
            device = sequence.device

        if n_features != self.n_features:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features))

        output = sequence

        # recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                h = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                c = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                hidden = (h, c)

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = torch.zeros(
                    self.num_directions_, batch_size, hidden_dim,
                    device=device, requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

        if packed_sequences:
            output, lengths = pad_packed_sequence(output, batch_first=True)

        # batch_size, n_samples, dimension

        # average temporal pooling
        if self.pooling == 'sum':
            output = output.sum(dim=1)
        elif self.pooling == 'max':
            if packed_sequences:
                msg = ('"max" pooling is not yet implemented '
                       'for variable length sequences.')
                raise NotImplementedError(msg)
            output, _ = output.max(dim=1)

        # batch_size, dimension

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = F.tanh(output)

        # batch_size, dimension

        # unit-normalize
        norm = torch.norm(output, 2, 1, keepdim=True)
        output = output / norm

        return output


class VGGVox(nn.Module):
    """VGGVox implementation

    Reference
    ---------
    Arsha Nagrani, Joon Son Chung, Andrew Zisserman. "VoxCeleb: a large-scale
    speaker identification dataset."

    """

    @staticmethod
    def _output_shape(input_shape, kernel_size, stride=1, padding=0, dilation=1):
        """Predict output shape"""

        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(stride, tuple):
            stride = (stride, stride)
        if not isinstance(padding, tuple):
            padding = (padding, padding)
        if not isinstance(dilation, tuple):
            dilation = (dilation, dilation)

        h_in, w_in = input_shape

        h_out = (h_in + 2 * padding[0] - dilation[0] * (kernel_size[0]- 1) - 1)
        h_out = h_out / stride[0] + 1

        w_out = (w_in + 2 * padding[1] - dilation[1] * (kernel_size[1]- 1) - 1)
        w_out = w_out / stride[1] + 1

        return int(math.floor(h_out)), int(math.floor(w_out))

    def __init__(self, n_features, output_dim=256):

        if n_features < 97:
            msg = (f'VGGVox expects features with at least 97 dimensions '
                   f'(here, n_features = {n_features:d})')
            raise ValueError(msg)

        super(VGGVox, self).__init__()
        self.n_features = n_features
        self.output_dim = output_dim

        h = self.n_features  # 512 in VoxCeleb paper. 201 in practice.
        w = 301 # typically 3s with 10ms steps

        self.conv1_ = nn.Conv2d(1, 96, (7, 7), stride=(2, 2), padding=1)
        # 254 x 148 when n_features = 512
        # 99 x 148 when n_features = 201
        h, w = self._output_shape((h, w), (7, 7), stride=(2, 2), padding=1)

        self.bn1_ = nn.BatchNorm2d(96)
        self.mpool1_ = nn.MaxPool2d((3, 3), stride=(2, 2))
        # 126 x 73 when n_features = 512
        # 49 x 73 when n_features = 201
        h, w = self._output_shape((h, w), (3, 3), stride=(2, 2))

        self.conv2_ = nn.Conv2d(96, 256, (5, 5), stride=(2, 2), padding=1)
        # 62 x 36 when n_features = 512
        # 24 x 36 when n_features = 201
        h, w = self._output_shape((h, w), (5, 5), stride=(2, 2), padding=1)

        self.bn2_ = nn.BatchNorm2d(256)
        self.mpool2_ = nn.MaxPool2d((3, 3), stride=(2, 2))
        # 30 x 17 when n_features = 512
        # 11 x 17 when n_features = 201
        h, w = self._output_shape((h, w), (3, 3), stride=(2, 2))

        self.conv3_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
        # 30 x 17 when n_features = 512
        # 11 x 17 when n_features = 201
        h, w = self._output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

        self.bn3_ = nn.BatchNorm2d(256)

        self.conv4_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
        # 30 x 17 when n_features = 512
        # 11 x 17 when n_features = 201
        h, w = self._output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

        self.bn4_ = nn.BatchNorm2d(256)

        self.conv5_ = nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=1)
        # 30 x 17 when n_features = 512
        # 11 x 17 when n_features = 201
        h, w = self._output_shape((h, w), (3, 3), stride=(1, 1), padding=1)

        self.bn5_ = nn.BatchNorm2d(256)

        self.mpool5_ = nn.MaxPool2d((5, 3), stride=(3, 2))
        # 9 x 8 when n_features = 512
        # 3 x 8 when n_features = 201
        h, w = self._output_shape((h, w), (5, 3), stride=(3, 2))

        self.fc6_ = nn.Conv2d(256, 4096, (h, 1), stride=(1, 1))
        # 1 x 8
        h, w = self._output_shape((h, w), (h, 1), stride=(1, 1))

        self.fc7_ = nn.Linear(4096, 1024)
        self.fc8_ = nn.Linear(1024, self.output_dim)


    def forward(self, sequences):
        """Embed sequences

        Parameters
        ----------
        sequences : torch.Tensor (batch_size, n_samples, n_features)
            Batch of sequences.

        Returns
        -------
        embeddings : torch.Tensor (batch_size, output_dim)
            Batch of embeddings.
        """

        # reshape batch to (batch_size, n_channels, n_features, n_samples)
        batch_size, n_samples, n_features = sequences.size()

        if n_features != self.n_features:
            msg = (f'Mismatch in feature dimension '
                   f'(should be: {self.n_features:d}, is: {n_features:d})')
            raise ValueError(msg)

        if n_samples < 65:
            msg = (f'VGGVox expects sequences with at least 65 samples '
                   f'(here, n_samples = {n_samples:d})')
            raise ValueError(msg)

        x = torch.transpose(sequences, 1, 2).view(
            batch_size, 1, n_features, n_samples)

        # conv1. shape => 254 x 148 => 126 x 73
        x = self.mpool1_(F.relu(self.bn1_(self.conv1_(x))))

        # conv2. shape =>
        x = self.mpool2_(F.relu(self.bn2_(self.conv2_(x))))

        # conv3. shape = 62 x 36
        x = F.relu(self.bn3_(self.conv3_(x)))

        # conv4. shape = 30 x 17
        x = F.relu(self.bn4_(self.conv4_(x)))

        # conv5. shape = 30 x 17
        x = self.mpool5_(F.relu(self.bn5_(self.conv5_(x))))

        # fc6. shape =
        x = F.dropout(F.relu(self.fc6_(x)))

        # (average) temporal pooling. shape =
        x = torch.mean(x, dim=-1)

        # fc7. shape =
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc7_(x)))

        # fc8. shape =
        x = self.fc8_(x)

        return x


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
        [64, 64, 64], i.e. three recurrent layers with hidden dimension of 64.
    bidirectional : bool, optional
        Use bidirectional recurrent layers. Defaults to False, i.e. use
        mono-directional RNNs.
    pooling : {'sum', 'max'}
        Temporal pooling strategy. Defaults to 'sum'.
    batch_normalize : boolean, optional
        Set to False to not apply batch normalization before embedding
        normalization. Defaults to True.
    normalize : {False, 'sphere', 'ball', 'ring'}, optional
        Normalize embeddings.
    weighted : bool, optional
        Add dimension-wise trainable weights. Defaults to False.
    linear : list, optional
        List of hidden dimensions of linear layers. Defaults to none.
    attention : list of int, optional
        List of hidden dimensions of attention linear layers (e.g. [16, ]).
        Defaults to no attention.

    Usage
    -----
    >>> model = ClopiNet(n_features)
    >>> embedding = model(sequence)
    """

    def __init__(self, n_features,
                 rnn='LSTM', recurrent=[64, 64, 64], bidirectional=False,
                 pooling='sum', batch_normalize=True, normalize=False,
                 weighted=False, linear=None, attention=None):

        super(ClopiNet, self).__init__()

        self.n_features = n_features
        self.rnn = rnn
        self.recurrent = recurrent
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.batch_normalize = batch_normalize
        self.normalize = normalize
        self.weighted = weighted
        self.linear = [] if linear is None else linear
        self.attention = [] if attention is None else attention

        self.num_directions_ = 2 if self.bidirectional else 1

        if self.pooling not in {'sum', 'max'}:
            raise ValueError('"pooling" must be one of {"sum", "max"}')

        # create list of recurrent layers
        self.recurrent_layers_ = []
        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.recurrent):
            if self.rnn == 'LSTM':
                recurrent_layer = nn.LSTM(input_dim, hidden_dim,
                                          bidirectional=self.bidirectional,
                                          batch_first=True)
            elif self.rnn == 'GRU':
                recurrent_layer = nn.GRU(input_dim, hidden_dim,
                                         bidirectional=self.bidirectional,
                                         batch_first=True)
            else:
                raise ValueError('"rnn" must be one of {"LSTM", "GRU"}.')
            self.add_module('recurrent_{0}'.format(i), recurrent_layer)
            self.recurrent_layers_.append(recurrent_layer)
            input_dim = hidden_dim * (2 if self.bidirectional else 1)

        # the output of recurrent layers are concatenated so the input
        # dimension of subsequent linear layers is the sum of their output
        # dimension
        input_dim = sum(self.recurrent) * (2 if self.bidirectional else 1)

        if self.weighted:
            self.alphas_ = nn.Parameter(torch.ones(input_dim))

        # create list of linear layers
        self.linear_layers_ = []
        for i, hidden_dim in enumerate(self.linear):
            linear_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('linear_{0}'.format(i), linear_layer)
            self.linear_layers_.append(linear_layer)
            input_dim = hidden_dim

        # batch normalization ~= embeddings whitening.
        if self.batch_normalize:
            self.batch_norm_ = nn.BatchNorm1d(input_dim, eps=1e-5,
                                              momentum=0.1, affine=False)

        if self.normalize in {'ball', 'ring'}:
            self.norm_batch_norm_ = nn.BatchNorm1d(1, eps=1e-5, momentum=0.1,
                                                   affine=False)

        # create attention layers
        self.attention_layers_ = []
        if not self.attention:
            return

        input_dim = self.n_features
        for i, hidden_dim in enumerate(self.attention):
            attention_layer = nn.Linear(input_dim, hidden_dim, bias=True)
            self.add_module('attention_{0}'.format(i), attention_layer)
            self.attention_layers_.append(attention_layer)
            input_dim = hidden_dim
        if input_dim > 1:
            attention_layer = nn.Linear(input_dim, 1, bias=True)
            self.add_module('attention_{0}'.format(len(self.attention)),
                            attention_layer)
            self.attention_layers_.append(attention_layer)

    @property
    def output_dim(self):
        if self.linear:
            return self.linear[-1]
        return sum(self.recurrent) * (2 if self.bidirectional else 1)

    def forward(self, sequence):

        packed_sequences = isinstance(sequence, PackedSequence)

        if packed_sequences:
            _, n_features = sequence.data.size()
            batch_size = sequence.batch_sizes[0].item()
            device = sequence.data.device
        else:
            # check input feature dimension
            batch_size, _, n_features = sequence.size()
            device = sequence.device

        if n_features != self.n_features:
            msg = 'Wrong feature dimension. Found {0}, should be {1}'
            raise ValueError(msg.format(n_features, self.n_features))

        output = sequence

        if self.weighted:
            self.alphas_ = self.alphas_.to(device)

        outputs = []
        # stack recurrent layers
        for hidden_dim, layer in zip(self.recurrent, self.recurrent_layers_):

            if self.rnn == 'LSTM':
                # initial hidden and cell states
                h = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                c = torch.zeros(self.num_directions_, batch_size, hidden_dim,
                                device=device, requires_grad=False)
                hidden = (h, c)

            elif self.rnn == 'GRU':
                # initial hidden state
                hidden = torch.zeros(
                    self.num_directions_, batch_size, hidden_dim,
                    device=device, requires_grad=False)

            # apply current recurrent layer and get output sequence
            output, _ = layer(output, hidden)

            outputs.append(output)

        if packed_sequences:
            outputs, lengths = zip(*[pad_packed_sequence(o, batch_first=True)
                                     for o in outputs])

        # concatenate outputs
        output = torch.cat(outputs, dim=2)
        # batch_size, n_samples, dimension

        if self.weighted:
            output = output * self.alphas_

        # stack linear layers
        for hidden_dim, layer in zip(self.linear, self.linear_layers_):

            # apply current linear layer
            output = layer(output)

            # apply non-linear activation function
            output = F.tanh(output)

        # n_samples, batch_size, dimension

        if self.attention_layers_:
            attn = sequence
            for layer, hidden_dim in zip(self.attention_layers_,
                                         self.attention + [1]):
                attn = layer(attn)
                attn = F.tanh(attn)

            if packed_sequences:
                msg = ('attention is not yet implemented '
                       'for variable length sequences.')
                raise NotImplementedError(msg)
            attn = F.softmax(attn, dim=1)
            output = output * attn

        # average temporal pooling
        if self.pooling == 'sum':
            output = output.sum(dim=1)
        elif self.pooling == 'max':
            if packed_sequences:
                msg = ('"max" pooling is not yet implemented '
                       'for variable length sequences.')
                raise NotImplementedError(msg)
            output, _ = output.max(dim=1)

        # batch_size, dimension

        # batch normalization
        if self.batch_normalize:
            output = self.batch_norm_(output)

        if self.normalize:
            norm = torch.norm(output, 2, 1, keepdim=True)

        if self.normalize == 'sphere':
            output = output / norm

        elif self.normalize == 'ball':
            output = output / norm * F.sigmoid(self.norm_batch_norm_(norm))

        elif self.normalize == 'ring':
            norm_ = self.norm_batch_norm_(norm)
            output = output / norm * (1 + F.sigmoid(self.norm_batch_norm_(norm)))

        # batch_size, dimension

        return output
