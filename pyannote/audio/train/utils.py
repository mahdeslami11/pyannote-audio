#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_sequence


def operator_packed(func, seq):
    """Same as func(seq) except it supports `PackedSequence` instances

    Parameters
    ----------
    seq : tuple
        List of (batch_size, n_samples, *) `torch.Tensor` or `PackedSequence`.
        They must all share the same batch size, and have consistent number of
        samples.

    Returns
    -------
    result : `torch.Tensor` or `PackedSequence`

    """
    if all(not isinstance(packed, PackedSequence) for packed in seq):
        return func(seq)

    paddeds, lengths = zip(*(pad_packed_sequence(packed, batch_first=True)
                             for packed in seq))

    return pack_sequence([
        func([p[:l].expand(1, -1, -1) for p in sequences])[0]
        for sequences, l in zip(zip(*paddeds), lengths[0])
    ])


def map_packed(func, sequences):
    """Same as func(sequences) except it supports `PackedSequence` instances

    It assumes that `func` returns a batch of sequences.

    Parameters
    ----------
    func : callable
        Function that takes a batch of sequence as input and returns a batch of
        sequences as output.
    sequences : (batch_size, n_samples, n_features) `torch.Tensor`
                or `PackedSequence`
        Batch of sequences.

    Returns
    -------
    mapped : (batch_size, *, *) `torch.Tensor` or `PackedSequence`
        Batch of mapped sequences.

    Example
    -------
    # apply instance normalization on each sequence
    >>> sequences = pack_sequence(...)
    >>> func = lambda b: F.instance_norm(b.transpose(1, 2)).transpose(1, 2)
    >>> mapped = map_packed(func, sequences)
    """

    if not isinstance(sequences, PackedSequence):
        return func(sequences)

    padded, length = pad_packed_sequence(sequences, batch_first=True)

    return pack_sequence([func(p[:l].expand(1, -1, -1))[0]
                          for p, l in zip(padded, length)])


def pool_packed(func, sequences):
    """Same as func(sequences) except it supports `PackedSequence` instances

    It assumes that `func` does temporal pooling.

    Parameters
    ----------
    func : callable
        Function that takes a batch of sequence as input and applies temporal
        pooling.

    sequences : (batch_size, n_samples, n_features) `torch.Tensor`
                or `PackedSequence`
        Batch of sequences.

    Returns
    -------
    pooled : (batch_size, n_features) `torch.Tensor`
        Batch of time-pooled sequences.

    Example
    -------
    # apply max pooling on each sequence
    >>> sequences = pack_sequence(...)
    >>> func = lambda batch: batch.max(dim=1)[0]
    >>> pooled = pool_packed(func, sequences)
    """

    if not isinstance(sequences, PackedSequence):
        return func(sequences)

    padded, length = pad_packed_sequence(sequences, batch_first=True)

    return torch.cat([func(p[:l].expand(1, -1, -1))
                      for p, l in zip(padded, length)])


def get_info(sequences):
    """Get info about batch of sequences

    Parameters
    ----------
    sequences : `torch.Tensor` or `PackedSequence`
        Batch of sequences given as a `torch.Tensor` of shape
        (batch_size, n_samples, n_features) if sequences all share the same
        length, or as a `PackedSequence` if they do not.

    Returns
    -------
    batch_size : `int`
        Number of sequences in batch.
    n_features : `int`
        Number of features.
    device : `torch.device`
        Device.
    """

    packed_sequences = isinstance(sequences, PackedSequence)

    if packed_sequences:
        _, n_features = sequences.data.size()
        batch_size = sequences.batch_sizes[0].item()
        device = sequences.data.device
    else:
        # check input feature dimension
        batch_size, _, n_features = sequences.size()
        device = sequences.device

    return batch_size, n_features, device
