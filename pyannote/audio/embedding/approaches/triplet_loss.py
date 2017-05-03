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


from ..base_autograd import SequenceEmbeddingAutograd
from autograd import numpy as ag_np

import numpy as np
import h5py

from pyannote.generators.indices import random_label_index
from pyannote.generators.batch import batchify


class TripletLoss(SequenceEmbeddingAutograd):
    """

    loss = d(anchor, positive) - d(anchor, negative) + margin

    * 'positive' clamping >= 0: loss = max(0, loss)
    * 'sigmoid' clamping [0, 1]: loss = sigmoid(loss)

    Parameters
    ----------
    margin: float, optional
        Defaults to 0.1
    clamp: {None, 'positive', 'sigmoid'}, optional
        If 'positive', loss = max(0, loss)
        If 'sigmoid', loss = sigmoid(loss)
    metric : {'sqeuclidean', 'euclidean', 'cosine', 'angular'}, optional
    per_fold : int, optional
        Number of speakers per batch. Defaults to 30.
    per_label : int, optional
        Number of sequences per speaker. Defaults to 3.
    """

    def __init__(self, margin=0.1, clamp=None, metric='cosine',
                 per_label=3, per_fold=30):
        self.margin = margin
        self.clamp = clamp
        self.per_label = per_label
        self.per_fold = per_fold
        super(TripletLoss, self).__init__(metric=metric)

    def get_batch_generator(self, data_h5):
        """Get batch generator

        Parameters
        ----------
        data_h5 : str
            Path to HDF5 file containing precomputed sequences.
            It must have to aligned datasets 'X' and 'y'.

        Returns
        -------
        batch_generator : iterable
        batches_per_epoch : int

        """

        fp = h5py.File(data_h5, mode='r')
        h5_X = fp['X']
        h5_y = fp['y']

        # keep track of number of labels and rename labels to integers
        unique, y = np.unique(h5_y, return_inverse=True)
        n_labels = len(unique)

        index_generator = random_label_index(
            y, per_label=self.per_label, return_label=False)

        def generator():
            while True:
                i = next(index_generator)
                yield {'X': h5_X[i], 'y': y[i]}

        signature = {'X': {'type': 'sequence'},
                     'y': {'type': 'sequence'}}
        batch_generator = batchify(generator(),
                                   signature,
                                   batch_size=self.per_label * self.per_fold)

        batches_per_epoch = n_labels // self.per_fold + 1

        return {'batch_generator': batch_generator,
                'batches_per_epoch': batches_per_epoch,
                'n_classes': n_labels}

    def loss(self, fX, y):
        """Differentiable loss

        Parameters
        ----------
        fX : (batch_size, n_dimensions) numpy array
            Embeddings.
        y : (batch_size, ) numpy array
            Labels.

        Returns
        -------
        loss : float
            Loss.
        """

        loss = 0.
        n_comparisons = 0

        distance = self.metric_(fX)

        # consider every embedding as anchor
        for anchor, y_anchor in enumerate(y):

            # consider every other embedding with the same label as positive
            for positive, y_positive in enumerate(y):

                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue

                for negative, y_negative in enumerate(y):

                    # if same label, skip
                    if y_negative == y_positive:
                        continue

                    loss_ = distance[anchor, positive] - \
                            distance[anchor, negative] + \
                            self.margin * self.metric_max_

                    if self.clamp == 'positive':
                        loss_ = ag_np.maximum(loss_, 0.)

                    elif self.clamp == 'sigmoid':
                        loss_ = 1. / (1. + ag_np.exp(-loss_))

                    # do not use += because autograd does not support it
                    loss = loss + loss_

                    n_comparisons = n_comparisons + 1

        return loss / n_comparisons
