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
from ..base_autograd import value_and_multigrad
from autograd import numpy as ag_np

import numpy as np
import h5py

from pyannote.generators.indices import random_label_index
from pyannote.generators.batch import batchify

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Lambda
from pyannote.audio.optimizers import SSMORMS3


class CenterLoss(SequenceEmbeddingAutograd):
    """

    loss = d(anchor, center) - d(anchor, other_center)

    * 'positive' clamping >= 0: loss = max(0, loss + margin)
    * 'sigmoid' clamping [0, 1]: loss = sigmoid(10 * (loss - margin))

    Parameters
    ----------
    margin: float, optional
        Defaults to 0.0
    clamp: {None, 'positive', 'sigmoid'}, optional
        If 'positive', loss = max(0, loss + margin).
        If 'sigmoid' (default), loss = sigmoid(10 * (loss - margin)).
    metric : {'sqeuclidean', 'euclidean', 'cosine', 'angular'}, optional
    per_fold : int, optional
        Number of speakers per batch. Defaults to 20.
    per_label : int, optional
        Number of sequences per speaker. Defaults to 3.
    update_centers : {'batch', 'all'}
        Whether to only update centers in current 'batch' (default), or to
        update 'all' centers (even though they are not part of current batch).
    """

    def __init__(self, metric='angular',
                 margin=0.0, clamp='sigmoid',
                 per_label=3, per_fold=20,
                 update_centers='batch'):

        self.margin = margin
        self.clamp = clamp
        self.per_label = per_label
        self.per_fold = per_fold
        self.update_centers = update_centers
        super(CenterLoss, self).__init__(metric=metric)

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
        n_classes = len(unique)

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

        batches_per_epoch = n_classes // self.per_fold + 1

        return {'batch_generator': batch_generator,
                'batches_per_epoch': batches_per_epoch,
                'n_classes': n_classes}

    def on_train_begin(self, logs=None):

        # dimension of embedding space
        output_dim = self.model.output_shape[-1]
        # number of classes
        n_classes = logs['n_classes']

        # centers model
        trigger = Input(shape=(n_classes, ), name="trigger")
        x = Dense(output_dim, activation='linear', name='dense')(trigger)
        centers = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                         output_shape=(output_dim, ),
                         name="centers")(x)

        self.centers_ = Model(inputs=trigger, outputs=centers)
        self.centers_.compile(optimizer=SSMORMS3(),
                              loss=self._gradient_loss)
        self.trigger_ = np.eye(n_classes)
        self.fC_ = self.centers_.predict(self.trigger_)

    def on_batch_end(self, batch_index, logs=None):
        self.centers_.train_on_batch(self.trigger_,
                                     logs['fC_grad'])

        self.fC_ = self.centers_.predict(self.trigger_)

    def loss(self, fX, fC, y):
        """Differentiable loss

        Parameters
        ----------
        fX : (batch_size, n_dimensions) numpy array
            Embeddings.
        fC : (n_classes, n_dimensions) numpy array
            Centers.
        y : (batch_size, ) numpy array
            Labels.

        Returns
        -------
        loss : float
            Loss.
        """

        loss = 0.
        n_comparisons = 0

        # compute distances between embeddings and centers
        distance = self.metric_(fX, other_embedding=self.fC_)

        # compare to every center...
        if self.update_centers == 'all':
            centers = list(range(self.fC_.shape[0]))

        # or just to the ones in current batch
        elif self.update_centers == 'batch':
            centers = list(np.unique(y))

        # consider every embedding as anchor
        for anchor, y_anchor in enumerate(y):

            # anchor is the index of current embedding
            # y_anchor is the index of corresponding center

            for y_center in centers:

                if y_center == y_anchor:
                    continue

                # y_center is the index of another center

                loss_ = distance[anchor, y_anchor] - \
                        distance[anchor, y_center]

                if self.clamp == 'positive':
                    loss_ = loss_ + self.margin * self.metric_max_
                    loss_ = ag_np.maximum(loss_, 0.)

                elif self.clamp == 'sigmoid':
                    loss_ = loss_ - self.margin * self.metric_max_
                    loss_ = 1. / (1. + ag_np.exp(-10. * loss_))

                # do not use += because autograd does not support it
                loss = loss + loss_

                n_comparisons = n_comparisons + 1

        return loss

    def loss_and_grad(self, batch, embed):

        X = batch['X']
        y = batch['y']

        fX = embed(X)

        func = value_and_multigrad(self.loss, argnums=[0, 1])

        loss, (fX_grad, fC_grad) = func(fX, self.fC_, y)

        return {'loss': loss,
                'gradient': fX_grad,
                'fC_grad': fC_grad}
