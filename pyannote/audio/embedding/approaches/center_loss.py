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


from .triplet_loss import TripletLoss
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
import keras.models
from pyannote.audio.optimizers import SSMORMS3
from pyannote.audio.embedding.losses import precomputed_gradient_loss
from pyannote.audio.callback import LoggingCallback
from pyannote.audio.keras_utils import CUSTOM_OBJECTS
from pyannote.core.util import pairwise


class CenterLoss(TripletLoss):
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
    learn_to_aggregate : boolean, optional
    """

    WEIGHTS_H5 = LoggingCallback.WEIGHTS_H5[:-3] + '.centers.h5'

    def __init__(self, metric='angular',
                 margin=0.0, clamp='sigmoid',
                 per_label=3, per_fold=20,
                 update_centers='batch',
                 learn_to_aggregate=False):

        super(CenterLoss, self).__init__(
            metric=metric, margin=margin, clamp=clamp,
            per_label=per_label, per_fold=per_fold,
            learn_to_aggregate=learn_to_aggregate)
        self.update_centers = update_centers

    def on_train_begin(self, logs=None):

        # number of classes
        n_classes = logs['n_classes']

        if logs['restart']:

            weights_h5 = self.WEIGHTS_H5.format(log_dir=logs['log_dir'],
                                                epoch=logs['epoch'])

            # TODO update this code once keras > 2.0.4 is released
            try:
                self.centers_ = keras.models.load_model(
                    weights_h5, custom_objects=CUSTOM_OBJECTS,
                    compile=True)
            except TypeError as e:
                self.centers_ = keras.models.load_model(
                    weights_h5, custom_objects=CUSTOM_OBJECTS)

        else:
            # dimension of embedding space
            output_dim = self.model.output_shape[-1]

            # centers model
            trigger = Input(shape=(n_classes, ), name="trigger")
            x = Dense(output_dim, activation='linear', name='dense')(trigger)
            centers = Lambda(lambda x: K.l2_normalize(x, axis=-1),
                             output_shape=(output_dim, ),
                             name="centers")(x)

            self.centers_ = Model(inputs=trigger, outputs=centers)
            self.centers_.compile(optimizer=SSMORMS3(),
                                  loss=precomputed_gradient_loss)

        self.trigger_ = np.eye(n_classes)
        self.fC_ = self.centers_.predict(self.trigger_)

    def on_batch_end(self, batch_index, logs=None):
        self.centers_.train_on_batch(self.trigger_,
                                     logs['center_gradient'])

        self.fC_ = self.centers_.predict(self.trigger_)

    def on_epoch_end(self, epoch, logs=None):
        """Save center weights after each epoch"""

        weights_h5 = self.WEIGHTS_H5.format(log_dir=logs['log_dir'],
                                            epoch=epoch)
        keras.models.save_model(self.centers_, weights_h5,
                                overwrite=logs['restart'],
                                include_optimizer=(epoch % 10 == 0))

        # TODO | plot distribution of distances between centers

    def loss_y(self, fX, fC, y):
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

    def loss_z(self, fX, fC, y, n):
        """Differentiable loss

        Parameters
        ----------
        fX : np.array (n_sequences, n_samples, n_dimensions)
            Stacked groups of internal embeddings.
        fC : (n_classes, n_dimensions) numpy array
            Centers.
        y : (batch_size, ) numpy array
            Label of each group.
        n :  (batch_size, ) numpy array
            Number of sequences per group (np.sum(n) == n_sequences)

        Returns
        -------
        loss : float
            Loss.
        """

        indices = np.hstack([[0], np.cumsum(n)])
        fX_sum = ag_np.stack([ag_np.sum(ag_np.sum(fX[i:j], axis=0), axis=0)
                              for i, j in pairwise(indices)])
        return self.loss_y(self.l2_normalize(fX_sum), fC, y)

    def loss_and_grad(self, batch, embedding):

        if self.learn_to_aggregate:
            fX = self.embed(embedding, batch['X'], internal=True)
            func = value_and_multigrad(self.loss_z, argnums=[0, 1])
            loss, (fX_grad, fC_grad) = func(fX, self.fC_, batch['y'],
                                            batch['n'])
            fX_grad = fX_grad[:, 0, :]

        else:
            fX = self.embed(embedding, batch['X'], internal=False)
            func = value_and_multigrad(self.loss_y, argnums=[0, 1])
            loss, (fX_grad, fC_grad) = func(fX, self.fC_, batch['y'])

        return {'loss': loss,
                'gradient': fX_grad,
                'center_gradient': fC_grad}
