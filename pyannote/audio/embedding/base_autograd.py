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

from autograd import numpy as ag_np
from autograd import value_and_grad
import numpy as np
import keras.backend as K
import keras.callbacks as cbks
import h5py
from pyannote.audio.callback import LoggingCallback
import functools

EPSILON = 1e-6


def value_and_multigrad(fun, argnums=[0]):
    """Takes gradients wrt multiple arguments simultaneously."""
    def combined_arg_fun(multi_arg, *args, **kwargs):
        extra_args_list = list(args)
        for argnum_ix, arg_ix in enumerate(argnums):
            extra_args_list[arg_ix] = multi_arg[argnum_ix]
        return fun(*extra_args_list, **kwargs)
    gradfun = value_and_grad(combined_arg_fun, argnum=0)
    def gradfun_rearranged(*args, **kwargs):
        multi_arg = tuple([args[i] for i in argnums])
        return gradfun(multi_arg, *args, **kwargs)
    return gradfun_rearranged


class MixinDistanceAutograd:
    """Differentiable distances between pairs of embeddings"""

    @staticmethod
    def get_metric_max(metric):
        """Maximum distance between two (L2-normalized) embeddings

        Parameters
        ----------
        metric : {'sqeuclidean', 'euclidean', 'cosine', 'angular'}
            Metric name.

        Returns
        -------
        max : float
            Maximum possible distance between two embeddings
        """

        return {'sqeuclidean': 4.,
                'euclidean': 2.,
                'cosine': 2.,
                'angular': np.pi}[metric]

    @staticmethod
    def sqeuclidean(embedding, other_embedding=None):
        """Compute squared euclidean distance

        Parameters
        ----------
        embedding : (n_samples, n_dimension) numpy array
        other_embedding : (n_other_samples, n_dimension, ) numpy array, optional

        Returns
        -------
        distance : (n_samples, n_other_samples) numpy array or float
            Squared euclidean distance
        """

        n_samples, n_dimensions = embedding.shape
        if other_embedding is None:
            other_embedding = embedding

        return ag_np.stack(
            ag_np.sum((embedding[i] - other_embedding) ** 2, axis=1)
            for i in range(n_samples))

    @staticmethod
    def euclidean(embedding, other_embedding=None):
        """Compute euclidean distance

        Parameters
        ----------
        embedding : (n_samples, n_dimension) numpy array
        other_embedding : (n_other_samples, n_dimension, ) numpy array, optional

        Returns
        -------
        distance : (n_samples, n_other_samples) numpy array or float
            Euclidean distance
        """
        return ag_np.sqrt(self.sqeuclidean(
            embedding, other_embedding=other_embedding))

    @staticmethod
    def cosine(embedding, other_embedding=None):
        """Compute cosine distance

        Parameters
        ----------
        embedding : (n_samples, n_dimension) numpy array
        other_embedding : (n_other_samples, n_dimension, ) numpy array, optional
            L2-normalized embeddings

        Returns
        -------
        distance : (n_samples, n_other_samples) numpy array or float
            Cosine distance
        """

        n_samples, n_dimensions = embedding.shape
        if other_embedding is None:
            other_embedding = embedding

        return 1. - ag_np.stack(
            ag_np.sum(embedding[i] * other_embedding, axis=1)
            for i in range(n_samples))

    @staticmethod
    def angular(embedding, other_embedding=None):
        """Compute angular distance

        Parameters
        ----------
        embedding : (n_samples, n_dimension) numpy array
        other_embedding : (n_other_samples, n_dimension, ) numpy array, optional
            L2-normalized embeddings

        Returns
        -------
        distance : (n_samples, n_other_samples) numpy array or float
            Angular distance
        """

        n_samples, _ = embedding.shape

        if other_embedding is None:
            other_embedding = embedding

        return ag_np.arccos(ag_np.clip(ag_np.stack(
            ag_np.sum(embedding[i] * other_embedding, axis=1)
            for i in range(n_samples)), -1. + EPSILON, 1. - EPSILON))


class SequenceEmbeddingAutograd(MixinDistanceAutograd, cbks.Callback):

    def __init__(self, metric='cosine'):
        super(SequenceEmbeddingAutograd, self).__init__()
        self.metric = metric
        self.metric_ = getattr(self, metric)
        self.metric_max_ = self.get_metric_max(metric)

    @staticmethod
    def _gradient_loss(y_true, y_pred):
        return K.sum((y_pred * y_true), axis=-1)

    def embed(self, embedding, X):
        """Apply embedding on sequences

        Parameters
        ----------
        embedding : keras.Model
            Current state of embedding network
        X : (n_sequences, n_samples, n_features) numpy array
            Batch of input sequences

        Returns
        -------
        fX : (n_sequences, ...) numpy array
            Batch of embeddings.

        """
        return embedding.predict(X)

    def fit(self, init_embedding, batch_generator, batches_per_epoch,
            n_classes=None, epochs=1000, log_dir=None, optimizer='rmsprop'):
        """

        Parameters
        ----------
        init_embedding : function or callable
            Takes 'input_shape' as input and returns a Keras model that takes
            a sequence as input, and returns the embedding as output.
        batch_generator : (infinite) iterable
            Generates (dict) batches such as batch['X'] is a numpy array of
            shape (batch_size, n_samples, n_features).
        batches_per_epoch : int
            Number of batches per epoch.
        n_classes : int
        log_dir : str
        optimizer: str, optional
            Keras optimizer. Defaults to 'rmsprop'.

        """

        # consume batch generator once
        batch = next(batch_generator)
        # infer batch size and input shape
        batch_size, n_samples, n_features = batch['X'].shape

        # initialize embedding
        input_shape = (n_samples, n_features)
        embedding = init_embedding(input_shape)
        embedding.compile(optimizer=optimizer,
                          loss=self._gradient_loss)

        embed = functools.partial(self.embed, embedding)

        callbacks = [self]
        callbacks.append(cbks.BaseLogger())
        callbacks.append(cbks.ProgbarLogger(count_mode='steps'))
        if log_dir is not None:
            callbacks.append(LoggingCallback(log_dir))
        self.history = cbks.History()
        callbacks.append(self.history)

        callbacks = cbks.CallbackList(callbacks)
        callbacks.set_model(embedding)
        callbacks.set_params({
            'epochs': epochs,
            # 'samples': batch_size,
            'steps': batches_per_epoch,
            'verbose': True,
            # 'do_validation': False,
            'metrics': ['loss'],
        })

        callbacks.on_train_begin(logs={'n_classes': n_classes})

        for epoch in range(epochs):

            epoch_logs = {}
            callbacks.on_epoch_begin(epoch, logs=epoch_logs)

            for batch_index in range(batches_per_epoch):

                batch_size, _, _ = batch['X'].shape
                batch_logs = {'batch': batch_index,
                              'size': batch_size}

                callbacks.on_batch_begin(batch_index, logs=batch_logs)

                # compute loss and its gradient
                logs = self.loss_and_grad(batch, embed=embed)
                batch_logs.update(logs)

                # backprop
                embedding.train_on_batch(batch['X'], batch_logs['gradient'])

                callbacks.on_batch_end(batch_index, logs=batch_logs)

                # next batch
                batch = next(batch_generator)

            callbacks.on_epoch_end(epoch, logs=epoch_logs)

        callbacks.on_train_end()

        return self.history
