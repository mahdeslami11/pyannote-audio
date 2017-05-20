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
from autograd.core import primitive

import numpy as np
import functools
import pickle
import h5py

import keras.backend as K
import keras.callbacks as cbks
import keras.models

from pyannote.audio.callback import LoggingCallback
from pyannote.audio.callback import BaseLogger
from pyannote.audio.callback import Debugging

# populate CUSTOM_OBJECTS with user-defined optimizers and layers
import pyannote.audio.optimizers
import pyannote.audio.embedding.models
from pyannote.audio.embedding.losses import precomputed_gradient_loss
from pyannote.audio.keras_utils import CUSTOM_OBJECTS

from pyannote.generators.batch import batchify

EPSILON = 1e-6

@primitive
def arccos(x):
    return np.arccos(np.clip(x, -1., 1))
def arccos_vjp(g, ans, vs, gvs, x):
    return -g / np.sqrt(np.maximum(1. - x**2, EPSILON))
arccos.defvjp(arccos_vjp)


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
    def l2_normalize(embedding):
        norm = ag_np.sqrt(ag_np.sum(embedding ** 2, axis=-1))
        return (embedding.T / norm).T

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

        return arccos(ag_np.stack(
            ag_np.sum(embedding[i] * other_embedding, axis=1)
            for i in range(n_samples)))

class SequenceEmbeddingAutograd(MixinDistanceAutograd, cbks.Callback):
    """Base class for sequence embedding

    Parameters
    ----------
    metric : {'sqeuclidean', 'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'sqeuclidean'.
    gradient_factor : float, optional
        Multiply gradient by this number. Defaults to 1.
    batch_size : int, optional
        Internal batch size. Defaults to 32.
    """

    def __init__(self, metric='cosine', gradient_factor=1., batch_size=32):
        super(SequenceEmbeddingAutograd, self).__init__()
        self.metric = metric
        self.gradient_factor = gradient_factor
        self.batch_size = batch_size

        self.metric_ = getattr(self, metric)
        self.metric_max_ = self.get_metric_max(metric)

        self.float_autograd_ = ag_np.array(0.).dtype.name
        self.float_backend_ = K.floatx()


    @classmethod
    def restart(cls, log_dir, epoch):

        weights_h5 = LoggingCallback.WEIGHTS_H5.format(log_dir=log_dir,
                                                       epoch=epoch)

        # TODO update this code once keras > 2.0.4 is released
        try:
            embedding = keras.models.load_model(
                weights_h5, custom_objects=CUSTOM_OBJECTS,
                compile=True)
        except TypeError as e:
            embedding = keras.models.load_model(
                weights_h5, custom_objects=CUSTOM_OBJECTS)

        embedding.epoch = epoch

        return embedding

    def embed(self, embedding, X, internal=False):
        """Apply embedding on sequences

        Parameters
        ----------
        embedding : keras.Model
            Current state of embedding network
        X : (n_sequences, n_samples, n_features) numpy array
            Batch of input sequences
        internal : bool, optional
            Set to True to return internal representation

        Returns
        -------
        fX : (n_sequences, ...) numpy array
            Batch of embeddings.

        """

        if internal:

            embed = K.function(
                [embedding.get_layer(name='input').input, K.learning_phase()],
                [embedding.get_layer(name='internal').output])

            # split large batch in smaller batches if needed
            if len(X) > self.batch_size:
                batch_generator = batchify(iter(X), {'type': 'ndarray'},
                                           batch_size=self.batch_size,
                                           incomplete=True)
                fX = np.vstack(embed([x, 0])[0] for x in batch_generator)
            else:
                fX = embed([X, 0])[0]

        else:
            fX = embedding.predict(X, batch_size=self.batch_size)

        return fX.astype(self.float_autograd_)


    def fit(self, init_embedding, batch_generator, batches_per_epoch,
            n_classes=None, epochs=1000, log_dir=None, optimizer='rmsprop'):
        """

        Parameters
        ----------
        init_embedding : callable or keras.Model
            If callable, takes 'input_shape' as input and returns a Keras model
            that takes a sequence as input, and returns the embedding as output.
            If keras.Model, it must already be compiled.
        batch_generator : (infinite) iterable
            Generates (dict) batches such as batch['X'] is a numpy array of
            shape (variable_batch_size, n_samples, n_features).
        batches_per_epoch : int
            Number of batches per epoch.
        n_classes : int
        log_dir : str
        optimizer: str, optional
            Keras optimizer. Defaults to 'rmsprop'.
        """

        # consume batch generator once
        batch = next(batch_generator)

        if isinstance(init_embedding, keras.models.Model):
            embedding = init_embedding

            init_epoch = embedding.epoch
            restart = True

        else:

            # infer batch size and input shape
            batch_size, n_samples, n_features = batch['X'].shape

            # initialize embedding
            input_shape = (n_samples, n_features)
            embedding = init_embedding(input_shape)
            embedding.compile(optimizer=optimizer,
                              loss=precomputed_gradient_loss)

            init_epoch = -1
            restart = False

        callbacks = [self]
        callbacks.append(Debugging())
        callbacks.append(BaseLogger())
        callbacks.append(cbks.ProgbarLogger(count_mode='steps'))
        if log_dir is not None:
            callbacks.append(LoggingCallback(log_dir, restart=restart))
        self.history = cbks.History()
        callbacks.append(self.history)

        callbacks = cbks.CallbackList(callbacks)
        callbacks.set_model(embedding)
        callbacks.set_params({
            'epochs': epochs,
            'steps': batches_per_epoch,
            'verbose': True,
            'metrics': ['loss'],
        })

        callbacks.on_train_begin(logs={'n_classes': n_classes,
                                       'log_dir': log_dir,
                                       'restart': restart,
                                       'epoch': init_epoch})

        for epoch in range(init_epoch + 1, epochs):

            epoch_logs = {'log_dir': log_dir, 'restart': restart}
            callbacks.on_epoch_begin(epoch, logs=epoch_logs)

            for batch_index in range(batches_per_epoch):

                batch_size, _, _ = batch['X'].shape
                batch_logs = {'batch': batch_index,
                              'size': batch_size,
                              'epoch': epoch,
                              'log_dir': log_dir}

                callbacks.on_batch_begin(batch_index, logs=batch_logs)

                # compute loss and its gradient
                logs = self.loss_and_grad(batch, embedding)
                batch_logs.update(logs)

                # can anyone tell me why this usually works better
                # when gradient_factor is large?
                gradient = self.gradient_factor * batch_logs['gradient']

                X = batch['X'].astype(self.float_backend_)
                y = gradient.astype(self.float_backend_)

                # split large batch in smaller batches if needed
                if len(X) > self.batch_size:
                    signature = ({'type': 'ndarray'}, {'type': 'ndarray'})
                    sub_batch_generator = batchify(zip(X, y), signature,
                                                   batch_size=self.batch_size,
                                                   incomplete=True)
                else:
                    sub_batch_generator = [(X, y)]

                # backprop
                for X, y in sub_batch_generator:
                    embedding.train_on_batch(X, y)

                callbacks.on_batch_end(batch_index, logs=batch_logs)

                # next batch
                batch = next(batch_generator)

            callbacks.on_epoch_end(epoch, logs=epoch_logs)

        callbacks.on_train_end()

        return self.history
