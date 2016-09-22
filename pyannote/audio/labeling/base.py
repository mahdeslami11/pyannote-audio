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

import os.path

from pyannote.audio.callback import LoggingCallback
from keras.models import model_from_yaml


class SequenceLabeling(object):
    """Sequence labeling

    Parameters
    ----------
    design_model : function or callable
        This function should take input_shape as input and return a Keras model
        that takes a sequence as input, and return the labels as output.
    optimizer: str, optional
        Keras optimizer. Defaults to 'rmsprop'.
    log_dir: str, optional
        When provided, log status after each epoch into this directory. This
        will create several files, including loss plots and weights files.

    See also
    --------
    An example of `design_model` can be found in
    pyannote.audio.labeling.models.StackedLSTM.__call__
    """
    def __init__(self, design_model=None, optimizer='rmsprop', log_dir=None):
        super(SequenceLabeling, self).__init__()
        self.design_model = design_model
        self.optimizer = optimizer
        self.log_dir = log_dir

    @classmethod
    def from_disk(cls, architecture, weights):
        """Load pre-trained sequence labeling from disk

        Parameters
        ----------
        architecture : str
            Path to architecture file (e.g. created by `to_disk` method)
        weights : str
            Path to pre-trained weight file (e.g. created by `to_disk` method)

        Returns
        -------
        sequence_labeling : SequenceLabeling
            Pre-trained sequence labeling model.
        """
        self = SequenceLabeling()

        with open(architecture, 'r') as fp:
            yaml_string = fp.read()
        self.model_ = model_from_yaml(yaml_string)
        self.model_.load_weights(weights)
        return self

    def to_disk(self, architecture=None, weights=None, overwrite=False):
        """Save trained sequence labeling to disk

        Parameters
        ----------
        architecture : str, optional
            When provided, path where to save architecture.
        weights : str, optional
            When provided, path where to save weights
        overwrite : boolean, optional
            Overwrite (architecture or weights) file in case they exist.
        """

        if not hasattr(self, 'model_'):
            raise AttributeError('Model must be trained first.')

        if architecture and os.path.isfile(architecture) and not overwrite:
            raise ValueError("File '{architecture}' already exists.".format(architecture=architecture))

        if weights and os.path.isfile(weights) and not overwrite:
            raise ValueError("File '{weights}' already exists.".format(weights=weights))

        if architecture:
            yaml_string = self.model_.to_yaml()
            with open(architecture, 'w') as fp:
                fp.write(yaml_string)

        if weights:
            self.model_.save_weights(weights, overwrite=overwrite)

    def fit(self, input_shape, generator,
            samples_per_epoch, nb_epoch, callbacks=[]):
        """Train the model

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence
        generator : iterable
            Batch generator
        samples_per_epoch : int
            Number of samples to process before going to the next epoch.
        nb_epoch : int
            Total number of iterations on the data
        callbacks : list, optional
            List of callbacks to be called during training.
            Defaults to [LoggingCallback()]

        See also
        --------
        keras.engine.training.Model.fit_generator
        """

        if not callbacks and self.log_dir:
            default_callback = LoggingCallback(self, log_dir=self.log_dir)
            callbacks = [default_callback]

        self.model_ = self.design_model(input_shape)
        self.model_.compile(optimizer=self.optimizer,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

        self.model_.fit_generator(
            generator, samples_per_epoch, nb_epoch,
            verbose=1, callbacks=callbacks)

    def predict(self, sequence, batch_size=32, verbose=0):
        """Apply pre-trained labeling to sequences

        Parameters
        ----------
        sequences : (n_samples, n_frames, n_features) array
            Array of sequences.
        batch_size : int, optional
            Number of samples per batch
        verbose : int, optional

        Returns
        -------
        labels : (n_samples, n_frames, n_classes) array
        """
        return self.model_.predict(
            sequence, batch_size=batch_size, verbose=verbose)
