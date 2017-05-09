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

import os.path

from pyannote.audio.callback import LoggingCallback
import keras.models

from pyannote.audio.keras_utils import CUSTOM_OBJECTS


class SequenceLabeling(object):
    """Sequence labeling
    """
    def __init__(self):
        super(SequenceLabeling, self).__init__()

    @classmethod
    def from_disk(cls, log_dir, epoch):
        """Load pre-trained sequence labeling from disk

        Parameters
        ----------
        log_dir : str
        epoch : int

        Returns
        -------
        sequence_labeling : SequenceLabeling
            Pre-trained sequence labeling model.
        """

        self = SequenceLabeling()

        weights_h5 = LoggingCallback.WEIGHTS_H5.format(log_dir=log_dir,
                                                       epoch=epoch)

        # TODO update this code once keras > 2.0.4 is released
        try:
            self.labeling_ = keras.models.load_model(
                weights_h5, custom_objects=CUSTOM_OBJECTS,
                compile=True)
        except TypeError as e:
            self.labeling_ = keras.models.load_model(
                weights_h5, custom_objects=CUSTOM_OBJECTS)

        self.labeling_.epoch = epoch

        return self

    def fit(self, input_shape, design_labeling, generator,
            steps_per_epoch, epochs, loss='categorical_crossentropy',
            optimizer='rmsprop', log_dir=None):
        """Train the model

        Parameters
        ----------
        input_shape : (n_frames, n_features) tuple
            Shape of input sequence
        design_labeling : function or callable
            This function should take input_shape as input and return a Keras
            model that takes a sequence as input, and returns the labeling as
            output.
        generator : iterable
            The output of the generator must be a tuple (inputs, targets) or a
            tuple (inputs, targets, sample_weights). All arrays should contain
            the same number of samples. The generator is expected to loop over
            its data indefinitely. An epoch finishes when `steps_per_epoch`
            samples have been seen by the model.
        steps_per_epoch : int
            Number of batches to process before going to the next epoch.
        epochs : int
            Total number of iterations on the data
        optimizer: str, optional
            Keras optimizer. Defaults to 'rmsprop'.
        log_dir: str, optional
            When provided, log status after each epoch into this directory.
            This will create several files, including loss plots and weights
            files.

        See also
        --------
        keras.engine.training.Model.fit_generator
        """

        callbacks = []

        if log_dir is not None:
            log = [('train', 'loss'), ('train', 'accuracy')]
            callback = LoggingCallback(log_dir, log=log)
            callbacks.append(callback)

        # in case the {generator | optimizer} define their own
        # callbacks, append them as well. this might be useful.
        for stuff in [generator, optimizer]:
            if hasattr(stuff, 'callbacks'):
                callbacks.extend(stuff.callbacks())

        self.labeling_ = design_labeling(input_shape)
        self.labeling_.compile(optimizer=optimizer,
                               loss=loss,
                               metrics=['accuracy'])

        return self.labeling_.fit_generator(
            generator, steps_per_epoch, epochs=epochs,
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
        return self.labeling_.predict(
            sequence, batch_size=batch_size, verbose=verbose)
