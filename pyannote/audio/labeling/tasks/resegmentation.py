#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018-2019 CNRS

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

"""Resegmentation"""

import torch
import tempfile
import collections
import numpy as np
from .base import LabelingTask
from .base import LabelingTaskGenerator
from .base import TASK_MULTI_CLASS_CLASSIFICATION
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.audio.labeling.models import StackedRNN
from pyannote.core.utils.numpy import one_hot_decoding
from pyannote.database import get_unique_identifier
from pyannote.database import get_annotated
from pyannote.audio.labeling.extraction import SequenceLabeling
from pyannote.audio.train.schedulers import ConstantScheduler
from torch.optim import SGD
from pyannote.audio.features import Precomputed
from pyannote.audio.features.utils import get_audio_duration
from pyannote.audio.util import mkdir_p
from pathlib import Path

class ResegmentationGenerator(LabelingTaskGenerator):
    """Batch generator for resegmentation self-training

    Parameters
    ----------
    precomputed : `pyannote.audio.features.Precomputed`
        Precomputed feature extraction
    current_file : `dict`
        # ...
    frame_info : `pyannote.core.SlidingWindow`, optional
        Override `feature_extraction.sliding_window`. This is useful for
        models that include the feature extraction step (e.g. SincNet) and
        therefore output a lower sample rate than that of the input.
        Defaults to `feature_extraction.sliding_window`
    frame_crop : {'center', 'loose', 'strict'}, optional
        Which mode to use when cropping labels. This is useful for models that
        include the feature extraction step (e.g. SincNet) and therefore use a
        different cropping mode. Defaults to 'center'.
    duration : float, optional
        Duration of sub-sequences. Defaults to 3.2s.
    batch_size : int, optional
        Batch size. Defaults to 32.
    """

    def __init__(self, precomputed, current_file,
                 frame_info=None, frame_crop=None,
                 duration=3.2, batch_size=32):

        if 'duration' not in current_file:
            msg = (
                '`current_file` is expected to contain a "duration" key that '
                'provides the audio file duration in seconds.'
            )
            raise ValueError(msg)

        self.current_file = current_file

        super(ResegmentationGenerator, self).__init__(
            precomputed, self.get_dummy_protocol(current_file), subset='train',
            frame_info=frame_info, frame_crop=frame_crop,
            duration=duration, step=0.25*duration, batch_size=batch_size,
            exhaustive=True, shuffle=True, parallel=1)

    def get_dummy_protocol(self, current_file):
        """Get dummy protocol containing only `current_file`

        Parameters
        ----------
        current_file : pyannote.database dict

        Returns
        -------
        protocol : SpeakerDiarizationProtocol instance
            Dummy protocol containing only `current_file` in both train,
            dev., and test sets.

        """

        class DummyProtocol(SpeakerDiarizationProtocol):

            def trn_iter(self):
                yield current_file

            def dev_iter(self):
                yield current_file

            def tst_iter(self):
                yield current_file

        return DummyProtocol()

    def postprocess_y(self, Y):
        """Generate labels for resegmentation

        Parameters
        ----------
        Y : (n_samples, n_speakers) numpy.ndarray
            Discretized annotation returned by
            `pyannote.core.utils.numpy.one_hot_encoding`.

        Returns
        -------
        y : (n_samples, 1) numpy.ndarray
            y[t] = 0 indicates non-speech,
            y[t] = i + 1 indicates speaker i.

        See also
        --------
        `pyannote.core.utils.numpy.one_hot_encoding`
        """

        # +1 because...
        y = np.argmax(Y, axis=1) + 1

        # ... 0 is for non-speech
        non_speech = np.sum(Y, axis=1) == 0
        y[non_speech] = 0

        return np.int64(y)[:, np.newaxis]

    @property
    def specifications(self):
        """Task & sample specifications

        Returns
        -------
        specs : `dict`
            ['task'] (`str`) : task name
            ['X']['dimension'] (`int`) : features dimension
            ['y']['classes'] (`list`) : list of classes
        """

        specs = {
            'task': TASK_MULTI_CLASS_CLASSIFICATION,
            'X': {'dimension': self.feature_extraction.dimension},
            'y': {'classes': ['non_speech'] + self.segment_labels_},
        }

        for key, classes in self.file_labels_.items():
            specs[key] = {'classes': classes}

        return specs

    @property
    def batches_per_epoch(self):
        """Number of batches needed to cover the whole file"""
        duration_per_epoch = 4 * self.current_file['duration']
        duration_per_batch = self.duration * self.batch_size
        return int(np.ceil(duration_per_epoch / duration_per_batch))


class Resegmentation(LabelingTask):
    """Re-segmentation

    Parameters
    ----------
    feature_extraction : `pyannote.audio.features.FeatureExtraction`
        Feature extraction.
    get_model : callable
        Callable that takes `specifications` as input and returns a
        `nn.Module` instance.
    keep_sad: `boolean`, optional
        Keep speech/non-speech state unchanged. Defaults to False.
    epochs : `int`, optional
        (Self-)train for that many epochs. Defaults to 30.
    ensemble : `int`, optional
        Average output of last `ensemble` epochs. Defaults to no ensembling.
    duration : `float`, optional
    batch_size : `int`, optional
    device : `torch.device`, optional
    """

    def __init__(self, feature_extraction, get_model, keep_sad=False,
                 epochs=30, learning_rate=0.1, ensemble=1, device=None,
                 duration=3.2, batch_size=32):

        self.feature_extraction = feature_extraction
        self.get_model = get_model
        self.keep_sad = keep_sad
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.ensemble = ensemble

        self.device = torch.device('cpu') if device is None else device

        super().__init__(duration=duration, batch_size=batch_size)

    def get_batch_generator(self, current_file):
        """Get batch generator for current file

        Parameters
        ----------
        current_file : `dict`
            Dictionary obtained by iterating over a subset of a
            `pyannote.database.Protocol` instance.

        Returns
        -------
        batch_generator : `ResegmentationGenerator`
        """

        if hasattr(self.get_model, 'get_frame_info'):
            frame_info = self.get_model.get_frame_info(**params)
        else:
            frame_info = None

        if hasattr(self.get_model, 'frame_crop'):
            frame_crop = self.get_model.frame_crop
        else:
            frame_crop = None

        return ResegmentationGenerator(
            self.feature_extraction, current_file,
            frame_info=frame_info, frame_crop=frame_crop,
            duration=self.duration, batch_size=self.batch_size)

    def apply(self, current_file, hypothesis=None):
        """Apply resegmentation using self-supervised sequence labeling

        Parameters
        ----------
        current_file : `dict`
            Dictionary obtained by iterating over a subset of a
            `pyannote.database.Protocol` instance.
        hypothesis : `pyannote.core.Annotation`, optional
            Current diarization output. Defaults to current_file['hypothesis'].

        Returns
        -------
        new_hypothesis : `pyannote.core.Annotation`
            Updated diarization output.
        """

        # use current diarization output as "training" labels
        if hypothesis is None:
            hypothesis = current_file['hypothesis']
        current_file = dict(current_file)
        current_file['annotation'] = hypothesis

        # HACK. we shouldn't need to do that here...
        current_file['duration'] = get_audio_duration(current_file)

        batch_generator = self.get_batch_generator(current_file)

        # create a temporary directory to store models and log files
        # it is removed automatically before returning.
        with tempfile.TemporaryDirectory() as log_dir:

            # create log_dir/weights
            mkdir_p(Path(log_dir) / 'weights')

            epochs = self.fit_iter(
                self.get_model, batch_generator,
                restart=0, epochs=self.epochs,
                get_optimizer=SGD,
                get_scheduler=ConstantScheduler,
                learning_rate=self.learning_rate,
                log_dir=log_dir, quiet=True,
                device=self.device)

            scores = []
            for i, current_model in enumerate(epochs):

                # do not compute scores that are not used in later ensembling
                # simply jump to next training epoch
                if i < self.epochs - self.ensemble:
                    continue

                current_model.eval()

                # initialize sequence labeling with model and features
                sequence_labeling = SequenceLabeling(
                    model=current_model,
                    feature_extraction=self.feature_extraction,
                    duration=self.duration, step=.25 * self.duration,
                    batch_size=self.batch_size, device=self.device)

                # compute scores and keep track of them for later ensembling
                scores.append(sequence_labeling(current_file))

                current_model.train()

        # ensemble scores
        scores = np.mean([s.data for s in scores], axis=0)

        # speaker labels
        labels = batch_generator.specifications['y']['classes'][1:]

        # features sliding window
        window = self.feature_extraction.sliding_window

        if self.keep_sad:

            # sequence of most likely speaker index
            # (even when non-speech is the most likely class)
            best_speaker_indices = np.argmax(scores[:, 1:], axis=1) + 1

            # reconstruct annotation
            new_hypothesis = one_hot_decoding(
                best_speaker_indices, window, labels=labels)

            # revert non-speech regions back to original
            speech = hypothesis.get_timeline().support()
            new_hypothesis = new_hypothesis.crop(speech)

        else:

            # sequence of most likely class index (including 0=non-speech)
            best_class_indices = np.argmax(scores, axis=1)

            # reconstruct annotation
            new_hypothesis = one_hot_decoding(
                best_class_indices, window, labels=labels)

        new_hypothesis.uri = hypothesis.uri
        return new_hypothesis
