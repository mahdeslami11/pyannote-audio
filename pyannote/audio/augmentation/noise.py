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


import numpy as np
from pyannote.core import Segment
from pyannote.audio.features.utils import RawAudio
from pyannote.audio.features.utils import get_audio_duration
from pyannote.generators.fragment import random_subsegment
from pyannote.database import get_protocol
from pyannote.database import FileFinder
from .base import Augmentation
from glob import glob


class AddNoise(Augmentation):
    """Add noise

    Parameters
    ----------
    collection : str or list of str
        `pyannote.database` collection(s) used for adding noise. Defaults to
        'MUSAN.Collection.BackgroundNoise' available in `pyannote.db.musan`
        package.
    db_yml : str, optional
        Path to `pyannote.database` configuration file.
    snr_min, snr_max : int, optional
        Defines Signal-to-Noise Ratio (SNR) range in dB. Defaults to [5, 20].
    """

    def __init__(self, collection=None, db_yml=None, snr_min=5, snr_max=20):
        super().__init__()

        if collection is None:
            collection = 'MUSAN.Collection.BackgroundNoise'
        if not isinstance(collection, (list, tuple)):
            collection = [collection]
        self.collection = collection
        self.db_yml = db_yml

        self.snr_min = snr_min
        self.snr_max = snr_max

        # load noise database
        self.filenames_, self.durations_ = [], []
        preprocessors = {'audio': FileFinder(config_yml=db_yml)}
        for c in self.collection:
            protocol = get_protocol(c, preprocessors=preprocessors)
            for current_file in protocol.files():
                self.filenames_.append(current_file['audio'])
                self.durations_.append(get_audio_duration(current_file))
        self.durations_ = np.array(self.durations_)

    def normalize(self, waveform):
        return waveform / np.sqrt(np.mean(waveform ** 2))

    def __call__(self, original, sample_rate):

        raw_audio = RawAudio(sample_rate=sample_rate, mono=True)

        original_duration = len(original) / sample_rate

        # accumulate enough noise to cover duration of original waveform
        noises = []
        left = original_duration
        while left > 0:

            # select noise file at random
            # chosen = np.random.choice(len(self.filenames_),
            #                           p=self.probabilities_)
            chosen = np.random.choice(len(self.filenames_))
            filename = self.filenames_[chosen]
            duration = self.durations_[chosen]

            # if noise file is longer than what is needed, crop it
            if duration > left:
                segment = next(random_subsegment(Segment(0, duration), left))
                noise = raw_audio.crop({'audio': filename}, segment)
                left -= left

            # otherwise, take the whole file
            else:
                noise = raw_audio({'audio': filename}).data
                left -= duration

            noise = self.normalize(noise)
            noises.append(noise)

        # concatenate
        # FIXME: use fade-in between concatenated noises
        noise = np.vstack(noises)

        # select SNR at random
        snr = (self.snr_max - self.snr_min) * np.random.random_sample() + self.snr_min
        alpha = np.exp(-np.log(10) * snr / 20)

        # add noise
        return self.normalize(original) + alpha * noise
