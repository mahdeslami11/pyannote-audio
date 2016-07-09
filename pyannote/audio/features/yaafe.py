#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2014-2016 CNRS

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

from __future__ import unicode_literals

import numpy as np
import yaafelib
import scipy.io.wavfile
from scipy.stats import zscore

from pyannote.core.segment import SlidingWindow
from pyannote.core.feature import SlidingWindowFeature
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.core import PYANNOTE_SEGMENT

from .utils import get_wav_duration


class YaafeFrame(SlidingWindow):
    """Yaafe frames

    Parameters
    ----------
    blockSize : int, optional
        Window size (in number of samples). Default is 512.
    stepSize : int, optional
        Step size (in number of samples). Default is 256.
    sampleRate : int, optional
        Sample rate (number of samples per second). Default is 16000.

    References
    ----------
    http://yaafe.sourceforge.net/manual/quickstart.html

    """
    def __init__(self, blockSize=512, stepSize=256, sampleRate=16000):

        duration = 1. * blockSize / sampleRate
        step = 1. * stepSize / sampleRate
        start = -0.5 * duration

        super(YaafeFrame, self).__init__(
            duration=duration, step=step, start=start
        )


class YaafeFeatureExtractor(object):
    """

    Parameters
    ----------
    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    block_size : int, optional
        Defaults to 512.
    step_size : int, optional
        Defaults to 256.

    """

    def __init__(
        self, sample_rate=16000, block_size=512, step_size=256
    ):

        super(YaafeFeatureExtractor, self).__init__()

        self.sample_rate = sample_rate
        self.block_size = block_size
        self.step_size = step_size

    def extract(self, wav):
        return self.__call__(wav)

    def dimension(self):
        raise NotImplementedError('')

    def get_frame(self):
        return YaafeFrame(blockSize=self.block_size,
                          stepSize=self.step_size,
                          sampleRate=self.sample_rate)

    def __call__(self, wav):
        """Extract features

        Parameters
        ----------
        wav : string
            Path to wav file.

        Returns
        -------
        features : SlidingWindowFeature

        """

        definition = self.definition()

        # --- prepare the feature plan
        feature_plan = yaafelib.FeaturePlan(sample_rate=self.sample_rate)
        for name, recipe in definition:
            assert feature_plan.addFeature(
                "{name}: {recipe}".format(name=name, recipe=recipe))

        # --- prepare the Yaafe engine
        data_flow = feature_plan.getDataFlow()

        engine = yaafelib.Engine()
        engine.load(data_flow)

        sample_rate, raw_audio = scipy.io.wavfile.read(wav)
        assert sample_rate == self.sample_rate, "sample rate mismatch"

        audio = np.array(raw_audio, dtype=np.float64, order='C').reshape(1, -1)

        features = engine.processAudio(audio)
        data = np.hstack([features[name] for name, _ in definition])

        sliding_window = YaafeFrame(
            blockSize=self.block_size, stepSize=self.step_size,
            sampleRate=self.sample_rate)

        return SlidingWindowFeature(data, sliding_window)


class YaafeCompound(YaafeFeatureExtractor):

    def __init__(
        self, extractors,
        sample_rate=16000, block_size=512, step_size=256
    ):

        assert all(e.sample_rate == sample_rate for e in extractors)
        assert all(e.block_size == block_size for e in extractors)
        assert all(e.step_size == step_size for e in extractors)

        super(YaafeCompound, self).__init__(
            sample_rate=sample_rate,
            block_size=block_size,
            step_size=step_size)

        self.extractors = extractors

    def dimension(self):
        return sum(extractor.dimension() for extractor in self.extractors)

    def definition(self):
        return [(name, recipe)
                for e in self.extractors for name, recipe in e.definition()]

    def __hash__(self):
        return hash(tuple(self.definition()))


class YaafeZCR(YaafeFeatureExtractor):

    def dimension(self):
        return 1

    def definition(self):

        d = [(
            "zcr",
            "ZCR blockSize=%d stepSize=%d" % (self.block_size, self.step_size)
        )]

        return d


class YaafeMFCC(YaafeFeatureExtractor):
    """
        | e    |  energy
        | c1   |
        | c2   |  coefficients
        | c3   |
        | ...  |
        | Δe   |  energy first derivative
        | Δc1  |
    x = | Δc2  |  coefficients first derivatives
        | Δc3  |
        | ...  |
        | ΔΔe  |  energy second derivative
        | ΔΔc1 |
        | ΔΔc2 |  coefficients second derivatives
        | ΔΔc3 |
        | ...  |


    Parameters
    ----------

    sample_rate : int, optional
        Defaults to 16000 (i.e. 16kHz)
    block_size : int, optional
        Defaults to 512.
    step_size : int, optional
        Defaults to 256.

    e : bool, optional
        Energy. Defaults to True.
    coefs : int, optional
        Number of coefficients. Defaults to 11.
    De : bool, optional
        Keep energy first derivative. Defaults to False.
    D : bool, optional
        Add first order derivatives. Defaults to False.
    DDe : bool, optional
        Keep energy second derivative. Defaults to False.
    DD : bool, optional
        Add second order derivatives. Defaults to False.

    Notes
    -----
    Default Yaafe values:
        * fftWindow = Hanning
        * melMaxFreq = 6854.0
        * melMinFreq = 130.0
        * melNbFilters = 40

    """

    def __init__(
        self, sample_rate=16000, block_size=512, step_size=256,
        e=True, coefs=11, De=False, DDe=False, D=False, DD=False,
    ):

        super(YaafeMFCC, self).__init__(
            sample_rate=sample_rate,
            block_size=block_size,
            step_size=step_size
        )

        self.e = e
        self.coefs = coefs
        self.De = De
        self.DDe = DDe
        self.D = D
        self.DD = DD

    def dimension(self):

        n_features = 0
        n_features += self.e
        n_features += self.De
        n_features += self.DDe
        n_features += self.coefs
        n_features += self.coefs * self.D
        n_features += self.coefs * self.DD

        return n_features

    def definition(self):

        d = []

        # --- coefficients
        # 0 if energy is kept
        # 1 if energy is removed
        d.append((
            "mfcc",
            "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
            "blockSize=%d stepSize=%d" % (
                0 if self.e else 1,
                self.coefs + self.e * 1,
                self.block_size, self.step_size
            )))

        # --- 1st order derivatives
        if self.De or self.D:
            d.append((
                "mfcc_d",
                "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                "blockSize=%d stepSize=%d > Derivate DOrder=1" % (
                    0 if self.De else 1,
                    self.D * self.coefs + self.De * 1,
                    self.block_size, self.step_size
                )))

        # --- 2nd order derivatives
        if self.DDe or self.DD:
            d.append((
                "mfcc_dd",
                "MFCC CepsIgnoreFirstCoeff=%d CepsNbCoeffs=%d "
                "blockSize=%d stepSize=%d > Derivate DOrder=2" % (
                    0 if self.DDe else 1,
                    self.DD * self.coefs + self.DDe * 1,
                    self.block_size, self.step_size
                )))

        return d


class YaafeFileBasedBatchGenerator(FileBasedBatchGenerator):

    def __init__(self, feature_extractor, fragment_generator, batch_size=32, normalize=False):
        super(YaafeFileBasedBatchGenerator, self).__init__(fragment_generator, batch_size=batch_size)
        self.feature_extractor = feature_extractor
        self.normalize = normalize
        self.fe_frame = self.feature_extractor.get_frame()
        self.fe_n = self.fe_frame.durationToSamples(fragment_generator.duration)
        self.X_ = {}

    def get_shape(self):
        return (self.fe_n, self.feature_extractor.dimension())

    # defaults to features pre-computing
    def preprocess(self, current_file, identifier=None):
        wav, _, _ = current_file
        if not identifier in self.X_:

            features = self.feature_extractor(wav)

            expected = get_wav_duration(wav)
            actual = features.getExtent().end
            if not np.isclose(expected, actual, atol=1., rtol=0.):
                msg = 'Feature duration should {expected:.3f}, is {actual:.3f}'
                raise ValueError(msg.format(expected=expected, actual=actual))

            self.X_[identifier] = features

        return current_file

    def process(self, fragment, signature=None, identifier=None):
        if signature['type'] == PYANNOTE_SEGMENT:
            # FIXME use feature.crop(mode='center', fixed=duration) instead
            i0, _ = self.fe_frame.segmentToRange(fragment)
            x = self.X_[identifier][i0:i0+self.fe_n]
            if self.normalize:
                x = zscore(x, axis=-1)

            expected = self.get_shape()
            actual = x.shape

            if actual != expected:
                msg = 'Shape should be {expected}, is {actual}'
                raise ValueError(msg.format(expected=self.get_shape(),
                                            actual=x.shape))
            return x
        else:
            return fragment
