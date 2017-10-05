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

import dask
import numpy as np
from .features.utils import read_audio
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature


class Stream:
    EndOfStream = 'EndOfStream'
    NoNewData = None

class More(object):
    def __init__(self, output):
        super(More, self).__init__()
        self.output = output

def stream_audio(current_file, sample_rate=None, mono=True, duration=1.):
    """Simulate audio file streaming

    Parameters
    ----------
    current_file : dict
        Dictionary given by pyannote.database.
    sample_rate: int, optional
        Target sampling rate. Defaults to using native sampling rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.
    duration : float, optional
        Buffer duration, in seconds. Defaults to 1.

    Returns
    -------
    buffer : iterable
        Yields SlidingWindowFeature instances

    Usage
    -----
    >>> for buffer in stream_audio(current_file):
    ...     do_something_with(buffer)

    Notes
    -----
    In case `current_file` contains a `channel` key, data of this (1-indexed)
    channel will be yielded.

    """

    y, sample_rate = read_audio(current_file,
                                sample_rate=sample_rate,
                                mono=mono)

    n_samples_total = len(y)
    n_samples_buffer = int(duration * sample_rate)

    for i in range(0, n_samples_total, n_samples_buffer):
        data = y[i: i + n_samples_buffer, np.newaxis]
        sw = SlidingWindow(start=i / sample_rate,
                           duration=1 / sample_rate,
                           step=1 / sample_rate)
        yield SlidingWindowFeature(data, sw)

    while True:
        yield Stream.EndOfStream


def stream_features(feature_extraction, current_file, duration=1.):
    """Simulate online feature extraction

    Parameters
    ----------
    feature_extraction : callable
        Feature extraction
    current_file : dict
        Dictionary given by pyannote.database.
    duration : float, optional
        Buffer duration, in seconds. Defaults to 1.

    Returns
    -------
    buffer : iterable
        Yields SlidingWindowFeature instances

    Usage
    -----
    >>> for buffer in stream_features(feature_extraction, current_file):
    ...     do_something_with(buffer)

    """

    features = feature_extraction(current_file)
    sliding_window = features.sliding_window
    data = features.data

    n_samples_total = len(data)
    n_samples_buffer = sliding_window.samples(duration, mode='center')

    for i in range(0, n_samples_total, n_samples_buffer):
        sw = SlidingWindow(start=sliding_window[i].start,
                           duration=sliding_window.duration,
                           step=sliding_window.step)
        yield SlidingWindowFeature(data[i: i+n_samples_buffer], sw)

    while True:
        yield Stream.EndOfStream


class Buffer(object):
    """This module concatenates (adjacent) input sequences and returns the
    result using a sliding window.

    Parameters
    ----------
    duration : float, optional
        Sliding window duration. Defaults to 3.2 seconds.
    step : float, optional
        Sliding window step. Defaults to `duration`.
    incomplete : bool, optional
        Set to True to return the current buffer on "end-of-stream"
        even if is is not complete. Defaults to False.
    """

    def __init__(self, duration=3.2, step=None, incomplete=False):
        super(Buffer, self).__init__()
        self.duration = duration
        self.step = duration if step is None else step
        self.incomplete = incomplete
        self.initialized_ = False

    def initalize(self, sequence):

        # common time base
        sw = sequence.sliding_window
        self.frames_ = SlidingWindow(start=sw.start,
                                     duration=sw.duration,
                                     step=sw.step)

        self.buffer_ = np.array(sequence.data)

        self.window_ = SlidingWindow(start=sw.start,
                                     duration=self.duration,
                                     step=self.step)
        self.current_window_ = next(self.window_)
        self.n_samples_ = self.frames_.samples(self.duration, mode='center')
        self.initialized_ = True

    def __call__(self, sequence=Stream.NoNewData):

        # if input stream has ended
        if sequence == Stream.EndOfStream:

            # if buffer has been emptied already, return "end-of-stream"
            if not self.initialized_:
                return Stream.EndOfStream

            # reset buffer
            self.initialized_ = False

            # if requested, return the current buffer on "end-of-stream"
            if self.incomplete:
                return SlidingWindowFeature(self.buffer_, self.frames_)

            return Stream.EndOfStream

        # if input stream continues
        elif sequence != Stream.NoNewData:

            # append to buffer
            if self.initialized_:

                # check that feature sequence uses the common time base
                sw = sequence.sliding_window
                assert sw.duration == self.frames_.duration
                assert sw.step == self.frames_.step

                # check that first frame is exactly the one that is expected
                expected = self.frames_[len(self.buffer_)]
                assert np.allclose(expected, sw[0])

                # append the new samples at the end of buffer
                self.buffer_ = np.concatenate([self.buffer_, sequence.data],
                                              axis=0)

            # initialize buffer
            else:
                self.initalize(sequence)

        # if not enough samples are available, there is nothing to return
        if not self.initialized_ or self.buffer_.shape[0] < self.n_samples_:
            return Stream.NoNewData

        # if enough samples are available, prepare output
        output = SlidingWindowFeature(self.buffer_[:self.n_samples_],
                                      self.frames_)

        # switch to next window
        self.current_window_ = next(self.window_)

        # update buffer by remove old samples and updating start time
        first_valid = self.frames_.crop(self.current_window_,
                                        mode='center',
                                        fixed=self.duration)[0]
        self.buffer_ = self.buffer_[first_valid:]
        self.frames_ = SlidingWindow(start=self.frames_[first_valid].start,
                                     duration=self.frames_.duration,
                                     step=self.frames_.step)

        # if enough samples are available for next window
        # wrap output into a More instance
        if self.buffer_.shape[0] >= self.n_samples_:
            output = More(output)

        return output


class Aggregate(object):

    def __init__(self, merge_func=np.nanmean):
        super(Aggregate, self).__init__()
        self.merge_func = merge_func
        self.initialized_ = False

    def initalize(self, sequence):

        # common time base
        sw = sequence.sliding_window
        self.frames_ = SlidingWindow(start=sw.start,
                                     duration=sw.duration,
                                     step=sw.step)

        data = sequence.data
        shape = (1,) + data.shape
        self.buffer_ = np.ones(shape, dtype=data.dtype)
        self.buffer_[0, :] = data

        self.initialized_ = True

        return Stream.NoNewData

    def __call__(self, sequence=Stream.NoNewData):

        # no input ==> no output
        if sequence is Stream.NoNewData:
            return Stream.NoNewData

        if sequence is Stream.EndOfStream:
            if not self.initialized_:
                return Stream.EndOfStream

            self.initialized_ = False
            data = self.merge_func(self.buffer_, axis=0)
            return SlidingWindowFeature(data, self.frames_)

        if not self.initialized_:
            return self.initalize(sequence)

        # check that feature sequence uses the common time base
        sw = sequence.sliding_window
        assert sw.duration == self.frames_.duration
        assert sw.step == self.frames_.step
        assert sw.start > self.frames_.start

        delta_start = sw.start - self.frames_.start
        ready = self.frames_.samples(delta_start, mode='center')
        data = self.merge_func(self.buffer_[:, :ready], axis=0)
        output = SlidingWindowFeature(data, self.frames_)

        self.buffer_ = self.buffer_[:, ready:]
        self.frames_ = SlidingWindow(start=sw.start,
                                     duration=sw.duration,
                                     step=sw.step)

        # remove empty (all NaN) buffers
        n_buffers = self.buffer_.shape[0]
        for i in range(n_buffers):
            if np.any(~np.isnan(self.buffer_[i])):
                break
        self.buffer_ = self.buffer_[i:]

        n_samples = self.buffer_.shape[1]
        n_new_samples = sequence.data.shape[0]
        pad_width = ((0, 1), (0, max(0, n_new_samples - n_samples)))
        for _ in sequence.data.shape[1:]:
            pad_width += ((0, 0), )
        self.buffer_ = np.pad(self.buffer_, pad_width, 'constant',
                              constant_values=np.NAN)
        self.buffer_[-1] = sequence.data

        return output


class Process(object):

    def __init__(self, process_func):

        super(Process, self).__init__()
        self.process_func = process_func

    def __call__(self, sequence=Stream.NoNewData):

        # no input ==> no output
        if sequence in [Stream.NoNewData, Stream.EndOfStream]:
            return sequence

        return self.process_func(sequence)


class Pipeline(object):

    def __init__(self, dsk):
        super(Pipeline, self).__init__()
        # TODO -- check that at least one input depends on 'input_buffer'
        self.dsk = dsk
        self.so_far_ = Segment(0, 0)

    @property
    def so_far(self):
        return self.so_far_

    def __call__(self, input_buffer):

        keys = sorted(['input_buffer'] + list(self.dsk.keys()))
        more = False

        while True:

            if more:
                self.dsk['input_buffer'] = Stream.NoNewData
                more = False
            else:
                buf = next(input_buffer)
                if buf not in [Stream.EndOfStream, Stream.NoNewData]:
                    self.so_far_ |= buf.getExtent()
                self.dsk['input_buffer'] = buf

            outputs = {key: output
                       for key, output in zip(keys, dask.get(self.dsk, keys))}

            for key in keys:
                if isinstance(outputs[key], More):
                    more = True
                    outputs[key] = outputs[key].output

            if all(o == Stream.EndOfStream for o in outputs.values()):
                return

            yield outputs

