# MIT License
#
# Copyright (c) 2020 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import math
import random

import numpy as np

from pyannote.audio.core.io import Audio
from pyannote.audio.core.task import Problem, Scale, Task, TaskSpecification
from pyannote.audio.utils.random import create_rng_for_worker
from pyannote.core import Segment
from pyannote.database import Protocol


class OverlappedSpeechDetection(Task):
    """Overlapped speech detection

    Overlapped speech detection is the task of detecting regions where at least
    two speakers are speaking at the same time.

    Here, it is addressed with the same approach as voice activity detection,
    except {"non-speech", "speech"} classes are replaced by {"non-overlap", "overlap"}
    where a frame is marked as "overlap" if two speakers or more are active.

    Note that data augmentation is used to increase the proporition of "overlap".
    This is achieved by generating chunks made out of the (weighted) sum of two
    random chunks.

    Parameters
    ----------
    protocol : Protocol
        pyannote.database protocol
    duration : float, optional
        Chunks duration. Defaults to 2s.
    augmentation_probability : float, optional
        Probability of artificial overlapping chunks. A probability of 0.6 means that,
        on average, 40% of training chunks are "real" chunks, while 60% are artifical
        chunks made out of the (weighted) sum of two chunks. Defaults to 0.5.
    snr_min, snr_max : float, optional
        Minimum and maximum signal-to-noise ratio between summed chunks, in dB.
        Defaults to 0.0 and 10.
    domain : str, optional
        Indicate that data augmentation will only overlap chunks from the same domain
        (i.e. share the same file[domain] value). Default behavior is to not contrain
        data augmentation with regards to domain.
    collar : int, optional.
        Mark frames less than `collar` frames away from actual change point as positive.
        Defaults to 1.
    batch_size : int, optional
        Number of training samples per batch.
    num_workers : int, optional
        Number of workers used for generating training samples.
    """

    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        augmentation_probability: float = 0.5,
        snr_min: float = 0.0,
        snr_max: float = 10.0,
        domain: str = None,
        batch_size: int = None,
        num_workers: int = 1,
    ):

        super().__init__(
            protocol, duration=duration, batch_size=batch_size, num_workers=num_workers
        )

        self.specifications = TaskSpecification(
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            scale=Scale.FRAME,
            classes=["non_overlap", "overlap"],
        )

        self.augmentation_probability = augmentation_probability
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.domain = domain

    def setup(self, stage=None):
        if stage == "fit":

            # loop over the training set, remove annotated regions shorter than
            # chunk duration, and keep track of the reference annotations.

            # also build the list of domains when needed

            self.train = []
            for f in self.protocol.train():
                segments = [
                    segment
                    for segment in f["annotated"]
                    if segment.duration > self.duration
                ]
                duration = sum(segment.duration for segment in segments)
                self.train.append(
                    {
                        "annotated": segments,
                        "annotation": f["annotation"],
                        "duration": duration,
                        "audio": f["audio"],
                    }
                )
                if self.domain is not None:
                    self.train[-1]["domain"] = f[self.domain]

            if self.domain:
                self.domains = list(set(f["domain"] for f in self.train))

    def train__iter__helper(self, rng: random.Random, domain: str = None):

        train = self.train

        if domain is not None:
            train = [f for f in train if f["domain"] == domain]

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = rng.choices(
                train,
                weights=[f["duration"] for f in train],
                k=1,
            )

            # select one annotated region at random (with probability proportional to its duration)
            segment, *_ = rng.choices(
                file["annotated"],
                weights=[s.duration for s in file["annotated"]],
                k=1,
            )

            # select one chunk at random (with uniform distribution)
            start_time = rng.uniform(segment.start, segment.end - self.duration)
            chunk = Segment(start_time, start_time + self.duration)

            yield self.prepare_chunk(
                file,
                chunk,
                duration=self.duration,
                return_y=True,
            )

    def prepare_y(self, one_hot_y: np.ndarray, collar: int = None):
        """Get overlapped speech detection targets

        Parameters
        ----------
        one_hot_y : (num_frames, num_speakers) np.ndarray
            One-hot-encoding of current chunk speaker activity:
                * one_hot_y[t, k] = 1 if kth speaker is active at tth frame
                * one_hot_y[t, k] = 0 otherwise.

        Returns
        -------
        y : (num_frames, ) np.ndarray
            y[t] = 1 if there is two or more active speakers at tth frame, 0 otherwise.
        """

        return np.int64(np.sum(one_hot_y, axis=1, keepdims=True) > 1)

    def train__iter__(self):
        """Iterate over training samples

        Yields
        ------
        X: (time, channel)
            Audio chunks.
        y: (frame, )
            Frame-level targets. Note that frame < time.
            `frame` is infered automagically from the
            example model output.
        """

        # create worker-specific random number generator
        rng = create_rng_for_worker()

        if self.domain is None:
            chunks = self.train__iter__helper(rng)
        else:
            chunks_by_domain = {
                domain: self.train__iter__helper(rng, domain=domain)
                for domain in self.domains
            }

        while True:

            if self.domain is not None:
                #  draw domain at random
                domain = rng.choice(self.domains)
                chunks = chunks_by_domain[domain]

            # generate random chunk
            X, one_hot_y, labels = next(chunks)

            if rng.random() > self.augmentation_probability:
                # yield it as it is
                yield {"X": X, "y": self.prepare_y(one_hot_y)}
                continue

            #  generate second random chunk
            other_X, other_one_hot_y, other_labels = next(chunks)

            #  sum both chunks with random SNR
            random_snr = (self.snr_max - self.snr_min) * rng.random() + self.snr_min
            alpha = np.exp(-np.log(10) * random_snr / 20)
            X = Audio.normalize(X) + alpha * Audio.normalize(other_X)

            # combine speaker-to-index mapping
            y_mapping = {label: i for i, label in enumerate(labels)}
            num_labels = len(y_mapping)
            for label in other_labels:
                if label not in y_mapping:
                    y_mapping[label] = num_labels
                    num_labels += 1

            #  combine one-hot-encoded speaker activities
            combined_y = np.zeros_like(one_hot_y, shape=(len(one_hot_y), num_labels))
            for i, label in enumerate(labels):
                combined_y[:, y_mapping[label]] += one_hot_y[:, i]
            for i, label in enumerate(other_labels):
                combined_y[:, y_mapping[label]] += other_one_hot_y[:, i]

            # handle corner case when the same label is active at the same time in both chunks
            combined_y = np.minimum(combined_y, 1, out=combined_y)

            yield {"X": X, "y": self.prepare_y(combined_y)}

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["duration"] for file in self.train)
        return math.ceil(duration / self.duration)
