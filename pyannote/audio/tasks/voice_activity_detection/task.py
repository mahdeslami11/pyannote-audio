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

from pyannote.audio.core.task import TaskSpecification, Problem, Scale, Task
from pyannote.database import Protocol

import numpy as np
import math
import random
from pyannote.core import Segment, Timeline, SlidingWindow
from pyannote.core.utils.numpy import one_hot_encoding


class VoiceActivityDetection(Task):
    def __init__(
        self,
        protocol: Protocol,
        duration: float = 2.0,
        batch_size: int = None,
        num_workers: int = 1,
    ):

        super().__init__(
            protocol, duration=duration, batch_size=batch_size, num_workers=num_workers
        )

        # for voice activity detection, task specification
        # does not depend on the data: we can define it in
        # __init__
        self.specifications = TaskSpecification(
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            scale=Scale.FRAME,
            classes=["non_speech", "speech"],
        )

    def setup(self, stage=None):
        if stage == "fit":
            # this is where we load the training set metadata
            # to be used later by the train_dataloader.

            # here, we simply loop over the training set, remove
            # annotated regions shorter than chunk duration, and
            # keep track of the reference annotations.
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

        random.seed()

        while True:

            # select one file at random (with probability proportional to its annotated duration)
            file, *_ = random.choices(
                self.train, weights=[f["duration"] for f in self.train], k=1,
            )

            # select one annotated region at random (with probability proportional to its duration)
            segment, *_ = random.choices(
                file["annotated"], weights=[s.duration for s in file["annotated"]], k=1,
            )

            # select one chunk at random (with uniform distribution)
            start_time = random.uniform(segment.start, segment.end - self.duration)
            chunk = Segment(start_time, start_time + self.duration)

            # extract features
            X, _ = self.audio.crop(file, chunk, mode="center", fixed=self.duration)

            # note how, contrary to what is currently done in pyannote.audio,
            # y is not precomputed for the whole file at initialization time.
            # here, we stick with pyannote.core.Annotation as long as possible
            # and "one hot" encode the data only when generating training samples.
            # this should allow to train on much larger datasets.

            # TODO | this one_hot_encoding thing needs to be rewritten into pyannote.audio
            # TODO | to make sure we always return the same number of frames for the same
            # TODO | input duration. we should also support variable-length chunks.
            frames = SlidingWindow(
                start=chunk.start,
                duration=self.frame_duration,
                step=self.frame_duration,
            )
            y = one_hot_encoding(
                file["annotation"].crop(chunk), Timeline([chunk]), frames, mode="center"
            ).data

            # this is the only part of this method that is specific to VAD
            # the rest should also work for any task with Scale.FRAME
            y = np.int64(np.sum(y, axis=1) > 0)

            yield {"X": X, "y": y}

    def train__len__(self):
        # Number of training samples in one epoch
        duration = sum(file["duration"] for file in self.train)
        return math.ceil(duration / self.duration)
