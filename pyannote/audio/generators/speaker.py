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


import numpy as np
from pyannote.audio.features import Precomputed
from pyannote.generators.fragment import random_segment
from pyannote.generators.fragment import random_subsegment
from pyannote.generators.batch import batchify
from pyannote.database import get_label_identifier


class SpeechTurnGenerator(object):
    """Generate batch of speech turns with associated speaker labels

    Parameters
    ----------
    precomputed : pyannote.audio.features.utils.Precomputed
    per_label : int, optional
        Number of speech turns per speaker in each batch
    per_fold : int, optional
    duration : float, optional
    min_duration : float, optional
    max_duration : float, optional
    fast : bool, optional
        Defaults to True.
    """

    def __init__(self, precomputed, per_label=3, per_fold=None,
                 duration=None, min_duration=None, max_duration=None,
                 fast=True):

        super(SpeechTurnGenerator, self).__init__()

        self.precomputed = precomputed
        self.per_label = per_label
        self.per_fold = per_fold
        self.duration = duration
        self.fast = fast

        if self.duration is None:
            self.min_duration = min_duration
            self.max_duration = max_duration
        else:
            self.min_duration = self.duration
            self.max_duration = self.duration

        self.min_duration_ = 0. if self.min_duration is None \
                                else self.min_duration

    def initialize(self, protocol, subset='train'):

        self.data_ = {}
        databases = set()

        # loop once on all files
        for current_file in getattr(protocol, subset)():

            # keep track of database
            database = current_file['database']
            databases.add(database)

            # get annotation for current file
            annotation = current_file['annotation']

            # pre-load features.
            if not self.precomputed.use_memmap:
                msg = ('Loading all precomputed features in memory. '
                       'Set "use_memmap" to True if you run out of memory.')
                warnings.warn(msg)

            if self.fast:
                features = self.precomputed(current_file)
            else:
                features = None

            # loop on each label in current file
            for label in annotation.labels():

                # get all segments with current label
                timeline = annotation.label_timeline(label)

                # remove segments shorter than min_duration (when provided)
                segments = [s for s in timeline
                              if s.duration > self.min_duration_]

                # corner case where no segment is long enough
                # and we removed them all...
                if not segments:
                    continue

                # total duration of label in current_file (after removal of
                # short segments).
                duration = sum(s.duration for s in segments)

                # this generator will randomly (and infinitely) select one
                # segment among remaining segments, with probability
                # proportional to its duration. in other words, longer segments
                # are more likely to be selected.
                segment_generator = random_segment(segments, weighted=True)

                # store all these in data_ dictionary
                datum = (segment_generator, duration, current_file, features)
                l = get_label_identifier(label, current_file)
                self.data_.setdefault(l, []).append(datum)

        self.labels_ = {label: i for i, label in enumerate(self.data_)}
        self.databases_ = {database: i for i, database in enumerate(databases)}

    def generator(self):

        labels = list(self.data_)

        while True:

            # shuffle labels
            np.random.shuffle(labels)

            # loop on each label
            for label in labels:

                # load data for this label
                segment_generators, durations, files, features = \
                    zip(*self.data_[label])

                # choose 'per_label' files at random with probability
                # proportional to the total duration of 'label' in those files
                probabilities = durations / np.sum(durations)
                chosen = np.random.choice(len(files), size=self.per_label,
                                          p=probabilities)

                # loop on (randomly) chosen files
                for i in chosen:

                    if features[i] is None:
                        features_ = self.precomputed(files[i])
                    else:
                        features_ = features[i]

                    # choose one segment at random with
                    # probability proportional to duration
                    segment = next(segment_generators[i])

                    if self.duration is None:

                        if self.min_duration is None:

                            # case: no duration | no min | no max
                            # keep segment as it is
                            if self.max_duration is None:
                                sub_segment = segment

                            # case: no duration | no min | max
                            else:

                                # if segment is too long, choose sub-segment
                                # at random at exactly max_duration
                                if segment.duration > self.max_duration:
                                    sub_segment = next(random_subsegment(
                                        segment, self.max_duration))

                                # otherwise, keep segment as it is
                                else:
                                    sub_segment = segment

                        else:
                            # case: no duration | min | no max
                            # keep segment as it is (too short segments have
                            # already been filtered out)
                            if self.max_duration is None:
                                sub_segment = segment

                            # case: no duration | min | max
                            else:
                                # choose sub-segment at random between
                                # min_duration and max_duration
                                sub_segment = next(random_subsegment(
                                    segment, self.max_duration,
                                    min_duration=self.min_duration))

                        X = features_.crop(sub_segment, mode='center')

                    else:

                        # choose sub-segment at random at exactly duration
                        sub_segment = next(random_subsegment(
                            segment, self.duration))

                        X = features_.crop(
                            sub_segment, mode='center', fixed=self.duration)

                    database = files[i]['database']
                    extra = {'label': label,
                             'database': database}

                    yield {'X': X,
                           'y': self.labels_[label],
                           'y_database': self.databases_[database],
                           'extra': extra}

    def __call__(self, protocol, subset='train'):

        self.initialize(protocol, subset=subset)
        generator = self.generator()

        signature_extra = {'label': {'type': 'str'},
                           'database': {'type': 'str'}}

        signature = {'X': {'type': 'ndarray'},
                     'y': {'type': 'scalar'},
                     'y_database': {'type': 'scalar'},
                     'extra': signature_extra}

        if self.per_fold is not None:
            batch_size = self.per_label * self.per_fold

        else:
            batch_size = self.per_label * len(self.data_)

        for batch in batchify(generator, signature, batch_size=batch_size):
            yield batch


    @property
    def n_sequences_per_batch(self):
        if self.per_fold is None:
            return self.per_label * len(self.data_)
        return self.per_label * self.per_fold

    @property
    def n_labels(self):
        return len(self.data_)
