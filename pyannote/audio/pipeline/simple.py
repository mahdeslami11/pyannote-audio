#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

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


from typing import Text, Union
from pathlib import Path

import numpy as np
from pyannote.pipeline import Pipeline
from pyannote.pipeline.parameter import Uniform
from pyannote.pipeline.parameter import LogUniform

from pyannote.audio.features.wrapper import Wrapper
from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.database import get_annotated
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation
from pyannote.core import Timeline
from pyannote.core import Segment
from pyannote.audio.utils.signal import Binarize
from pyannote.audio.utils.signal import Peak
from pyannote.core.utils.hierarchy import pool
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist


class SimpleDiarization(Pipeline):
    """Simple diarization pipeline

    Parameters
    ----------
    sad : str or Path, optional
        Pretrained speech activity detection model. Defaults to "sad".
    scd : str or Path, optional
        Pretrained speaker change detection model. Defaults to "scd".
    emb : str or Path, optional
        Pretrained speaker embedding model. Defaults to "emb".

    Hyper-parameters
    ----------------
    sad_onset : float
        Threshold applied on speech activity detection scores.
    scd_threshold : float
        Threshold applied on speaker change detection scores local maxima.
    seg_min_duration : float
        Minimum duration of speech turns.
    min_duration_off : float
        Minimum duration of gaps between speech turns.
    emb_duration : float
        Do not cluster segments shorter than `emb_duration` duration. Short
        segments will eventually be assigned to the most similar cluster.
    emb_threshold : float
        Distance threshold used as stopping criterion for hierarchical
        agglomeratice clustering.
    """

    def __init__(
        self,
        sad: Union[Text, Path] = "sad",
        scd: Union[Text, Path] = "scd",
        emb: Union[Text, Path] = "emb",
    ):

        super().__init__()

        self.sad = Wrapper(sad)
        self.sad_speech_index_ = self.sad.classes.index("speech")

        self.scd = Wrapper(scd)
        self.scd_change_index_ = self.scd.classes.index("change")
        self.emb = Wrapper(emb)

        self.sad_threshold_on = Uniform(0.0, 1.0)
        self.sad_threshold_off = Uniform(0.0, 1.0)
        self.scd_threshold = LogUniform(1e-8, 1.0)
        self.seg_min_duration = Uniform(0.0, 0.5)
        self.gap_min_duration = Uniform(0.0, 0.5)
        self.emb_duration = Uniform(0.5, 4.0)
        self.emb_threshold = Uniform(0.0, 2.0)

    def initialize(self):

        self.sad_binarize_ = Binarize(
            onset=self.sad_onset,
            offset=self.sad_onset,
            min_duration_on=self.seg_min_duration,
            min_duration_off=self.gap_min_duration,
        )

        self.scd_peak_ = Peak(
            alpha=self.scd_threshold, min_duration=self.seg_min_duration
        )

    def get_embedding(self, current_file, segment):

        try:
            embeddings = self.emb.crop(current_file, segment)

        except RuntimeError as e:

            # A RuntimeError exception is raised by the pretrained "emb" model
            # when the input waveform is too short (i.e. < 157ms).

            # We catch it and extend the segment on both sides until it reaches
            # this target duration.

            # This ugly hack will probably bite us later...

            MIN_DURATION = 0.157
            if segment.middle - 0.5 * MIN_DURATION < 0.0:
                extended_segment = Segment(0.0, MIN_DURATION)
            elif segment.middle + 0.5 * MIN_DURATION > current_file["duration"]:
                extended_segment = Segment(
                    current_file["duration"] - MIN_DURATION, current_file["duration"]
                )
            else:
                extended_segment = Segment(
                    segment.middle - 0.5 * MIN_DURATION,
                    segment.middle + 0.5 * MIN_DURATION,
                )
            embeddings = self.emb.crop(current_file, extended_segment)

        return np.mean(embeddings, axis=0)

    def __call__(self, current_file: ProtocolFile) -> Annotation:

        uri = current_file.get("uri", "pyannote")

        # apply pretrained SAD model and turn log-probabilities into probabilities
        if "sad_scores" in current_file:
            sad_scores = current_file["sad_scores"]
        else:
            sad_scores = self.sad(current_file)
            if np.nanmean(sad_scores) < 0:
                sad_scores = np.exp(sad_scores)
            current_file["sad_scores"] = sad_scores

        # apply SAD binarization
        sad = self.sad_binarize_.apply(sad_scores, dimension=self.sad_speech_index_)

        # apply pretrained SCD model and turn log-probabilites into probabilites
        if "scd_scores" in current_file:
            scd_scores = current_file["scd_scores"]
        else:
            scd_scores = self.scd(current_file)
            if np.nanmean(scd_scores) < 0:
                scd_scores = np.exp(scd_scores)
            current_file["scd_scores"] = scd_scores

        # apply SCD peak detection
        scd = self.scd_peak_.apply(scd_scores, dimension=self.scd_change_index_)

        # split (potentially multi-speaker) speech regions at change points
        seg = scd.crop(sad, mode="intersection")

        # remove resulting tiny segments, inconsistent with SCD duration constraint
        seg = [s for s in seg if s.duration > min(0.1, self.seg_min_duration)]

        # separate long segments from short ones
        seg_long = [s for s in seg if s.duration >= self.emb_duration]

        if len(seg_long) == 0:
            # there are only short segments. put each of them in its own cluster
            return Timeline(segments=seg, uri=uri).to_annotation(generator="string")

        elif len(seg_long) == 1:
            # there is exactly one long segment. put everything in one cluster
            return Timeline(segments=seg, uri=uri).to_annotation(
                generator=iter(lambda: "A", None)
            )

        else:

            # extract embeddings of long segments
            emb_long = np.vstack(
                [self.get_embedding(current_file, s) for s in seg_long]
            )

            # apply clustering
            Z = pool(
                emb_long,
                metric="cosine",
                # pooling_func="average",
                # cannot_link=None,
                # must_link=None,
            )
            cluster_long = fcluster(Z, self.emb_threshold, criterion="distance")

            seg_shrt = [s for s in seg if s.duration < self.emb_duration]

            if len(seg_shrt) == 0:
                # there are only long segments.
                return Timeline(segments=seg, uri=uri).to_annotation(
                    generator=iter(cluster_long)
                )

            # extract embeddings of short segments
            emb_shrt = np.vstack(
                [self.get_embedding(current_file, s) for s in seg_shrt]
            )

            # assign each short segment to the cluster containing the closest long segment
            cluster_shrt = cluster_long[
                np.argmin(cdist(emb_long, emb_shrt, metric="cosine"), axis=0)
            ]

            seg_shrt = Timeline(segments=seg_shrt, uri=uri)
            seg_long = Timeline(segments=seg_long, uri=uri)

            return seg_long.to_annotation(generator=iter(cluster_long)).update(
                seg_shrt.to_annotation(generator=iter(cluster_shrt))
            )

    def loss(self, current_file: ProtocolFile, hypothesis: Annotation) -> float:
        """Compute diarization error rate

        Parameters
        ----------
        current_file : ProtocolFile
            Protocol file
        hypothesis : Annotation
            Hypothesized diarization output

        Returns
        -------
        der : float
            Diarization error rate.
        """

        return DiarizationErrorRate()(
            current_file["annotation"], hypothesis, uem=get_annotated(current_file)
        )

    def get_metric(self) -> DiarizationErrorRate:
        return DiarizationErrorRate(collar=0.0, skip_overlap=False)
