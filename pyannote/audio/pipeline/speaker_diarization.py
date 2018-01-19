#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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
from pyannote.audio.keras_utils import load_model
from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.signal import Binarize, Peak
from pyannote.audio.embedding.extraction import SequenceEmbedding
from pyannote.audio.embedding.clustering import Clustering
from pyannote.core import Annotation
from pyannote.audio.embedding.utils import l2_normalize, cdist
from pyannote.database import get_annotated


class SpeakerDiarization(object):
    """Speaker diarization pipeline

    1. Speech activity detection
    2. Speaker change detection
    3. Speech turn clustering
    TODO. 4. Resegmentation

    Parameters
    ----------
    sad : pyannote.audio.features.Precomputed
    scd : pyannote.audio.features.Precomputed
    emb : pyannote.audio.features.Precomputed
    sad__onset, sad__offset : float, optional
        See pyannote.audio.signal.Binarize. Default to 0.7 and 0.7.
    sad__dimension : int, optional
        See pyannote.audio.signal.Binarize.apply(). Defaults to 1.
    scd__alpha, scd__min_duration : float, optional
        See pyannote.audio.signal.Peak. Default to 0.5 and 1.
    scd__dimension : int, optional
        See pyannote.audio.signal.Peak.apply(). Defaults to 1.
    cls__min_cluster_size, cls__min_samples, cls__metric : optional
        See pyannote.audio.embedding.clustering.Clustering
    long_first : boolean, optional
        Start by clustering longer speech turns, then associate each smaller
        speech turns to the closest cluster. Defaults to cluster all speech
        turns at once.
    """

    def __init__(self, sad, scd, emb,
                 sad__onset=0.7, sad__offset=0.7, sad__dimension=1,
                 scd__alpha=0.5, scd__min_duration=1., scd__dimension=1,
                 cls__min_cluster_size=5, cls__min_samples=None,
                 cls__metric='cosine', long_first=False):

        super(SpeakerDiarization, self).__init__()

        # speech activity detection hyper-parameters
        self.sad = sad
        self.sad__onset = sad__onset
        self.sad__offset = sad__offset
        self.sad__dimension = sad__dimension

        # speaker change detection hyper-parameters
        self.scd = scd
        self.scd__alpha = scd__alpha
        self.scd__min_duration = scd__min_duration
        self.scd__dimension = scd__dimension

        # embedding hyper-parameters
        self.emb = emb

        # clustering hyper-parameters
        self.cls__min_cluster_size = cls__min_cluster_size
        self.cls__min_samples = cls__min_samples
        self.cls__metric = cls__metric
        self.long_first = long_first

        # initialize speech activity detection module
        self.sad_binarize_ = Binarize(onset=self.sad__onset,
                                      offset=self.sad__offset)

        # initialize speaker change detection module
        self.scd_peak_ = Peak(alpha=self.scd__alpha,
                              min_duration=self.scd__min_duration,
                              percentile=False)

        # initialize clustering module
        self.cls_ = Clustering(metric=self.cls__metric,
                               min_cluster_size=self.cls__min_cluster_size,
                               min_samples=self.cls__min_samples)

    def __call__(self, current_file, annotated=False):
        """Process current file

        Parameters
        ----------
        current_file : dict
        annotated : boolean, optional
            Only process annotated region.

        Return
        ------
        result : Annotation
            Speaker diarization result.
        """

        # speech activity detection
        soft_sad = self.sad(current_file)
        hard_sad = self.sad_binarize_.apply(
            soft_sad, dimension=self.sad__dimension)

        # speaker change detection
        soft_scd = self.scd(current_file)
        hard_scd = self.scd_peak_.apply(
            soft_scd, dimension=self.scd__dimension)

        # speech turns
        speech_turns = hard_scd.crop(hard_sad)

        if annotated:
            speech_turns = speech_turns.crop(
                get_annotated(current_file))

        hypothesis = Annotation(uri=current_file['uri'])
        if not speech_turns:
            return hypothesis

        # speech turns embedding
        emb = self.emb(current_file)

        # long speech turns = those with at least one embedding fully included
        # shrt speech turns = all the other speech turns
        long_speech_turns, long_fX = [], []
        shrt_speech_turns, shrt_fX = [], []

        # when long_first == False (i.e. mode == 'center'), there can only be
        # long speech turns (according to the above definition)
        mode = 'strict' if self.long_first else 'center'
        for speech_turn in speech_turns:
            fX = emb.crop(speech_turn, mode=mode)
            if len(fX):
                long_speech_turns.append(speech_turn)
                long_fX.append(np.sum(fX, axis=0))
            else:
                shrt_speech_turns.append(speech_turn)
                fX = emb.crop(speech_turn, mode='center')
                shrt_fX.append(np.sum(fX, axis=0))

        # stack long (and short) speech turns embedding into numpy arrays
        long_fX = np.vstack(long_fX)
        if shrt_speech_turns:
            shrt_fX = np.vstack(shrt_fX)

        # cluster long speech turns based on their normalized embeddings
        long_clusters = self.cls_.apply(l2_normalize(long_fX))

        # get a unique embedding for each cluster by summing the embedding of
        # all the included speech turns
        cluster_fX = np.vstack(
            [np.sum(long_fX[long_clusters == i], axis=0)
             for i in np.unique(long_clusters)])

        # associate each short speech turn to the closest cluster
        shrt_clusters = []
        if shrt_speech_turns:
            distances = cdist(l2_normalize(cluster_fX),
                              l2_normalize(shrt_fX),
                              metric=self.cls__metric)
            shrt_clusters = np.argmin(distances, axis=0)

        # build hypothesis from clustering results
        speech_turns = long_speech_turns + shrt_speech_turns
        clusters = list(long_clusters) + list(shrt_clusters)
        for speech_turn, cluster in zip(speech_turns, clusters):
            hypothesis[speech_turn] = cluster
        return hypothesis
