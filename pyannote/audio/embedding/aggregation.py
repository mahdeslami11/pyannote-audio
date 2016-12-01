#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

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
from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pyannote.generators.batch import FileBasedBatchGenerator
from pyannote.generators.fragment import SlidingSegments
from pyannote.audio.generators.yaafe import YaafeMixin
from ..features.utils import get_wav_duration


class SequenceEmbeddingAggregation(YaafeMixin, FileBasedBatchGenerator):
    """Embedding extraction

    Parameters
    ----------
    sequence_embedding : SequenceEmbedding
        Pre-trained sequence embedding.
    feature_extractor : YaafeFeatureExtractor
        Yaafe feature extractor
    duration : float, optional
    step : float, optional
        Sliding window duration and step (in seconds).
        Defaults to 1 second window with 100ms step.
    layer_index : int, optional
        Index of layer for which to return the activation.
        Defaults to returning the activation of the final layer.

    Usage
    -----
    >>> sequence_embedding = SequenceEmbedding.from_disk('architecture_yml', 'weights.h5')
    >>> feature_extraction = YaafeFeatureExtractor(...)
    >>> extraction = SequenceEmbeddingAggregation(sequence_embedding, feature_extraction)
    >>> embedding = extraction.apply('audio.wav')

    """
    def __init__(self, sequence_embedding, feature_extractor,
                 duration=5.0, min_duration=None, step=1.0, layer_index=None):

        # feature extractor (e.g. YaafeMFCC)
        self.feature_extractor = feature_extractor

        # sequence embedding
        self.sequence_embedding = sequence_embedding

        #
        self.duration = duration
        self.min_duration = min_duration
        self.step = step
        generator = SlidingSegments(
            duration=duration, min_duration=min_duration,
            step=step, source='annotation')

        self.layer_index = layer_index

        # pre-compute shape of zero-padded sequences
        n_features = self.feature_extractor.dimension()
        n_samples = self.feature_extractor.sliding_window().samples(
            self.duration, mode='center')
        self.shape_ = (n_samples, n_features)

        super(SequenceEmbeddingAggregation, self).__init__(
            generator, batch_size=-1)

    def signature(self):
        shape = self.shape
        return ({'type': 'segment'},
                {'type': 'sequence', 'shape': shape})

    def process_segment(self, segment, signature=None, identifier=None):
        return (segment,
                super(SequenceEmbeddingAggregation, self).process_segment(
                    segment, signature=signature, identifier=identifier))

    def pack_sequence(self, sequences):
        zero_padded = []
        masks = []
        for sequence in sequences:
            n_samples = min(self.shape_[0], sequence.shape[0])

            zeros = np.zeros(self.shape_, dtype=np.float32)
            zeros[:n_samples, :] = sequence[:n_samples]
            zero_padded.append(zeros)

            mask = np.zeros((self.shape_[0], 1), dtype=np.int8)
            mask[:n_samples] = 1
            masks.append(mask)

        return np.stack(zero_padded), np.stack(masks)

    def postprocess_sequence(self, mono_batch):
        sequences, masks = mono_batch
        return self.sequence_embedding.transform(
            sequences, layer_index=self.layer_index), masks

    def apply(self, wav, from_annotation):
        """Compute embeddings on a partition

        Parameter
        ---------
        wav : str
            Path to wav audio file
        from_annotation : Annotation

        Returns
        -------
        embeddings : SlidingWindowFeature
        """

        current_file = {'uri': wav,
                        'wav': wav,
                        'annotation': from_annotation}
        (segments, (embeddings, masks)) = next(self.from_file(current_file))

        # HACK make sure last segment is cropped at wav duration
        duration = get_wav_duration(wav)
        segments[-1] = segments[-1] & Segment(0., duration)

        n_sequences, _, dimension = embeddings.shape

        # estimate total number of frames (over the duration of the whole file)
        # based on feature extractor internal sliding window and file duration
        samples_window = self.feature_extractor.sliding_window()
        n_samples = samples_window.samples(get_wav_duration(wav)) + 3

        # +3 is a hack to avoid later IndexError resulting from rounding error
        # when cropping samples_window

        # k[i] contains the number of sequences that overlap with frame #i
        k = np.zeros((n_samples, 1), dtype=np.int8)

        # fX[i] contains the sum of embeddings for frame #i
        # over all overlapping samples
        fX = np.zeros((n_samples, dimension), dtype=np.float32)

        for i, segment in enumerate(segments):

            # indices of frames overlapped by sequence #i
            indices = samples_window.crop(
                segment, mode='center', fixed=self.duration)

            k[indices] += masks[i]
            fX[indices] += embeddings[i] * masks[i]

        fX = fX / np.maximum(k, 1)

        return SlidingWindowFeature(fX, samples_window)
