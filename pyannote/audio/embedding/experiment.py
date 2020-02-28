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

from typing import Union
from typing import Text
from typing import List
from typing import Tuple
from pathlib import Path

import warnings
import itertools
from tqdm import tqdm

import numpy as np
from pyannote.database.protocol.protocol import Protocol
from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.audio.features import Precomputed

from pyannote.audio.applications.speaker_embedding import SpeakerEmbedding
get_hash = lambda file: SpeakerEmbedding.get_hash(file)

from pyannote.core.utils.distance import cdist
from pyannote.core.utils.distance import pdist

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from pyannote.metrics.binary_classification import Calibration


class Experiment:
    """

    Parameters
    ----------
    protocol : Protocol
    embedding : Path
        Path to directory containing precomputed embeddings.
        Describes how raw speaker embeddings should be obtained. It can be
        either the name of a torch.hub model, or the path to the output of the
        validation step of a model trained locally, or the path to embeddings
        precomputed on disk. Defaults to "@emb" that indicates that protocol
        files provide the embeddings in the "emb" key.
    metric : Text, optional
        Metric to use to compare embeddings. Defaults to 'cosine'.
    cohort : 'train' or 'development', optional
        Subset to use as cohort.
    """

    def __init__(self, embedding: Path,
                       metric : Text = 'cosine'):
        super().__init__()

        self.embedding = embedding
        self.metric = metric

        self._precomputed = Precomputed(root_dir=self.embedding)

        window = self._precomputed.sliding_window
        self._downsample = max(1, int(.5 * window.duration / window.step))

        self._gpu = TORCH_AVAILABLE and torch.cuda.is_available()


    def get_hash(self, file: ProtocolFile):
        hashable = []
        for f in file.files():
            if 'try_with' in f:
                hashable.append((f['uri'], tuple(f['try_with'])))
            else:
                hashable.append((f['uri'], tuple(f['annotation'].get_timeline())))
        return hash(tuple(sorted(hashable)))

    def get_embedding(self, file: ProtocolFile,
                            mean: bool = False,
                            stack: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """Get embedding from file

        Parameters
        ----------
        file : ProtocolFile
            File
        mean : bool, optional
            Return average embedding. Defaults to return embedding arrays.
        stack : bool, optional
            Stack embedding arrays. Defaults to return list of embeddings array.
        cache : bool, optional
            Cache result.

        Returns
        -------
        embedding : np.ndarray or list of np.ndarray
        """

        emb = []
        for f in file.files():

            if 'try_with' in f:
                segments = f['try_with']
            else:
                segments = f['annotation'].get_timeline()

            for segment in segments:
                for mode in ['center', 'loose']:
                    e = self._precomputed.crop(f, segment, mode=mode)
                    if len(e) > 0:
                        break
                emb.append(e)

        if stack:
            emb = np.vstack(emb)
            if mean:
                emb = np.mean(emb, axis=0, keepdims=True)
        else:
            if mean:
                emb = [np.mean(e, axis=0, keepdims=True) for e in emb]

        return emb

    def pdist(self, emb) -> np.ndarray:
        """Compute pairwise distance

        Parameters
        ----------
        emb : (n_samples, dimension) np.ndarray

        Returns
        -------
        distance : np.ndarray

        """

        if not self._gpu:
            return pdist(emb, metric=self.metric)

        emb_gpu = torch.tensor(emb).to('cuda')

        if self.metric == 'euclidean':
            distance = F.pdist(emb_gpu, 2)

        elif self.metric in ('cosine', 'angular'):
            distance = 0.5 * torch.pow(F.pdist(F.normalize(emb_gpu), 2), 2)

            if self.metric == 'angular':
                distance =  torch.acos(torch.clamp(1. - distance,
                                                   -1 + 1e-12,
                                                   1 - 1e-12))
        return distance.to('cpu').numpy()

    def cdist(self, emb1, emb2) -> np.ndarray:
        """Compute distance between collections

        Parameters
        ----------
        emb1 : (n_samples_1, dimension) np.ndarray
        emb2 : (n_samples_2, dimension) np.ndarray

        Returns
        -------
        distance : (n_samples_1, n_samples_2) np.ndarray
        """

        if not self._gpu:
            return cdist(emb1, emb2, metric=self.metric)

        emb1_gpu = torch.tensor(emb1).to('cuda')
        emb2_gpu = torch.tensor(emb2).to('cuda')

        if self.metric == 'euclidean':
            distance = torch.cdist(emb1_gpu, emb2_gpu, 2)

        elif self.metric in ('cosine', 'angular'):
            distance = 0.5 * torch.pow(torch.cdist(F.normalize(emb1_gpu),
                                                   F.normalize(emb2_gpu),
                                                   2), 2)

            if self.metric == 'angular':
                distance =  torch.acos(torch.clamp(1. - distance,
                                                   -1 + 1e-12,
                                                   1 - 1e-12))

        return distance.to('cpu').numpy()

    def get_positive(self, embeddings: List[np.ndarray],
                           downsample: bool = False) -> np.ndarray:
        """Get distribution of positive distances

        Parameters
        ----------
        embeddings : list of np.ndarray
            One np.ndarray per file. All files must come from the same speaker
        downsample : bool, optional
            Downsample embeddings to prevent embeddings with more than 50% of
            overlap (which would lead to over-optimistic positive distances).

        Returns
        -------
        positive : np.ndarray
            Sampled positive distances
        """

        positive = []

        if len(embeddings) == 1:
            emb = embeddings[0]
            if downsample:
                emb = emb[::self._downsample]
            else:
                i = np.random.choice(len(emb), size=min(5, len(emb)), replace=False)
                emb = emb[i]
            return self.pdist(emb)

        for emb1, emb2 in itertools.combinations(embeddings, 2):
            if downsample:
                d = self.cdist(emb1[::self._downsample], emb2[::self._downsample])
            else:
                i1 = np.random.choice(len(emb1), size=min(5, len(emb1)), replace=False)
                i2 = np.random.choice(len(emb2), size=min(5, len(emb2)), replace=False)
                d = self.cdist(emb1[i1], emb2[i2])
            positive.append(d.reshape((-1, )))
        return np.hstack(positive)

    def load_cohort(self, protocol: Protocol,
                          subset: Text = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """Load cohort

        Returns
        -------
        self._cohort : (..., dimension) np.ndarray
        self._positive : (..., ) np.ndarray


        """

        files = getattr(protocol, subset)()

        cohort_embedding = dict()
        desc = 'Loading cohort...'
        for file in tqdm(files, desc=desc):
            speaker = file['annotation'].argmax()
            embedding = self.get_embedding(file,
                                           mean=False,
                                           stack=True)
            cohort_embedding.setdefault(speaker, []).append(embedding)

        cohort = []
        positive = []
        desc = 'Estimating positive distribution on cohort...'
        for speaker, embeddings in tqdm(cohort_embedding.items(), desc=desc):

            # choose at most 10 files per cohort speaker
            ten = np.random.choice(embeddings, size=min(10, len(embeddings)), replace=False)

            # compute inter-file distance
            positive.append(self.get_positive(ten, downsample=False))

            # choose at most 5 embeddings per file
            for emb in ten:
                i = np.random.choice(len(emb), size=min(5, len(emb)), replace=False)
                cohort.append(emb[i])

        return np.vstack(cohort), np.hstack(positive)

    def get_negative(self, embeddings: List[np.ndarray],
                           cohort: np.ndarray = None) -> np.ndarray:
        """
        """

        if cohort is None:
            cohort = self._cohort

        negative = []
        for emb in embeddings:
            negative.append(self.cdist(emb, cohort).reshape((-1, )))

        return np.hstack(negative)

    def __call__(self, protocol: Protocol,
                       cohort: Text = 'train',
                       subset: Text = 'test'):

        self._cache_embedding = dict()
        self._cache_calibration = dict()

        if getattr(self, '_cohort', None) is not None:
            warnings.warn('using preloaded cohort...')
            cohort = self._cohort
            positive = self._positive
        else:
            cohort, positive = self.load_cohort(protocol, subset=cohort)
            self._cohort = cohort
            self._positive = positive

        y_true, prob = [], []

        desc = f'{subset} trials...'
        trials = getattr(protocol, f'{subset}_trial')()
        for t, trial in enumerate(tqdm(trials, desc=desc)):

            if t > 10:
                break

            file1 = trial['file1']
            hash1 = self.get_hash(file1)

            if hash1 in self._cache_embedding:
                emb1 = self._cache_embedding[hash1]
                calibration1 = self._cache_calibration[hash1]

            else:
                emb1 = self.get_embedding(file1, mean=False, stack=False)
                self._cache_embedding[hash1] = emb1

                if len(emb1) > 1:
                    pos1 = self.get_positive(emb1, downsample=False)
                else:
                    pos1 = positive
                neg1 = self.get_negative(emb1, cohort=cohort)[::100]

                calibration1 = Calibration(equal_priors=True, method='isotonic')
                calibration1.fit(
                    -np.hstack([pos1, neg1]),
                    np.hstack([np.ones(len(pos1), dtype=np.float32),
                               np.zeros(len(neg1), dtype=np.float32)]))
                self._cache_calibration[hash1] = calibration1

            file2 = trial['file2']
            hash2 = self.get_hash(file2)

            if hash2 in self._cache_embedding:
                emb2 = self._cache_embedding[hash2]
                calibration2 = self._cache_calibration[hash2]

            else:
                emb2 = self.get_embedding(file2, mean=False, stack=False)
                self._cache_embedding[hash2] = emb2

                if len(emb2) > 2:
                    pos2 = self.get_positive(emb2, downsample=False)
                else:
                    pos2 = positive
                neg2 = self.get_negative(emb2, cohort=cohort)[::100]

                calibration2 = Calibration(equal_priors=True, method='isotonic')
                calibration2.fit(
                    -np.hstack([pos2, neg2]),
                    np.hstack([np.ones(len(pos2), dtype=np.float32),
                               np.zeros(len(neg2), dtype=np.float32)]))
                self._cache_calibration[hash2] = calibration2

            distance = self.cdist(np.vstack(emb1), np.vstack(emb2)).reshape((-1, ))
            prob1 = calibration1.transform(-distance)
            prob2 = calibration2.transform(-distance)

            prob.append(np.mean(.5 * (prob1 + prob2)))
            y_true.append(trial['reference'])

        return y_true, prob




if __name__ == '__main__':

    pass
