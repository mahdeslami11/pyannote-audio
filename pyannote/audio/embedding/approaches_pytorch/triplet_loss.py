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

import itertools
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pyannote.audio.generators.speaker import SpeechTurnGenerator
from pyannote.audio.callback import LoggingCallbackPytorch
from torch.optim import RMSprop
from pyannote.audio.embedding.utils import to_condensed
from scipy.spatial.distance import squareform


class TripletLoss(object):
    """

    delta = d(anchor, positive) - d(anchor, negative)

    * with 'positive' clamping:
        loss = max(0, delta + margin x D)
    * with 'sigmoid' clamping:
        loss = sigmoid(10 * delta)

    where d(., .) varies in range [0, D] (e.g. D=2 for euclidean distance).

    Parameters
    ----------
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'cosine'.
    margin: float, optional
        Margin factor. Defaults to 0.2.
    clamp : {'positive', 'sigmoid'}, optional
        Defaults to 'positive'.
    sampling : {'all', 'hard'}, optional
        Triplet sampling strategy.
    per_label : int, optional
        Number of sequences per speaker in each batch. Defaults to 3.
    per_fold : int, optional
        If provided, sample triplets from groups of `per_fold` speakers at a
        time. Defaults to sample triplets from the whole speaker set.
    """

    def __init__(self, duration=3.2,
                 metric='cosine', margin=0.2, clamp='positive',
                 sampling='all', per_label=3, per_fold=None):

        super(TripletLoss, self).__init__()

        self.metric = metric
        self.margin = margin

        if self.metric == 'cosine':
            self.margin_ = self.margin * 2.
        elif self.metric == 'angular':
            self.margin_ = self.margin * np.pi
        elif self.metric == 'euclidean':
            self.margin_ = self.margin * 2.
        else:
            msg = "'metric' must be one of {'euclidean', 'cosine', 'angular'}."
            raise ValueError(msg)

        if clamp not in {'positive', 'sigmoid'}:
            msg = "'clamp' must be one of {'positive', 'sigmoid'}."
            raise ValueError(msg)
        self.clamp = clamp

        if sampling not in {'all', 'hard'}:
            msg = "'sampling' must be one of {'all', 'hard'}."
            raise ValueError(msg)
        self.sampling = sampling

        self.per_fold = per_fold
        self.per_label = per_label
        self.duration = duration

    def pdist(self, fX):
        """Compute pdist à-la scipy.spatial.distance.pdist

        Parameters
        ----------
        fX : (n, d) torch.autograd.Variable
            Embeddings.

        Returns
        -------
        distances : (n * (n-1) / 2,) torch.autograd.Variable
            Condensed pairwise distance matrix
        """

        n_sequences, _ = fX.size()
        distances = []

        for i in range(1, n_sequences):

            if self.metric in ('cosine', 'angular'):

                d = 1. - F.cosine_similarity(
                    fX[i, :].expand(i, -1), fX[:i, :],
                    dim=1, eps=1e-8)

                if self.metric == 'angular':
                    d = torch.acos(1. - d)

            elif self.metric == 'euclidean':
                d = F.pairwise_distance(
                    fX[i, :].expand(i, -1), fX[:i, :],
                    p=2, eps=1e-06)

            distances.append(d)

        return torch.cat(distances)

    def batch_hard(self, y, distances):
        """

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n * (n-1) / 2,) torch.autograd.Variable
            Condensed pairwise distance matrix

        Returns
        -------
        anchors, positives, negatives : list of int
            Triplets indices.
        """

        anchors, positives, negatives = [], [], []

        if distances.is_cuda:
            distances = squareform(distances.data.cpu().numpy())
        else:
            distances = squareform(distances.data.numpy())
        y = np.array(y)

        for anchor, y_anchor in enumerate(y):

            d = distances[anchor]
            pos = np.where(y == y_anchor)[0]

            # hardest positive
            positive = pos[np.argmax(d[pos])]

            # hardest negative
            neg = np.where(y != y_anchor)[0]
            negative = neg[np.argmin(d[neg])]

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return anchors, positives, negatives

    def batch_all(self, y, distances):
        """

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n * (n-1) / 2,) torch.autograd.Variable
            Condensed pairwise distance matrix

        Returns
        -------
        anchors, positives, negatives : list of int
            Triplets indices.
        """

        anchors, positives, negatives = [], [], []

        for anchor, y_anchor in enumerate(y):
            for positive, y_positive in enumerate(y):

                # if same embedding or different labels, skip
                if (anchor == positive) or (y_anchor != y_positive):
                    continue

                for negative, y_negative in enumerate(y):

                    if y_negative == y_anchor:
                        continue

                    anchors.append(anchor)
                    positives.append(positive)
                    negatives.append(negative)

        return anchors, positives, negatives

    def triplet_loss(self, distances, anchors, positives, negatives):
        """Compute average triplet loss

        Parameters
        ----------
        distances : torch.autograd.Variable
            Condensed matrix of pairwise distances.
        anchors, positives, negatives : list of int
            Triplets indices.

        Returns
        -------
        loss : torch.autograd.Variable
            Average triplet loss.
        """

        # convert indices from squared matrix
        # to condensed matrix referential
        p = list(map(to_condensed, anchors, positives))
        n = list(map(to_condensed, anchors, negatives))

        # compute raw triplet loss (no margin, no clamping)
        delta = distances[p] - distances[n]

        # clamp triplet loss
        if self.clamp == 'positive':
            loss = torch.clamp(delta + self.margin_, min=0)

        elif self.clamp == 'sigmoid':
            loss = F.sigmoid(10 * delta)

        # return average triplet loss
        return torch.mean(loss)

    def fit(self, model, feature_extraction, protocol, log_dir, subset='train',
            n_epochs=1000, gpu=False):

        logging_callback = LoggingCallbackPytorch(log_dir=log_dir)

        try:
            batch_generator = SpeechTurnGenerator(
                feature_extraction,
                per_label=self.per_label, per_fold=self.per_fold,
                duration=self.duration)
            batches = batch_generator(protocol, subset=subset)
            batch = next(batches)
        except OSError as e:
            del batch_generator.data_
            batch_generator = SpeechTurnGenerator(
                feature_extraction,
                per_label=self.per_label, per_fold=self.per_fold,
                duration=self.duration, fast=False)
            batches = batch_generator(protocol, subset=subset)
            batch = next(batches)

        # one minute per speaker
        duration_per_epoch = 60. * batch_generator.n_labels
        duration_per_batch = self.duration * batch_generator.n_sequences_per_batch
        batches_per_epoch = int(np.ceil(duration_per_epoch / duration_per_batch))

        if gpu:
            model = model.cuda()

        optimizer = RMSprop(model.parameters())
        model.internal = False

        while True:

            for epoch in range(n_epochs):

                running_tloss = 0.

                desc = 'Epoch #{0}'.format(epoch)
                for _ in tqdm(range(batches_per_epoch), desc=desc):

                    model.zero_grad()

                    batch = next(batches)

                    X = Variable(torch.from_numpy(
                        np.array(np.rollaxis(batch['X'], 0, 2),
                                 dtype=np.float32)))

                    if gpu:
                        X = X.cuda()

                    fX = model(X)

                    # pre-compute pairwise distances
                    distances = self.pdist(fX)

                    # sample triplets
                    if self.sampling == 'all':
                        anchors, positives, negatives = self.batch_all(
                            batch['y'], distances)

                    elif self.sampling == 'hard':
                        anchors, positives, negatives = self.batch_hard(
                            batch['y'], distances)

                    # compute triplet loss
                    loss = self.triplet_loss(
                        distances, anchors, positives, negatives)

                    loss.backward()
                    optimizer.step()

                    if gpu:
                        running_tloss += float(loss.data.cpu().numpy())
                    else:
                        running_tloss += float(loss.data.numpy())

                running_tloss /= batches_per_epoch

                logs = {'loss': running_tloss}
                logging_callback.model = model
                logging_callback.optimizer = optimizer
                logging_callback.on_epoch_end(epoch, logs=logs)
