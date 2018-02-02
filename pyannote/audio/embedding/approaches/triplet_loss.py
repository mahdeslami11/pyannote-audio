#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2018 CNRS

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
from pyannote.audio.generators.speaker import SpeechSegmentGenerator
from pyannote.audio.callback import LoggingCallbackPytorch
from torch.optim import Adam
from pyannote.audio.embedding.utils import to_condensed
from scipy.spatial.distance import pdist, squareform
from pyannote.metrics.binary_classification import det_curve


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
    duration : float, optional
        Defautls to 3.2 seconds.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'cosine'.
    margin: float, optional
        Margin factor. Defaults to 0.2.
    clamp : {'positive', 'sigmoid', 'softmargin'}, optional
        Defaults to 'positive'.
    sampling : {'all', 'hard', 'negative'}, optional
        Triplet sampling strategy.
    per_label : int, optional
        Number of sequences per speaker in each batch. Defaults to 3.
    per_fold : int, optional
        If provided, sample triplets from groups of `per_fold` speakers at a
        time. Defaults to sample triplets from the whole speaker set.
    parallel : int, optional
        Number of prefetching background generators. Defaults to 1.
        Each generator will prefetch enough batches to cover a whole epoch.
        Set `parallel` to 0 to not use background generators.
    """

    def __init__(self, duration=3.2,
                 metric='cosine', margin=0.2, clamp='positive',
                 sampling='all', per_label=3, per_fold=None,
                 parallel=1):

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

        if clamp not in {'positive', 'sigmoid', 'softmargin'}:
            msg = "'clamp' must be one of {'positive', 'sigmoid', 'softmargin'}."
            raise ValueError(msg)
        self.clamp = clamp

        if sampling not in {'all', 'hard', 'negative'}:
            msg = "'sampling' must be one of {'all', 'hard', 'negative'}."
            raise ValueError(msg)
        self.sampling = sampling

        self.per_fold = per_fold
        self.per_label = per_label
        self.duration = duration
        self.parallel = parallel

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

        for i in range(n_sequences - 1):

            if self.metric in ('cosine', 'angular'):
                d = 1. - F.cosine_similarity(
                    fX[i, :].expand(n_sequences - 1 - i, -1),
                    fX[i+1:, :], dim=1, eps=1e-8)

                if self.metric == 'angular':
                    d = torch.acos(torch.clamp(1. - d, -1 + 1e-6, 1 - 1e-6))

            elif self.metric == 'euclidean':
                d = F.pairwise_distance(
                    fX[i, :].expand(n_sequences - 1 - i, -1),
                    fX[i+1:, :], p=2, eps=1e-06).view(-1)

            distances.append(d)

        return torch.cat(distances)

    def batch_hard(self, y, distances):
        """Build triplet with both hardest positive and hardest negative

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

            # hardest positive
            pos = np.where(y == y_anchor)[0]
            pos = [p for p in pos if p != anchor]
            positive = int(pos[np.argmax(d[pos])])

            # hardest negative
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

        return anchors, positives, negatives

    def batch_negative(self, y, distances):
        """Build triplet with hardest negative

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

            # hardest negative
            d = distances[anchor]
            neg = np.where(y != y_anchor)[0]
            negative = int(neg[np.argmin(d[neg])])

            for positive in np.where(y == y_anchor)[0]:
                if positive == anchor:
                    continue

                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

        return anchors, positives, negatives

    def batch_all(self, y, distances):
        """Build all possible triplet

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

    def triplet_loss(self, distances, anchors, positives, negatives,
                     return_delta=False):
        """Compute triplet loss

        Parameters
        ----------
        distances : torch.autograd.Variable
            Condensed matrix of pairwise distances.
        anchors, positives, negatives : list of int
            Triplets indices.
        return_delta : bool, optional
            Return delta before clamping.

        Returns
        -------
        loss : torch.autograd.Variable
            Triplet loss.
        """

        # estimate total number of embeddings from pdist shape
        n = int(.5 * (1 + np.sqrt(1 + 8 * len(distances))))
        n = [n] * len(anchors)

        # convert indices from squared matrix
        # to condensed matrix referential
        pos = list(map(to_condensed, n, anchors, positives))
        neg = list(map(to_condensed, n, anchors, negatives))

        # compute raw triplet loss (no margin, no clamping)
        # the lower, the better
        delta = distances[pos] - distances[neg]

        # clamp triplet loss
        if self.clamp == 'positive':
            loss = torch.clamp(delta + self.margin_, min=0)

        elif self.clamp == 'softmargin':
            loss = torch.log1p(torch.exp(delta))

        elif self.clamp == 'sigmoid':
            loss = F.sigmoid(10 * delta)

        # return triplet losses
        if return_delta:
            return loss, delta.view((-1, 1))
        else:
            return loss

    def fit(self, model, feature_extraction, protocol, log_dir, subset='train',
            epochs=1000, restart=None, gpu=False):

        import tensorboardX
        writer = tensorboardX.SummaryWriter(log_dir=log_dir)

        logging_callback = LoggingCallbackPytorch(
            log_dir=log_dir, restart=(False if restart is None else True))

        batch_generator = SpeechSegmentGenerator(
            feature_extraction,
            per_label=self.per_label, per_fold=self.per_fold,
            duration=self.duration, parallel=self.parallel)
        batches = batch_generator(protocol, subset=subset)
        batch = next(batches)

        batches_per_epoch = batch_generator.batches_per_epoch

        if restart is not None:
            weights_pt = logging_callback.WEIGHTS_PT.format(
                log_dir=log_dir, epoch=restart)
            model.load_state_dict(torch.load(weights_pt))

        if gpu:
            model = model.cuda()

        model.internal = False

        optimizer = Adam(model.parameters())
        if restart is not None:
            optimizer_pt = logging_callback.OPTIMIZER_PT.format(
                log_dir=log_dir, epoch=restart)
            optimizer.load_state_dict(torch.load(optimizer_pt))
            if gpu:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

        restart = 0 if restart is None else restart + 1
        for epoch in range(restart, restart + epochs):

            tloss_avg = 0.

            if epoch % 10 == 0:
                positive, negative = [], []
                norms = []

            desc = 'Epoch #{0}'.format(epoch)
            for i in tqdm(range(batches_per_epoch), desc=desc):

                model.zero_grad()

                batch = next(batches)

                X = Variable(torch.from_numpy(
                    np.array(np.rollaxis(batch['X'], 0, 2),
                             dtype=np.float32)))

                if gpu:
                    X = X.cuda()

                fX = model(X)

                if epoch % 10 == 0:
                    if gpu:
                        fX_ = fX.data.cpu().numpy()
                    else:
                        fX_ = fX.data.numpy()
                    norms.append(np.linalg.norm(fX_, axis=1))

                # pre-compute pairwise distances
                distances = self.pdist(fX)

                if epoch % 10 == 0:
                    if gpu:
                        distances_ = distances.data.cpu().numpy()
                    else:
                        distances_ = distances.data.numpy()
                    is_positive = pdist(batch['y'].reshape((-1, 1)), metric='chebyshev') < 1
                    positive.append(distances_[np.where(is_positive)])
                    negative.append(distances_[np.where(~is_positive)])

                # sample triplets
                triplets = getattr(self, 'batch_{0}'.format(self.sampling))
                anchors, positives, negatives = triplets(batch['y'], distances)

                # compute triplet loss
                losses = self.triplet_loss(
                    distances, anchors, positives, negatives)

                loss = torch.mean(losses)

                # log loss
                if gpu:
                    loss_ = float(loss.data.cpu().numpy())
                else:
                    loss_ = float(loss.data.numpy())
                tloss_avg += loss_

                loss.backward()
                optimizer.step()

            tloss_avg /= batches_per_epoch
            writer.add_scalar('tloss', tloss_avg, global_step=epoch)

            if epoch % 10 == 0:

                positive = np.hstack(positive)
                negative = np.hstack(negative)
                writer.add_histogram(
                    'embedding/pairwise_distance/positive', positive,
                    global_step=epoch, bins='auto')
                writer.add_histogram(
                    'embedding/pairwise_distance/negative', negative,
                    global_step=epoch, bins='auto')

                _, _, _, eer = det_curve(
                    np.hstack([np.ones(len(positive)), np.zeros(len(negative))]),
                    np.hstack([positive, negative]), distances=True)
                writer.add_scalar('eer', eer, global_step=epoch)

                norms = np.hstack(norms)
                writer.add_histogram(
                    'embedding/norm', norms,
                    global_step=epoch, bins='auto')

            logging_callback.model = model
            logging_callback.optimizer = optimizer
            logging_callback.on_epoch_end(epoch)
