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
from pyannote.audio.generators.speaker import SpeechTurnSubSegmentGenerator
from pyannote.audio.callback import LoggingCallbackPytorch
from torch.optim import Adam
from pyannote.audio.embedding.utils import to_condensed
from scipy.spatial.distance import pdist, squareform
from .triplet_loss import TripletLoss


class AggTripletLoss(TripletLoss):
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
    clamp : {'positive', 'sigmoid', 'softmargin'}, optional
        Defaults to 'positive'.
    sampling : {'all', 'hard'}, optional
        Triplet sampling strategy.
    per_label : int, optional
        Number of speech turns per speaker in each batch. Defaults to 3.
    per_fold : int, optional
        Number of speakers in each batch. Defaults to all speakers.
    per_turn : int, optional
        Number of segments per speech turn. Defaults to 10.
        For short speech turns, a heuristic adapts this number to reduce the
        number of overlapping segments.
    normalize : boolean, optional
        Normalize between aggregation and distance computation.
    """

    def __init__(self, duration=3.2,
                 metric='cosine', margin=0.2, clamp='positive',
                 sampling='all', per_label=3, per_fold=None, per_turn=10,
                 normalize=False):

        super(AggTripletLoss, self).__init__(
            duration=duration, metric=metric, margin=margin, clamp=clamp,
            sampling=sampling, per_label=per_label, per_fold=per_fold)

        self.per_turn = per_turn
        self.normalize = normalize

    def fit(self, model, feature_extraction, protocol, log_dir, subset='train',
            epochs=1000, restart=None, gpu=False):

        import tensorboardX
        writer = tensorboardX.SummaryWriter(log_dir=log_dir)

        logging_callback = LoggingCallbackPytorch(
            log_dir=log_dir, restart=(False if restart is None else True))

        try:
            batch_generator = SpeechTurnSubSegmentGenerator(
                feature_extraction, self.duration,
                per_label=self.per_label, per_fold=self.per_fold,
                per_turn=self.per_turn)
            batches = batch_generator(protocol, subset=subset)
            batch = next(batches)
        except OSError as e:
            del batch_generator.data_
            batch_generator = SpeechTurnSubSegmentGenerator(
                feature_extraction, self.duration,
                per_label=self.per_label, per_fold=self.per_fold,
                per_turn=self.per_turn, fast=False)
            batches = batch_generator(protocol, subset=subset)
            batch = next(batches)

        # TODO. learn feature normalization and store it as a layer in the model

        # one minute per speaker
        duration_per_epoch = 60. * batch_generator.n_labels
        duration_per_batch = self.duration * batch_generator.n_sequences_per_batch
        batches_per_epoch = int(np.ceil(duration_per_epoch / duration_per_batch))

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

            loss_avg = 0.
            positive, negative = [], []
            if not model.normalize:
                norms = []

            desc = 'Epoch #{0}'.format(epoch)
            for i in tqdm(range(batches_per_epoch), desc=desc):

                model.zero_grad()

                batch = next(batches)

                n_segments = len(batch['X'])

                X = torch.from_numpy(np.array(np.rollaxis(batch['X'], 0, 2),
                                              dtype=np.float32))

                X = Variable(X, requires_grad=False)
                if gpu:
                    X = X.cuda()
                fX = model(X)

                fX_avg = []
                y = []

                # iterate over segments, speech turn by speech turn
                nyz = np.vstack([np.arange(n_segments), batch['y'], batch['z']]).T
                for _, nyz_ in itertools.groupby(nyz, lambda v: v[2]):

                    # (n, 3) numpy array where
                    # * n is the number of segments in current speech turn
                    # * dim #0 is the index of segment in original batch
                    # * dim #1 is the label of speech turn
                    # * dim #2 is the index of speech turn (used for grouping)
                    nyz_ = np.stack(nyz_)

                    # compute (and stack) average embedding over all segments
                    # of current speech turn
                    indices = nyz_[:, 0]
                    fX_avg.append(
                        torch.mean(fX[indices], dim=0, keepdim=True))

                    # stack label of current speech turn
                    # (for later triplet sampling)
                    y.append(nyz_[0, 1])

                fX_avg = torch.cat(fX_avg, dim=0)

                if self.normalize:
                    fX_avg = fX_avg / torch.norm(fX_avg, 2, 1, keepdim=True)

                y = np.array(y)

                # pre-compute pairwise distances
                distances = self.pdist(fX_avg)

                if gpu:
                    distances_ = distances.data.cpu().numpy()
                else:
                    distances_ = distances.data.numpy()
                is_positive = pdist(y.reshape((-1, 1)), metric='chebyshev') < 1
                positive.append(distances_[np.where(is_positive)])
                negative.append(distances_[np.where(~is_positive)])

                # sample triplets
                if self.sampling == 'all':
                    anchors, positives, negatives = self.batch_all(
                        y, distances)

                elif self.sampling == 'hard':
                    anchors, positives, negatives = self.batch_hard(
                        y, distances)

                # compute triplet loss
                losses = self.triplet_loss(
                    distances, anchors, positives, negatives)

                loss = torch.mean(losses)

                # log batch loss
                if gpu:
                    loss_ = float(loss.data.cpu().numpy())
                else:
                    loss_ = float(loss.data.numpy())
                loss_avg += loss_
                writer.add_scalar(
                    'loss/batch', loss_,
                    global_step=i + epoch * batches_per_epoch)

                loss.backward()
                optimizer.step()

            loss_avg /= batches_per_epoch
            writer.add_scalar('loss', loss_avg, global_step=epoch)

            positive = np.hstack(positive)
            negative = np.hstack(negative)
            writer.add_histogram(
                'embedding/pairwise_distance/positive', positive,
                global_step=epoch, bins='auto')
            writer.add_histogram(
                'embedding/pairwise_distance/negative', negative,
                global_step=epoch, bins='auto')

            logging_callback.model = model
            logging_callback.optimizer = optimizer
            logging_callback.on_epoch_end(epoch)
