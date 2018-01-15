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
from tqdm import tqdm
import torch
from torch.autograd import Variable, Function
import torch.nn.functional as F
import torch.nn as nn
from pyannote.audio.generators.speaker import SpeechTurnGenerator
from pyannote.audio.callback import LoggingCallbackPytorch
from torch.optim import Adam
from .triplet_loss import TripletLoss


class GradReverse(Function):
    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.alpha)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class DomainClassifier(nn.Module):

    def __init__(self, n_dimensions, n_domains, alpha=1.):
        super(DomainClassifier, self).__init__()
        n_hidden = 100
        self.fc1 = nn.Linear(n_dimensions, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_domains)
        self.drop = nn.Dropout2d(0.25)
        self.alpha = alpha

    def forward(self, x):
        x = grad_reverse(x, self.alpha)
        x = F.leaky_relu(self.drop(self.fc1(x)))
        x = self.fc2(x)
        return F.softmax(x, dim=0)

class DomainAdversarialTripletLoss(TripletLoss):
    """

    delta = d(anchor, positive) - d(anchor, negative)

    * with 'positive' clamping:
        loss = max(0, delta + margin x D)
    * with 'sigmoid' clamping:
        loss = sigmoid(10 * delta)

    where d(., .) varies in range [0, D] (e.g. D=2 for euclidean distance).

    Parameters
    ----------
    domain : {'database'}, optional
        Domain. Typically 'database'.
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Defaults to 'cosine'.
    margin: float, optional
        Margin factor. Defaults to 0.2.
    clamp : {'positive', 'sigmoid', 'softmargin'}, optional
        Defaults to 'positive'.
    sampling : {'all', 'hard'}, optional
        Triplet sampling strategy.
    per_label : int, optional
        Number of sequences per speaker in each batch. Defaults to 3.
    per_fold : int, optional
        If provided, sample triplets from groups of `per_fold` speakers at a
        time. Defaults to sample triplets from the whole speaker set.
    """

    def __init__(self, domain='database', duration=3.2,
                 metric='cosine', margin=0.2, clamp='positive',
                 sampling='all', per_label=3, per_fold=None):

        super(DomainAdversarialTripletLoss, self).__init__(
            duration=duration, metric=metric, margin=margin, clamp=clamp,
            sampling=sampling, per_label=per_label, per_fold=per_fold)
        self.domain = domain

    def fit(self, model, feature_extraction, protocol, log_dir, subset='train',
            epochs=1000, restart=None, gpu=False):

        logging_callback = LoggingCallbackPytorch(
            log_dir=log_dir, restart=(False if restart is None else True))

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

        if restart is not None:
            weights_pt = logging_callback.WEIGHTS_PT.format(
                log_dir=log_dir, epoch=restart)
            model.load_state_dict(torch.load(weights_pt))

        if gpu:
            model = model.cuda()

        model.internal = False

        n_domains = len(batch_generator.domains_[self.domain])
        if n_domains < 2:
            raise ValueError('There must be more than one domain.')

        domain_clf = DomainClassifier(model.output_dim, n_domains, alpha=1.)
        if gpu:
            domain_clf = domain_clf.cuda()

        domain_loss = nn.CrossEntropyLoss()

        optimizer = Adam(list(model.parameters()) + list(domain_clf.parameters()))
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

            loss_min = np.inf
            loss_max = -np.inf
            loss_avg = 0.

            nonzero_min = 1.
            nonzero_max = 0.
            nonzero_avg = 0.

            distances_avg = 0.
            norm_avg = 0.

            desc = 'Epoch #{0}'.format(epoch)
            for _ in tqdm(range(batches_per_epoch), desc=desc):

                model.zero_grad()

                batch = next(batches)

                X = Variable(torch.from_numpy(
                    np.array(np.rollaxis(batch['X'], 0, 2),
                             dtype=np.float32)))

                y_domain = batch['y_{domain}'.format(domain=self.domain)])
                y_domain = Variable(torch.from_numpy(np.array(y_domain))

                if gpu:
                    X = X.cuda()
                    y_domain = y_domain.cuda()

                fX = model(X)
                if gpu:
                    fX_ = fX.data.cpu().numpy()
                else:
                    fX_ = fX.data.numpy()
                norm_avg += np.mean(np.linalg.norm(fX_, axis=0))
                # TODO. percentile

                # pre-compute pairwise distances
                distances = self.pdist(fX)

                if gpu:
                    distances_ = distances.data.cpu().numpy()
                else:
                    distances_ = distances.data.numpy()
                distances_avg += np.mean(distances_)
                # TODO. percentile

                # sample triplets
                if self.sampling == 'all':
                    anchors, positives, negatives = self.batch_all(
                        batch['y'], distances)

                elif self.sampling == 'hard':
                    anchors, positives, negatives = self.batch_hard(
                        batch['y'], distances)

                # compute triplet loss
                triplet_losses = self.triplet_loss(
                    distances, anchors, positives, negatives)

                domain_score = domain_clf(fX)
                dloss = domain_loss(domain_score, y_domain)

                # log ratio of non-zero triplets
                if gpu:
                    nonzero_ = np.mean(triplet_losses.data.cpu().numpy() > 0)
                else:
                    nonzero_ = np.mean(triplet_losses.data.numpy() > 0)
                nonzero_avg += nonzero_
                nonzero_min = min(nonzero_min, nonzero_)
                nonzero_max = max(nonzero_max, nonzero_)

                loss = torch.mean(triplet_losses) + dloss

                # log batch loss
                if gpu:
                    loss_ = float(loss.data.cpu().numpy())
                else:
                    loss_ = float(loss.data.numpy())
                loss_avg += loss_
                loss_min = min(loss_min, loss_)
                loss_max = max(loss_max, loss_)

                loss.backward()
                optimizer.step()

            loss_avg /= batches_per_epoch
            nonzero_avg /= batches_per_epoch
            distances_avg /= batches_per_epoch
            norm_avg /= batches_per_epoch

            logs = {'loss_avg': loss_avg,
                    'loss_min': loss_min,
                    'loss_max': loss_max,
                    'nonzero_avg': nonzero_avg,
                    'nonzero_max': nonzero_max,
                    'nonzero_min': nonzero_min,
                    'distances_avg': distances_avg,
                    'norm_avg': norm_avg}

            logging_callback.model = model
            logging_callback.optimizer = optimizer
            logging_callback.on_epoch_end(epoch, logs=logs)
