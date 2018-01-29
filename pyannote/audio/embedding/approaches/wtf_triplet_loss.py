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

import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from pyannote.audio.generators.speaker import SpeechSegmentGenerator
from pyannote.audio.callback import LoggingCallbackPytorch
from torch.optim import Adam
from scipy.spatial.distance import pdist
from .triplet_loss import TripletLoss


class WTFTripletLoss(TripletLoss):
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
    margin: float, optional
        Margin factor. Defaults to 0.2.
    sampling : {'all', 'hard'}, optional
        Triplet sampling strategy.
    per_label : int, optional
        Number of sequences per speaker in each batch. Defaults to 3.
    per_fold : int, optional
        If provided, sample triplets from groups of `per_fold` speakers at a
        time. Defaults to sample triplets from the whole speaker set.
    """

    def __init__(self, duration=3.2, sampling='all',
                 per_label=3, per_fold=None):

        super(WTFTripletLoss, self).__init__(
            duration=duration, metric='angular', clamp='sigmoid',
            sampling=sampling, per_label=per_label, per_fold=per_fold)

    def fit(self, model, feature_extraction, protocol, log_dir, subset='train',
            epochs=1000, restart=None, gpu=False):

        import tensorboardX
        writer = tensorboardX.SummaryWriter(log_dir=log_dir)

        logging_callback = LoggingCallbackPytorch(
            log_dir=log_dir, restart=(False if restart is None else True))

        try:
            batch_generator = SpeechSegmentGenerator(
                feature_extraction,
                per_label=self.per_label, per_fold=self.per_fold,
                duration=self.duration)
            batches = batch_generator(protocol, subset=subset)
            batch = next(batches)
        except OSError as e:
            del batch_generator.data_
            batch_generator = SpeechSegmentGenerator(
                feature_extraction,
                per_label=self.per_label, per_fold=self.per_fold,
                duration=self.duration, fast=False)
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

            tloss_avg, closs_avg = 0., 0.
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

                if gpu:
                    fX_ = fX.data.cpu().numpy()
                else:
                    fX_ = fX.data.numpy()
                norms.append(np.linalg.norm(fX_, axis=1))

                # pre-compute pairwise distances
                distances = self.pdist(fX)

                if gpu:
                    distances_ = distances.data.cpu().numpy()
                else:
                    distances_ = distances.data.numpy()
                is_positive = pdist(batch['y'].reshape((-1, 1)), metric='chebyshev') < 1
                positive.append(distances_[np.where(is_positive)])
                negative.append(distances_[np.where(~is_positive)])

                # sample triplets
                if self.sampling == 'all':
                    anchors, positives, negatives = self.batch_all(
                        batch['y'], distances)

                elif self.sampling == 'hard':
                    anchors, positives, negatives = self.batch_hard(
                        batch['y'], distances)

                # compute triplet loss
                tlosses, deltas = self.triplet_loss(
                    distances, anchors, positives, negatives,
                    return_delta=True)
                tloss = torch.mean(tlosses)

                # compute wtf loss
                # closses = F.sigmoid(
                #     torch.norm(fX[anchors], 2, 1, keepdim=True) * deltas)
                closses = F.sigmoid(
                    F.softsign(deltas) * torch.norm(fX[anchors], 2, 1, keepdim=True))
                closs = torch.mean(closses)

                # log batch loss
                if gpu:
                    tloss_ = float(tloss.data.cpu().numpy())
                    closs_ = float(closs.data.cpu().numpy())
                else:
                    tloss_ = float(tloss.data.numpy())
                    closs_ = float(closs.data.numpy())
                tloss_avg += tloss_
                closs_avg += closs_
                writer.add_scalar(
                    'tloss/batch', tloss_,
                    global_step=i + epoch * batches_per_epoch)
                writer.add_scalar(
                    'closs/batch', closs_,
                    global_step=i + epoch * batches_per_epoch)

                loss = tloss + closs

                loss.backward()
                optimizer.step()

            tloss_avg /= batches_per_epoch
            writer.add_scalar('tloss', tloss_avg, global_step=epoch)

            closs_avg /= batches_per_epoch
            writer.add_scalar('closs', closs_avg, global_step=epoch)

            positive = np.hstack(positive)
            negative = np.hstack(negative)
            writer.add_histogram(
                'embedding/pairwise_distance/positive', positive,
                global_step=epoch, bins='auto')
            writer.add_histogram(
                'embedding/pairwise_distance/negative', negative,
                global_step=epoch, bins='auto')

            norms = np.hstack(norms)
            writer.add_histogram(
                'embedding/norm', norms,
                global_step=epoch, bins='auto')

            logging_callback.model = model
            logging_callback.optimizer = optimizer
            logging_callback.on_epoch_end(epoch)
