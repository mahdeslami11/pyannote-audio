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
import torch
import torch.nn as nn
import torch.nn.functional as F
from .triplet_loss import TripletLoss


class Centroids(nn.Module):
    """Centroids embeddings

    Parameters
    ----------
    n_dimensions : int
        Embedding dimension
    n_classes : int
        Number of classes.
    """

    def __init__(self, n_dimensions, n_classes):
        super(Centroids, self).__init__()
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.embeddings = nn.Embedding(n_classes, n_dimensions)

    def forward(self, indices):
        return self.embeddings(indices)

class CentroidLoss(TripletLoss):

    CLASSES_TXT = '{log_dir}/classes.txt'
    CENTROIDS_PT = '{log_dir}/weights/{epoch:04d}.centroids.pt'

    def extra_init(self, model, device, checkpoint=None,
                   labels=None):
        """Initialize centroids

        Parameters
        ----------
        model : `torch.nn.Module`
            Embedding model.
        device : `torch.device`
            Device used by model parameters.
        labels : `list` of `str`, optional
            List of classes.
        checkpoint : `pyannote.audio.train.checkpoint.Checkpoint`, optional
            Checkpoint.

        Returns
        -------
        parameters : list
            Centroids parameters
        """

        # dimension of embedding space
        n_dimensions = model.output_dim

        # number of labels in training set
        n_classes = len(labels)

        self.triggers_ = torch.LongTensor(range(n_classes))
        self.triggers_ = self.triggers_.to(device)
        self.centroids_ = Centroids(n_dimensions, n_classes)
        self.centroids_ = self.centroids_.to(device)

        # TODO. make sure classes_txt does not exist already
        # or, if it does, that it is coherent with "labels"
        log_dir = checkpoint.log_dir
        classes_txt = self.CLASSES_TXT.format(log_dir=log_dir)
        with open(classes_txt, mode='w') as fp:
            for label in labels:
                fp.write(f'{label}\n')

        return self.centroids_.parameters()

    def on_epoch_end(self, iteration, checkpoint, **kwargs):
        """Save centroids to disk at the end of current epoch

        Parameters
        ----------
        iteration : `int`
            Current epoch.
        checkpoint : `pyannote.audio.train.checkpoint.Checkpoint`
            Checkpoint.
        """

        centroids_pt = self.CENTROIDS_PT.format(
            log_dir=checkpoint.log_dir, epoch=iteration)
        torch.save(self.centroids_.state_dict(), centroids_pt)

    def extra_restart(self, checkpoint, restart):
        """Load centroids weights

        Parameters
        ----------
        checkpoint : `pyannote.audio.train.checkpoint.Checkpoint`
            Checkpoint.
        restart : `int`
            Epoch used for warm restart.
        """

        centroids_pt = self.CENTROIDS_PT.format(
            log_dir=checkpoint.log_dir, epoch=restart)
        centroids_state = torch.load(centroids_pt,
            map_location=lambda storage, loc: storage)
        self.centroids_.load_state_dict(centroids_state)

    def cdist(self, fX):
        """Compute distances to centroids

        Parameters
        ----------
        fX : (n, d) torch.Tensor
            Embeddings.

        Returns
        -------
        distances : (n, n_classes) torch.Tensor
            Distance matrix
        """

        fC = self.centroids_(self.triggers_)

        if self.metric in ('cosine', 'angular'):

            X_norm = torch.norm(fX, 2, 1, keepdim=True)
            fX_ = fX / X_norm

            C_norm = torch.norm(fC, 2, 1, keepdim=True)
            fC_ = fC / C_norm

            cosine = torch.mm(fX_, fC_.t())

            if self.metric == 'angular':
                return torch.acos(torch.clamp(cosine, 1e-6 - 1, 1 - 1e-6))

            return 1. - cosine

        if self.metric == 'euclidean':
            raise NotImplementedError('')
            # return F.pairwise_distance(fX, fC, p=2, eps=1e-6)

    def batch_all(self, y, distances):
        """Build all possible (sample, centroid, other centroid)

        Parameters
        ----------
        y : list
            Sequence labels.
        distances : (n, n_classes) torch.Tensor
            Distance matrix

        Returns
        -------
        samples, positives, negatives : list of int
            Triplets indices.
        """

        samples, positives, negatives = [], [], []

        for sample, positive_centroid in enumerate(y):
            for negative_centroid in range(self.centroids_.n_classes):
                if negative_centroid == positive_centroid:
                    continue
                samples.append(sample)
                positives.append(positive_centroid)
                negatives.append(negative_centroid)

        return samples, positives, negatives

    def centroid_loss(self, distances, samples, positives, negatives):

        delta = distances[samples, positives] - distances[samples, negatives]

        # clamp triplet loss
        if self.clamp == 'positive':
            loss = torch.clamp(delta + self.margin_, min=0)

        elif self.clamp == 'softmargin':
            loss = torch.log1p(torch.exp(delta))

        elif self.clamp == 'sigmoid':
            # TODO. tune this "10" hyperparameter
            # TODO. log-sigmoid
            loss = F.sigmoid(10 * (delta + self.margin_))

        return loss

    def batch_loss(self, batch, model, device, writer=None, **kwargs):
        """Compute loss for current `batch`

        Parameters
        ----------
        batch : `dict`
            ['X'] (`numpy.ndarray`)
            ['y'] (`numpy.ndarray`)
        model : `torch.nn.Module`
            Model currently being trained.
        device : `torch.device`
            Device used by model parameters.
        writer : `tensorboardX.SummaryWriter`, optional
            Tensorboard writer.

        Returns
        -------
        loss : `torch.Tensor`
            Negative log likelihood loss
        """

        fX = self.forward(batch, model, device)

        batch['fX'] = fX
        batch = self.aggregate(batch)

        fX = batch['fX']
        y = batch['y']

        # pre-compute distances to centroids
        distances = self.cdist(fX)

        # sample (sample, centroid, other centroid) triplets
        triplets = getattr(self, 'batch_{0}'.format(self.sampling))
        samples, positives, negatives = triplets(y, distances)

        # compute loss for each (sample, centroid, other centroid) triplet
        losses = self.centroid_loss(distances, samples, positives, negatives)

        # average over all triplets
        return torch.mean(losses)
