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

"""
Speaker embedding

Usage:
  speaker_embedding train <config.yml> <dataset> <medium_template>
  speaker_embedding tune <config.yml> <weights_dir> <dataset> <medium_template> <output_dir>
  speaker_embedding apply <config.yml> <weights.h5> <dataset> <medium_template> <output_dir>
  speaker_embedding compare <config.yml> <dataset> <medium_template> <output_dir>
  speaker_embedding -h | --help
  speaker_embedding --version

Options:
  <config.yml>              Use this configuration file.
  <dataset>                 Use this dataset (e.g. "Etape.SpeakerDiarization.TV.train" for training)
  <medium_template>         Path template to actual media files (e.g. '/Users/bredin/Corpora/etape/{uri}.wav')
  <weights.h5>              Path to pre-trained model weights. File
                            'architecture.yml' must live in the same directory.
  <output_dir>              Path where to save results.
  -h --help                 Show this screen.
  --version                 Show version.

"""

import matplotlib
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import yaml
import os.path
import numpy as np
from docopt import docopt

import pyannote.core
from pyannote.audio.callback import LoggingCallback
from pyannote.audio.features.yaafe import YaafeMFCC
from pyannote.audio.embedding.base import SequenceEmbedding
from pyannote.audio.embedding.models import TristouNet
from pyannote.audio.embedding.losses import TripletLoss
from pyannote.audio.embedding.generator import TripletBatchGenerator
from pyannote.audio.embedding.generator import LabeledSequencesBatchGenerator
from pyannote.database import get_database
from scipy.spatial.distance import pdist, squareform

from pyannote.metrics.plot.binary_classification import plot_det_curve, \
                                                        plot_distributions, \
                                                        plot_precision_recall_curve


def train(dataset, medium_template, config_yml):

    # load configuration file
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # deduce workdir from path of configuration file
    workdir = os.path.dirname(config_yml)

    # this is where model weights are saved after each epoch
    log_dir = workdir + '/' + dataset

    # -- DATASET --
    db, task, protocol, subset = dataset.split('.')
    database = get_database(db, medium_template=medium_template)
    protocol = database.get_protocol(task, protocol)

    if not hasattr(protocol, subset):
        raise NotImplementedError('')

    file_generator = getattr(protocol, subset)()

    # -- FEATURE EXTRACTION --
    # input sequence duration
    duration = config['feature_extraction']['duration']
    # MFCCs
    feature_extractor = YaafeMFCC(**config['feature_extraction']['mfcc'])
    # normalization
    normalize = config['feature_extraction']['normalize']

    # -- NETWORK STRUCTURE --
    # internal model structure
    output_dim = config['network']['output_dim']
    lstm = config['network']['lstm']
    pooling = config['network'].get('pooling', 'last')
    dense = config['network']['dense']
    # bi-directional
    bidirectional = config['network']['bidirectional']
    space = config['network']['space']

    # -- TRAINING --
    # batch size
    batch_size = config['training']['batch_size']
    # number of epochs
    nb_epoch = config['training']['nb_epoch']
    # optimizer
    optimizer = config['training']['optimizer']

    # -- TRIPLET LOSS --
    margin = config['training']['triplet_loss']['margin']
    per_fold = config['training']['triplet_loss']['per_fold']
    per_label = config['training']['triplet_loss']['per_label']
    overlap = config['training']['triplet_loss']['overlap']

    # embedding
    get_embedding = TristouNet(
        lstm=lstm, bidirectional=bidirectional, pooling=pooling,
        dense=dense, output_dim=output_dim, space=space)
    embedding = TripletLoss(
        get_embedding, margin=margin, optimizer=optimizer, log_dir=log_dir)

    # triplet generator for training
    batch_generator = TripletBatchGenerator(
        feature_extractor, file_generator, embedding,
        duration=duration, overlap=overlap, normalize=normalize,
        per_fold=per_fold, per_label=per_label, batch_size=batch_size)

    # log loss during training and keep track of best model
    log = [('train', 'loss')]
    callback = LoggingCallback(log_dir=log_dir, log=log,
                               get_model=embedding.get_embedding)

    # estimated number of triplets per epoch
    # (rounded to closest batch_size multiple)
    samples_per_epoch = per_label * (per_label - 1) * batch_generator.n_labels
    samples_per_epoch = samples_per_epoch - (samples_per_epoch % batch_size)

    # input shape (n_samples, n_features)
    input_shape = batch_generator.get_shape()

    embedding.fit(input_shape, batch_generator, samples_per_epoch, nb_epoch,
                  callbacks=[callback])




def generate_test(dataset, medium_template, config):

    # -- DATASET --
    dataset, subset = dataset.split('.')
    if dataset != 'etape':
        msg = '{dataset} dataset is not supported.'
        raise NotImplementedError(msg.format(dataset=dataset))

    # -- DATASET --
    db, task, protocol, subset = dataset.split('.')
    database = get_database(db, medium_template=medium_template)
    protocol = database.get_protocol(task, protocol)

    if not hasattr(protocol, subset):
        raise NotImplementedError('')

    file_generator = getattr(protocol, subset)()

    # -- FEATURE EXTRACTION --
    # input sequence duration
    duration = config['feature_extraction']['duration']
    # MFCCs
    feature_extractor = YaafeMFCC(**config['feature_extraction']['mfcc'])
    # normalization
    normalize = config['feature_extraction']['normalize']

    overlap = config['testing']['overlap']
    per_label = config['testing']['per_label']
    batch_size = config['testing']['batch_size']

    batch_generator = LabeledSequencesBatchGenerator(
        feature_extractor,
        duration=duration,
        normalize=normalize,
        step=(1 - overlap) * duration,
        batch_size=-1)

    X, y = [], []
    for sequences, labels in batch_generator(file_generator):
        X.append(sequences)
        y.append(labels)
    X = np.vstack(X)
    y = np.hstack(y)

    unique, y, counts = np.unique(y, return_inverse=True, return_counts=True)

    # randomly (but deterministically) select 'per_label' samples from each class
    # only compute (positive vs. negative distances for those samples)
    # this should ensure all speakers have the same weights
    np.random.seed(1337)

    # indices contains the list of indices of all sequences
    # to be used for later triplet selection
    indices = []

    n_labels = len(unique)
    for label in range(n_labels):

        # randomly choose 'per_label' sequences
        # from the set of available sequences
        i = np.random.choice(
            np.where(y == label)[0],
            size=per_label,
            replace=True)

        # append indices of selected sequences
        indices.append(i)

    # turn indices into a 1-dimensional numpy array.
    indices = np.hstack(indices)

    # selected sequences
    X = X[indices]

    # their pairwise similarity
    y_true = pdist(y[indices, np.newaxis], metric='chebyshev') < 1

    return X, y_true

def tune(dataset, medium_template, config_yml, weights_dir, output_dir):

    # load configuration file
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    X, y_true = generate_test(dataset, medium_template, config)

    # this is where model architecture was saved
    architecture_yml = os.path.dirname(weights_dir) + '/architecture.yml'

    output_dir = output_dir + '/' + dataset

    try:
        os.makedirs(output_dir)
    except Exception as e:
        pass

    nb_epoch = config['training']['nb_epoch']
    WEIGHTS_H5 = weights_dir + '/{epoch:04d}.h5'

    LINE = '{epoch:04d} {eer:.6f}\n'
    PATH = output_dir + '/eer.txt'
    with open(PATH.format(dataset=dataset), 'w') as fp:

        for epoch in range(nb_epoch):

            # load model for this epoch
            weights_h5 = WEIGHTS_H5.format(epoch=epoch)
            if not os.path.isfile(weights_h5):
                continue

            sequence_embedding = SequenceEmbedding.from_disk(
                architecture_yml, weights_h5)

            # pairwise euclidean distances between embeddings
            batch_size = config['testing']['batch_size']
            x = sequence_embedding.transform(X, batch_size=batch_size, verbose=0)
            distances = pdist(x, metric='euclidean')
            PATH = output_dir + '/plot.{epoch:04d}'
            eer = plot_det_curve(y_true, -distances, PATH.format(epoch=epoch))

            msg = 'Epoch #{epoch:04d} | EER = {eer:.2f}%'
            print(msg.format(epoch=epoch, eer=100 * eer))

            fp.write(LINE.format(epoch=epoch, eer=eer))
            fp.flush()

            # save distribution plots after each epoch
            space = config['network']['space']
            xlim = (0, 2 if space == 'sphere' else np.sqrt(2.))
            plot_distributions(y_true, distances, PATH.format(epoch=epoch),
                               xlim=xlim, ymax=3, nbins=100)

def compare(dataset, medium_template, config_yml, output_dir):

    import itertools
    from pyannote.algorithms.stats.gaussian import Gaussian

    # load configuration file
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    X, y_true = generate_test(dataset, medium_template, config)

    n_sequences = X.shape[0]

    gaussians = []
    for x in X:
        g = Gaussian(covariance_type='diag').fit(x)
        gaussians.append(g)

    bic = np.zeros((n_sequences, n_sequences), dtype=np.float)
    for i, j in itertools.combinations(range(n_sequences), 2):
        bic[i, j], _ = gaussians[i].bic(gaussians[j], penalty_coef=0.)

    distances = squareform(bic, checks=False)

    # -- distances distributions
    plot_distributions(y_true, distances, output_dir + '/plot.bic', xlim=(0, 20), ymax=0.5, nbins=100)

    # -- precision / recall curve
    auc = plot_precision_recall_curve(y_true, -distances, output_dir + '/plot.bic')
    msg = 'BIC | AUC = {auc:.2f}%'
    print(msg.format(auc=100 * auc))

    # -- det curve
    eer = plot_det_curve(y_true, -distances, output_dir + '/plot.bic')
    msg = 'BIC | EER = {eer:.2f}%'
    print(msg.format(eer=100 * eer))

    divergence = np.zeros((n_sequences, n_sequences), dtype=np.float)
    for i, j in itertools.combinations(range(n_sequences), 2):
        divergence[i, j] = gaussians[i].divergence(gaussians[j])

    distances = squareform(divergence, checks=False)

    # -- distances distributions
    plot_distributions(y_true, distances, output_dir + '/plot.divergence', xlim=(0, 20), ymax=0.5, nbins=100)

    # -- precision / recall curve
    auc = plot_precision_recall_curve(y_true, -distances, output_dir + '/plot.divergence')
    msg = 'Divergence | AUC = {auc:.2f}%'
    print(msg.format(auc=100 * auc))

    # -- det curve
    eer = plot_det_curve(y_true, -distances, output_dir + '/plot.divergence')
    msg = 'Divergence | EER = {eer:.2f}%'
    print(msg.format(eer=100 * eer))



def test(dataset, medium_template, config_yml, weights_h5, output_dir):

    # load configuration file
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    X, y_true = generate_test(dataset, medium_template, config)

    # this is where model architecture was saved
    architecture_yml = os.path.dirname(os.path.dirname(weights_h5)) + '/architecture.yml'

    sequence_embedding = SequenceEmbedding.from_disk(
        architecture_yml, weights_h5)

    # pairwise euclidean distances between embeddings
    batch_size = config['testing']['batch_size']
    x = sequence_embedding.transform(X, batch_size=batch_size, verbose=0)
    distances = pdist(x, metric='euclidean')

    # -- distances distributions
    space = config['network']['space']
    xlim = (0, 2 if space == 'sphere' else np.sqrt(2.))
    plot_distributions(y_true, distances, output_dir + '/plot', xlim=xlim, ymax=3, nbins=100)

    # -- precision / recall curve
    auc = plot_precision_recall_curve(y_true, -distances, output_dir + '/plot')
    msg = 'AUC = {auc:.2f}%'
    print(msg.format(auc=100 * auc))

    # -- det curve
    eer = plot_det_curve(y_true, -distances, output_dir + '/plot')
    msg = 'EER = {eer:.2f}%'
    print(msg.format(eer=100 * eer))

if __name__ == '__main__':

    arguments = docopt(__doc__, version='Speaker embedding')

    if arguments['train']:

        # arguments
        dataset = arguments['<dataset>']
        medium_template = arguments['<medium_template>']
        config_yml = arguments['<config.yml>']

        # train the model
        train(dataset, medium_template, config_yml)

    if arguments['apply']:

        # arguments
        config_yml = arguments['<config.yml>']
        weights_h5 = arguments['<weights.h5>']
        dataset = arguments['<dataset>']
        medium_template = arguments['<medium_template>']
        output_dir = arguments['<output_dir>']

        test(dataset, medium_template, config_yml, weights_h5, output_dir)

    if arguments['tune']:

        # arguments
        config_yml = arguments['<config.yml>']
        weights_dir = arguments['<weights_dir>']
        dataset = arguments['<dataset>']
        medium_template = arguments['<medium_template>']
        output_dir = arguments['<output_dir>']

        tune(dataset, medium_template, config_yml, weights_dir, output_dir)

    if arguments['compare']:

        # arguments
        config_yml = arguments['<config.yml>']
        dataset = arguments['<dataset>']
        medium_template = arguments['<medium_template>']
        output_dir = arguments['<output_dir>']

        compare(dataset, medium_template, config_yml, output_dir)
