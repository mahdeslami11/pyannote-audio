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
  speaker_embedding train <config.yml> <dataset> <dataset_dir>
  speaker_embedding -h | --help
  speaker_embedding --version

Options:
  <config.yml>              Use this configuration file.
  <dataset>                 Use this dataset (e.g. "etape.train" for training)
  <dataset_dir>             Path to actual dataset material (e.g. '/Users/bredin/Corpora/etape')
  -h --help                 Show this screen.
  --version                 Show version.

"""

import yaml
import os.path
import numpy as np
from docopt import docopt

from pyannote.audio.callback import LoggingCallback
from pyannote.audio.features.yaafe import YaafeMFCC
from pyannote.audio.embedding.models import TripletLossBiLSTMSequenceEmbedding
from pyannote.audio.embedding.generator import TripletBatchGenerator
from etape import Etape

def train(dataset, dataset_dir, config_yml):

    # load configuration file
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # deduce workdir from path of configuration file
    workdir = os.path.dirname(config_yml)

    # this is where model weights are saved after each epoch
    log_dir = workdir + '/' + dataset

    # -- DATASET --
    dataset, subset = dataset.split('.')
    if dataset != 'etape':
        msg = '{dataset} dataset is not supported.'
        raise NotImplementedError(msg.format(dataset=dataset))

    protocol = Etape(dataset_dir)

    if subset == 'train':
        file_generator = protocol.train_iter()
    elif subset == 'dev':
        file_generator = protocol.dev_iter()
    else:
        msg = 'Training on {subset} subset is not allowed.'
        raise NotImplementedError(msg.format(subset=subset))

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
    embedding = TripletLossBiLSTMSequenceEmbedding(
        output_dim, lstm=lstm, dense=dense, bidirectional=bidirectional,
        space=space, margin=margin, optimizer=optimizer, log_dir=log_dir)

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

if __name__ == '__main__':

    arguments = docopt(__doc__, version='Speaker embedding')

    if arguments['train']:

        # arguments
        dataset = arguments['<dataset>']
        dataset_dir = arguments['<dataset_dir>']
        config_yml = arguments['<config.yml>']

        # train the model
        train(dataset, dataset_dir, config_yml)
