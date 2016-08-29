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
Speech activity detection

Usage:
  speech_activity_detection train <config.yml> <dataset> <dataset_dir>
  speech_activity_detection apply <config.yml> <weights.h5> <dataset> <dataset_dir> <output_dir>
  speech_activity_detection -h | --help
  speech_activity_detection --version

Options:
  <config.yml>              Use this configuration file.
  <dataset>                 Use this dataset (e.g. "etape.train" for training)
  <dataset_dir>             Path to actual dataset material (e.g. '/Users/bredin/Corpora/etape')
  <weights.h5>              Path to pre-trained model weights. File
                            'architecture.yml' must live in the same directory.
  <output_dir>              Path where to save results.
  -h --help                 Show this screen.
  --version                 Show version.

"""

import yaml
import os.path
import numpy as np
from docopt import docopt

from pyannote.audio.callback import LoggingCallback
from pyannote.audio.features.yaafe import YaafeMFCC
from pyannote.audio.labeling.models import SequenceLabeling, BiLSTMSequenceLabeling
from pyannote.audio.generators.speech import SpeechActivityDetectionBatchGenerator

from pyannote.audio.labeling.aggregation import SequenceLabelingAggregation
from pyannote.audio.signal import Binarize
from pyannote.metrics.detection import DetectionErrorRate, DetectionAccuracy, \
                                       DetectionPrecision, DetectionRecall
from pyannote.metrics import f_measure
from pyannote.core.json import dump_to

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
    lstm = config['network']['lstm']
    dense = config['network']['dense']
    # bi-directional
    bidirectional = config['network']['bidirectional']

    # -- TRAINING --
    # number training set hours (speech + non speech) to use in each epoch
    # FIXME -- update ETAPE so that we can query this information directly
    hours_per_epoch = config['training']['hours_per_epoch']
    # overlap ratio between each window
    overlap = config['training']['overlap']
    # batch size
    batch_size = config['training']['batch_size']
    # number of epochs
    nb_epoch = config['training']['nb_epoch']
    # optimizer
    optimizer = config['training']['optimizer']

    # labeling
    output_dim = 2
    labeling = BiLSTMSequenceLabeling(
        output_dim,
        lstm=lstm, dense=dense, bidirectional=bidirectional,
        optimizer=optimizer, log_dir=log_dir)

    # segment generator for training
    step = duration * (1. - overlap)
    batch_generator = SpeechActivityDetectionBatchGenerator(
        feature_extractor, duration=duration, normalize=normalize,
        step=step, batch_size=batch_size)

    # log loss and accuracy during training and
    # keep track of best models for both metrics
    log = [('train', 'loss'), ('train', 'accuracy')]
    callback = LoggingCallback(log_dir=log_dir, log=log)

    # number of samples per epoch + round it to closest batch
    samples_per_epoch = batch_size * int(np.ceil((3600 * hours_per_epoch / step) / batch_size))

    # input shape (n_samples, n_features)
    input_shape = batch_generator.get_shape()

    labeling.fit(input_shape, batch_generator(file_generator, infinite=True),
                 samples_per_epoch, nb_epoch, callbacks=[callback])


def test(dataset, dataset_dir, config_yml, weights_h5, output_dir):

    # load configuration file
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    # this is where model architecture was saved
    architecture_yml = os.path.dirname(os.path.dirname(weights_h5)) + '/architecture.yml'

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
    elif subset == 'test':
        file_generator = protocol.test_iter()
    else:
        msg = 'Testing on {subset} subset is not supported.'
        raise NotImplementedError(msg.format(subset=subset))

    # -- FEATURE EXTRACTION --
    # input sequence duration
    duration = config['feature_extraction']['duration']
    # MFCCs
    feature_extractor = YaafeMFCC(**config['feature_extraction']['mfcc'])
    # normalization
    normalize = config['feature_extraction']['normalize']

    # -- TESTING --
    # overlap ratio between each window
    overlap = config['testing']['overlap']
    step = duration * (1. - overlap)

    # prediction smoothing
    onset = config['testing']['binarize']['onset']
    offset = config['testing']['binarize']['offset']
    binarizer = Binarize(onset=0.5, offset=0.5)

    sequence_labeling = SequenceLabeling.from_disk(
        architecture_yml, weights_h5)

    aggregation = SequenceLabelingAggregation(
        sequence_labeling, feature_extractor, normalize=normalize,
        duration=duration, step=step)

    collar = 0.500
    error_rate = DetectionErrorRate(collar=collar)
    accuracy = DetectionAccuracy(collar=collar)
    precision = DetectionPrecision(collar=collar)
    recall = DetectionRecall(collar=collar)

    LINE = '{uri} {e:.3f} {a:.3f} {p:.3f} {r:.3f} {f:.3f}\n'

    PATH = '{output_dir}/eval.{dataset}.{subset}.txt'
    path = PATH.format(output_dir=output_dir, dataset=dataset, subset=subset)

    with open(path, 'w') as fp:

        header = '# uri error accuracy precision recall f_measure\n'
        fp.write(header)
        fp.flush()

        for wav, uem, reference in file_generator:

            uri = os.path.splitext(os.path.basename(wav))[0]

            predictions = aggregation.apply(wav)
            hypothesis = binarizer.apply(predictions, dimension=1)

            e = error_rate(reference, hypothesis, uem=uem)
            a = accuracy(reference, hypothesis, uem=uem)
            p = precision(reference, hypothesis, uem=uem)
            r = recall(reference, hypothesis, uem=uem)
            f = f_measure(p, r)

            line = LINE.format(uri=uri, e=e, a=a, p=p, r=r, f=f)
            fp.write(line)
            fp.flush()

            PATH = '{output_dir}/{uri}.json'
            path = PATH.format(output_dir=output_dir, uri=uri)
            dump_to(hypothesis, path)

        # average on whole corpus
        uri = '{dataset}.{subset}'.format(dataset=dataset, subset=subset)
        e = abs(error_rate)
        a = abs(accuracy)
        p = abs(precision)
        r = abs(recall)
        f = f_measure(p, r)
        line = LINE.format(uri=uri, e=e, a=a, p=p, r=r, f=f)
        fp.write(line)
        fp.flush()


if __name__ == '__main__':

    arguments = docopt(__doc__, version='Speech activity detection')

    if arguments['train']:

        # arguments
        dataset = arguments['<dataset>']
        dataset_dir = arguments['<dataset_dir>']
        config_yml = arguments['<config.yml>']

        # train the model
        train(dataset, dataset_dir, config_yml)

    if arguments['apply']:

        # arguments
        config_yml = arguments['<config.yml>']
        weights_h5 = arguments['<weights.h5>']
        dataset = arguments['<dataset>']
        dataset_dir = arguments['<dataset_dir>']
        output_dir = arguments['<output_dir>']

        test(dataset, dataset_dir, config_yml, weights_h5, output_dir)
