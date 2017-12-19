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

"""
Speaker embedding

Usage:
  pyannote-speaker-embedding train [--database=<db.yml> --subset=<subset>] <experiment_dir> <database.task.protocol>
  pyannote-speaker-embedding validate [--database=<db.yml> --subset=<subset> --from=<epoch> --to=<epoch> --every=<epoch>] <train_dir> <database.task.protocol>
  pyannote-speaker-embedding apply [--database=<db.yml> --subset=<subset>] <tune_dir> <database.task.protocol>
  pyannote-speaker-embedding -h | --help
  pyannote-speaker-embedding --version

Common options:
  <database.task.protocol>   Experimental protocol (e.g. "Etape.SpeakerDiarization.TV")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset (train|developement|test).
                             In "train" mode, default subset is "train".
                             In "validate" mode, defaults to "development".
                             In "tune" mode, defaults to "development".
                             In "apply" mode, defaults to "test".
  --from=<epoch>             Start validating/tuning at epoch <epoch>.
                             Defaults to first available epoch.
  --to=<epoch>               End validation/tuning at epoch <epoch>.
                             In "validate" mode, defaults to never stop.
                             In "tune" mode, defaults to last available epoch at launch time.

"train" mode:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.

"validation" mode:
  --every=<epoch>            Validate model every <epoch> epochs [default: 1].
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).

"apply" mode:
  <tune_dir>                 Path to the directory containing optimal
                             hyper-parameters (i.e. the output of "tune" mode).
  -h --help                  Show this screen.
  --version                  Show version.

Database configuration file <db.yml>:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.database.util.FileFinder` docstring for more
    information on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the architecture of the neural
    network used for sequence labeling (0 vs. 1, non-speech vs. speech), the
    feature extraction process (e.g. MFCCs) and the sequence generator used for
    both training and testing.

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: Precomputed
       params:
          root_dir: /path/to/precomputed/features
          use_memmap: True

    architecture:
       name: ClopiNet
       params:
          rnn: GRU
          recurrent: [64, 64, 64]
          bidirectional: True
          linear: []
          weighted: True

    sequences:
       duration: 3.2
       per_label: 3
    ...................................................................

"train" mode:
    This will create the following directory that contains the pre-trained
    neural network weights after each epoch:

        <experiment_dir>/train/<database.task.protocol>.<subset>

    This means that the network was trained on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <train_dir> in the subsequent "validate" mode.

"validate" mode:
    In parallel to training, one should validate the performance of the model
    epoch after epoch, using "validate" mode. This will create a bunch of files
    in the following directory:

        <train_dir>/validate/<database.task.protocol>

    This means that the network was validated on the <database.task.protocol>
    protocol. By default, validation is done on the "development" subset:
    "developement.DetectionErrorRate.{txt|png|eps}" files are created and
    updated continuously, epoch after epoch. This directory is called
    <validate_dir> in the subsequent "tune" mode.

    In practice, for each epoch, "validate" mode will report the detection
    error rate obtained by applying (possibly not optimal) onset/offset
    thresholds of 0.5.

"apply" mode
    Finally, one can apply speech activity detection using "apply" mode.
    This will create the following files that contains the hard (mdtm) and
    soft (h5) outputs of speech activity detection, based on the set of hyper-
    parameters obtain with "tune" mode:

        <tune_dir>/apply/<database.task.protocol>.<subset>.mdtm
        <tune_dir>/apply/{database}/{uri}.h5

    This means that file whose unique resource identifier is {uri} has been
    processed.
"""

import io
import yaml
from os.path import expanduser

from docopt import docopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyannote.database.util import get_annotated
from pyannote.database import get_protocol
from pyannote.database import get_unique_identifier

from pyannote.audio.util import mkdir_p

from pyannote.audio.features.utils import Precomputed
from pyannote.audio.embedding.approaches_pytorch.triplet_loss import TripletLoss

from .base import Application


class SpeakerEmbeddingPytorch(Application):

    # created by "train" mode
    TRAIN_DIR = '{experiment_dir}/train/{protocol}.{subset}'
    APPLY_DIR = '{tune_dir}/apply'

    def __init__(self, experiment_dir, db_yml=None):

        super(SpeakerEmbeddingPytorch, self).__init__(
            experiment_dir, db_yml=db_yml, backend='pytorch')

        # architecture
        architecture_name = self.config_['architecture']['name']
        models = __import__('pyannote.audio.embedding.models_pytorch',
                            fromlist=[architecture_name])
        Architecture = getattr(models, architecture_name)
        self.model_ = Architecture(
            int(self.feature_extraction_.dimension()),
            **self.config_['architecture'].get('params', {}))

        approach_name = self.config_['approach']['name']
        approaches = __import__('pyannote.audio.embedding.approaches_pytorch',
                                fromlist=[approach_name])
        Approach = getattr(approaches, approach_name)
        self.approach_ = Approach(
            **self.config_['approach'].get('params', {}))

    def train(self, protocol_name, subset='train'):

        train_dir = self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        self.approach_.fit(self.model_, self.feature_extraction_, protocol,
                           train_dir, subset=subset, n_epochs=1000)

    def validate_init(self, protocol_name, subset='development'):
        from pyannote.audio.generators.speaker import SpeechTurnGenerator
        from pyannote.audio.embedding.utils import pdist

        import numpy as np
        np.random.seed(1337)

        batch_generator = SpeechTurnGenerator(
            self.feature_extraction_, per_label=10,
            duration=self.approach_.duration)

        protocol = get_protocol(
            protocol_name, progress=False, preprocessors=self.preprocessors_)

        batch = next(batch_generator(protocol, subset=subset))

        _, y = np.unique(batch['y'], return_inverse=True)
        y = pdist(y.reshape((-1, 1)), metric='chebyshev') < 1

        return {'X': batch['X'], 'y': y}

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):

        import numpy as np
        import torch
        from torch.autograd import Variable
        from pyannote.metrics.binary_classification import det_curve
        from pyannote.audio.embedding.utils import pdist

        model = self.load_model(epoch)
        model.eval()

        X = Variable(torch.from_numpy(
            np.array(np.rollaxis(validation_data['X'], 0, 2),
                     dtype=np.float32)))
        fX, _ = model(X)

        y_pred = pdist(fX.data.numpy(), metric=self.approach_.metric)

        _, _, _, eer = det_curve(validation_data['y'], y_pred,
                                 distances=True)

        metrics = {}
        metrics['EER.1seq'] = {'minimize': True, 'value': eer}
        return metrics

    # def apply(self, protocol_name, subset='test'):
    #
    #     apply_dir = self.APPLY_DIR.format(tune_dir=self.tune_dir_)
    #
    #     mkdir_p(apply_dir)
    #
    #     # load tuning results
    #     tune_yml = self.TUNE_YML.format(tune_dir=self.tune_dir_)
    #     with io.open(tune_yml, 'r') as fp:
    #         self.tune_ = yaml.load(fp)
    #
    #     # load model for epoch 'epoch'
    #     epoch = self.tune_['epoch']
    #     model = self.load_model(epoch)
    #
    #     # initialize sequence labeling
    #     duration = self.config_['sequences']['duration']
    #     step = self.config_['sequences']['step']
    #     sequence_labeling = SequenceLabeling(
    #         model, self.feature_extraction_, duration,
    #         step=step)
    #
    #     # initialize protocol
    #     protocol = get_protocol(protocol_name, progress=True,
    #                             preprocessors=self.preprocessors_)
    #
    #     file_generator = getattr(protocol, subset)
    #     processed_uris = set()
    #
    #     for i, item in enumerate(file_generator()):
    #
    #        # corner case when the same file is iterated several times
    #         uri = get_unique_identifier(item)
    #         if uri in processed_uris:
    #             continue
    #
    #         predictions = sequence_labeling.apply(item)
    #
    #         if i == 0:
    #             # create metadata file at root that contains
    #             # sliding window and dimension information
    #             precomputed = Precomputed(
    #                 root_dir=apply_dir,
    #                 sliding_window=predictions.sliding_window,
    #                 dimension=2)
    #
    #         precomputed.dump(item, predictions)
    #         processed_uris.add(uri)
    #
    #     # initialize binarizer
    #     onset = self.tune_['onset']
    #     offset = self.tune_['offset']
    #     binarize = Binarize(onset=onset, offset=offset)
    #
    #     precomputed = Precomputed(root_dir=apply_dir)
    #
    #     writer = MDTMParser()
    #     path = self.HARD_MDTM.format(apply_dir=apply_dir,
    #                                  protocol=protocol_name,
    #                                  subset=subset)
    #     with io.open(path, mode='w') as gp:
    #
    #         processed_uris = set()
    #
    #         for item in file_generator():
    #
    #            # corner case when the same file is iterated several times
    #             uri = get_unique_identifier(item)
    #             if uri in processed_uris:
    #                 continue
    #
    #             predictions = precomputed(item)
    #             segmentation = binarize.apply(predictions, dimension=1)
    #             writer.write(segmentation.to_annotation(),
    #                          f=gp, uri=item['uri'], modality='speaker')
    #
    #             processed_uris.add(uri)

def main():

    arguments = docopt(__doc__, version='Speaker embedding')

    db_yml = expanduser(arguments['--database'])
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']

        if subset is None:
            subset = 'train'

        application = SpeakerEmbeddingPytorch(experiment_dir, db_yml=db_yml)
        application.train(protocol_name, subset=subset)

    if arguments['validate']:
        train_dir = arguments['<train_dir>']

        if subset is None:
            subset = 'development'

        # start validating at this epoch (defaults to None)
        start = arguments['--from']
        if start is not None:
            start = int(start)

        # stop validating at this epoch (defaults to None)
        end = arguments['--to']
        if end is not None:
            end = int(end)

        # validate every that many epochs (defaults to 1)
        every = int(arguments['--every'])

        application = SpeakerEmbeddingPytorch.from_train_dir(
            train_dir, db_yml=db_yml)
        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every)
    #
    # if arguments['tune']:
    #     train_dir = arguments['<train_dir>']
    #
    #     if subset is None:
    #         subset = 'development'
    #
    #     # start tuning at this epoch (defaults to None)
    #     start = arguments['--from']
    #     if start is not None:
    #         start = int(start)
    #
    #     # stop tuning at this epoch (defaults to None)
    #     end = arguments['--to']
    #     if end is not None:
    #         end = int(end)
    #
    #     at = arguments['--at']
    #     if at is not None:
    #         at = int(at)
    #
    #     application = SpeakerEmbeddingPytorch.from_train_dir(
    #         train_dir, db_yml=db_yml)
    #     application.tune(protocol_name, subset=subset,
    #                      start=start, end=end, at=at)
    #
    # if arguments['apply']:
    #     tune_dir = arguments['<tune_dir>']
    #
    #     if subset is None:
    #         subset = 'test'
    #
    #     application = SpeakerEmbeddingPytorch.from_tune_dir(
    #         tune_dir, db_yml=db_yml)
    #     application.apply(protocol_name, subset=subset)
