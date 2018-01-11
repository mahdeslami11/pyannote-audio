#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017-2018 CNRS

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
  pyannote-speaker-embedding validate [--database=<db.yml> --subset=<subset> --from=<epoch> --to=<epoch> --every=<epoch>] <train_dir> <database.task.protocol>
  pyannote-speaker-embedding apply [--database=<db.yml> --step=<step> --internal] <validate.txt> <database.task.protocol> <output_dir>
  pyannote-speaker-embedding train [--database=<db.yml> --subset=<subset> --gpu] <experiment_dir> <database.task.protocol>
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
  --gpu                      Run on GPUs. Defaults to using CPUs.

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
          root_dir: /vol/work1/bredin/feature_extraction/mfcc

    architecture:
       name: ClopiNet
       params:
         rnn: LSTM
         recurrent: [64, 64, 64]
         bidirectional: True
         linear: []
         weighted: False

    approach:
       name: TripletLoss
       params:
         metric: cosine
         margin: 0.1
         clamp: positive
         duration: 3.2
         sampling: all
         per_fold: 100
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
    Use the "validate" mode to run validation in parallel to training.
    "validate" mode will watch the <train_dir> directory, and run validation
    experiments every time a new epoch has ended. This will create the
    following directory that contains validation results:

        <train_dir>/validate/<database.task.protocol>

    You can run multiple "validate" in parallel (e.g. for every subset,
    protocol, task, or database).
"""

import io
import yaml
from os.path import expanduser, dirname, basename

from docopt import docopt
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyannote.database.util import get_annotated
from pyannote.database import get_protocol
from pyannote.database import get_unique_identifier

from pyannote.audio.util import mkdir_p

from pyannote.audio.features.utils import Precomputed
from pyannote.audio.embedding.extraction import SequenceEmbedding

from .base import Application
from sortedcontainers import SortedDict


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

    def train(self, protocol_name, subset='train', gpu=False):

        train_dir = self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        self.approach_.fit(self.model_, self.feature_extraction_, protocol,
                           train_dir, subset=subset, n_epochs=1000, gpu=gpu)

    def validate_init(self, protocol_name, subset='development'):

        task = protocol_name.split('.')[1]
        if task == 'SpeakerVerification':
            return self._validate_init_verification(protocol_name,
                                                    subset=subset)

        return self._validate_init_default(protocol_name, subset=subset)

    def _validate_init_verification(self, protocol_name, subset='development'):
        return {}

    def _validate_init_default(self, protocol_name, subset='development'):

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

        task = protocol_name.split('.')[1]
        if task == 'SpeakerVerification':
            return self._validate_epoch_verification(
                epoch, protocol_name, subset=subset,
                validation_data=validation_data)

        return self._validate_epoch_default(epoch, protocol_name, subset=subset,
                                           validation_data=validation_data)

    def _validate_epoch_verification(self, epoch, protocol_name,
                                     subset='development',
                                     validation_data=None):

        from pyannote.metrics.binary_classification import det_curve
        from pyannote.audio.embedding.utils import cdist

        # load current model
        model = self.load_model(epoch)
        model.eval()

        duration = self.approach_.duration
        step = .25 * duration

        protocol = get_protocol(
        protocol_name, progress=False, preprocessors=self.preprocessors_)

        # initialize embedding extraction
        #batch_size = self.approach_.batch_size
        batch_size = 32

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        try:
            # use internal representation when available
            internal = True
            sequence_embedding = SequenceEmbedding(
                model, self.feature_extraction_, duration,
                step=step, batch_size=batch_size,
                internal=internal)

        except ValueError as e:
            # else use final representation
            internal = False
            sequence_embedding = SequenceEmbedding(
                model, self.feature_extraction_, duration,
                step=step, batch_size=batch_size,
                internal=internal)

        metrics = {}
        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        enrolment_models, enrolment_khashes = {}, {}
        enrolments = getattr(protocol, '{0}_enrolment'.format(subset))()
        for i, enrolment in enumerate(enrolments):
            model_id = enrolment['model_id']
            embedding = sequence_embedding.apply(enrolment)
            data = embedding.crop(enrolment['enrol_with'],
                                  mode='center', return_data=True)
            enrolment_models[model_id] = np.mean(data, axis=0, keepdims=True)

            # in some specific speaker verification protocols,
            # enrolment data may be  used later as trial data.
            # therefore, we cache information about enrolment data
            # to speed things up by reusing the enrolment as trial
            h = hash((get_unique_identifier(enrolment),
                      tuple(enrolment['enrol_with'])))
            enrolment_khashes[h] = model_id

        trial_models = {}
        trials = getattr(protocol, '{0}_trial'.format(subset))()
        y_true, y_pred = [], []
        for i, trial in enumerate(trials):
            model_id = trial['model_id']

            h = hash((get_unique_identifier(trial),
                      tuple(trial['try_with'])))

            # re-use enrolment model whenever possible
            if h in enrolment_khashes:
                model = enrolment_models[enrolment_khashes[h]]

            # re-use trial model whenever possible
            elif h in trial_models:
                model = trial_models[h]

            else:
                embedding = sequence_embedding.apply(trial)
                data = embedding.crop(trial['try_with'],
                                      mode='center', return_data=True)
                model = np.mean(data, axis=0, keepdims=True)
                # cache trial model for later re-use
                trial_models[h] = model

            distance = cdist(enrolment_models[model_id], model,
                             metric=self.approach_.metric)[0, 0]
            y_pred.append(distance)
            y_true.append(trial['reference'])

        _, _, _, eer = det_curve(np.array(y_true), np.array(y_pred),
                                 distances=True)
        metrics['EER.internal' if internal else 'EER.final'] = \
            {'minimize': True, 'value': eer}

        return metrics

    def _validate_epoch_default(self, epoch, protocol_name,
                                subset='development', validation_data=None):

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
        fX = model(X)

        y_pred = pdist(fX.data.numpy(), metric=self.approach_.metric)

        _, _, _, eer = det_curve(validation_data['y'], y_pred,
                                 distances=True)

        metrics = {}
        metrics['EER.1seq'] = {'minimize': True, 'value': eer}
        return metrics

    def apply(self, protocol_name, output_dir, step=None, internal=False):

        # load best performing model
        with open(self.validate_txt_, 'r') as fp:
            eers = SortedDict(np.loadtxt(fp))
        best_epoch = int(eers.iloc[np.argmin(eers.values())])
        model = self.load_model(best_epoch)

        # guess sequence duration from path (.../3.2+0.8/...)
        directory = basename(dirname(self.experiment_dir))
        duration = self.approach_.duration
        if step is None:
            step = 0.5 * duration

        # initialize embedding extraction
        batch_size = 32

        # do not use memmap as this would lead to too many open files
        if isinstance(self.feature_extraction_, Precomputed):
            self.feature_extraction_.use_memmap = False

        sequence_embedding = SequenceEmbedding(
            model, self.feature_extraction_, duration,
            step=step, batch_size=batch_size,
            internal=internal)
        sliding_window = sequence_embedding.sliding_window
        dimension = sequence_embedding.dimension

        # create metadata file at root that contains
        # sliding window and dimension information
        precomputed = Precomputed(
            root_dir=output_dir,
            sliding_window=sliding_window,
            dimension=dimension)

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        processed_uris = set()

        for subset in ['development', 'test', 'train']:

            try:
                file_generator = getattr(protocol, subset)()
                first_item = next(file_generator)
            except NotImplementedError as e:
                continue

            file_generator = getattr(protocol, subset)()

            for current_file in file_generator:

                # corner case when the same file is iterated several times
                uri = get_unique_identifier(current_file)
                if uri in processed_uris:
                    continue

                fX = sequence_embedding.apply(current_file)

                precomputed.dump(current_file, fX)
                processed_uris.add(uri)

def main():

    arguments = docopt(__doc__, version='Speaker embedding')

    db_yml = expanduser(arguments['--database'])
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']
    gpu = arguments['--gpu']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']

        if subset is None:
            subset = 'train'

        application = SpeakerEmbeddingPytorch(experiment_dir, db_yml=db_yml)
        application.train(protocol_name, subset=subset, gpu=gpu)

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

    if arguments['apply']:

        validate_txt = arguments['<validate.txt>']
        output_dir = arguments['<output_dir>']
        if subset is None:
            subset = 'test'

        step = arguments['--step']
        if step is not None:
            step = float(step)

        internal = arguments['--internal']

        application = SpeakerEmbeddingPytorch.from_validate_txt(validate_txt)
        application.apply(protocol_name, output_dir, step=step,
                          internal=internal)
