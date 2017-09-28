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
# Ruiqing YIN
# Hervé BREDIN - http://herve.niderb.fr

"""
Speaker change detection

Usage:
  pyannote-change-detection train [--database=<db.yml> --subset=<subset>] <experiment_dir> <database.task.protocol>
  pyannote-change-detection validate [--database=<db.yml> --subset=<subset> --from=<epoch> --to=<epoch> --every=<epoch> --purity=<purity>] <train_dir> <database.task.protocol>
  pyannote-change-detection tune [--database=<db.yml> --subset=<subset> [--from=<epoch> --to=<epoch> | --at=<epoch>] --purity=<purity>] <train_dir> <database.task.protocol>
  pyannote-change-detection apply [--database=<db.yml> --subset=<subset>] <tune_dir> <database.task.protocol>
  pyannote-change-detection -h | --help
  pyannote-change-detection --version

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
  --at=<epoch>               In "tune" mode, use this very epoch.
  --purity=<purity>          Target segment purity [default: 0.9].

"train" mode:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.

"validation" mode:
  --every=<epoch>            Validate model every <epoch> epochs [default: 1].
  <train_dir>                Path to the directory containing pre-trained
                             models (i.e. the output of "train" mode).

"tune" mode:
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
       name: YaafeMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives

    architecture:
       name: StackedLSTM
       params:                       # this experiments relies
         n_classes: 2                # on one LSTM layer (16 outputs)
         lstm: [16]                  # and one dense layer.
         mlp: [16]                   # LSTM is bidirectional
         bidirectional: ave

    sequences:
       duration: 3.2                 # this experiments relies
       step: 0.8                     # on sliding windows of 3.2s
                                     # with a step of 0.8s
       batch_size: 1024
    ...................................................................

"train" mode:
    First, one should train the raw sequence labeling neural network using
    "train" mode. This will create the following directory that contains
    the pre-trained neural network weights after each epoch:

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
    "developement.DiarizationCoverage.{txt|png|eps}" files are created and
    updated continuously, epoch after epoch. This directory is called
    <validate_dir> in the subsequent "tune" mode.

    In practice, for each epoch, "validate" mode will approximate  the set of
    hyper-parameters that maximizes coverage, given that the purity must be
    higher than the value provided by the "--purity" option.

"tune" mode:
    To actually use the optimal set of hyper-parameters, one should tune the
    system using "tune" mode. This will create the following files describing
    the best hyper-parameters to use:

        <train_dir>/tune/<database.task.protocol>.<subset>/tune.yml
        <train_dir>/tune/<database.task.protocol>.<subset>/tune.png

    This means that hyper-parameters were tuned on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "development".
    This directory is called <tune_dir> in the subsequent "apply" mode.

    In practice, "tune" mode will look for the set of hyper-parameters that
    maximizes coverage, given that purity must be higher than the value
    provided by the "--purity" option.

"apply" mode
    Finally, one can apply speaker change detection using "apply" mode.
    This will create the following files that contains the hard (mdtm) and
    soft (h5) outputs of speaker change detection, based on the set of hyper-
    parameters obtain with "tune" mode:

        <tune_dir>/apply/<database.task.protocol>.<subset>.mdtm
        <tune_dir>/apply/{database}/{uri}.h5

    This means that file whose unique resource identifier is {uri} has been
    processed.
"""

import io
import yaml
from os.path import dirname, expanduser
import numpy as np

from docopt import docopt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.generators.change import \
    ChangeDetectionBatchGenerator
from pyannote.audio.optimizers import SSMORMS3

from pyannote.audio.signal import Peak
from pyannote.database.util import get_annotated
from pyannote.database import get_protocol


from pyannote.parser import MDTMParser

from pyannote.audio.util import mkdir_p

from pyannote.audio.features.utils import Precomputed
import h5py

from .speech_detection import SpeechActivityDetection

import skopt
import skopt.space
from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure


def tune_peak(app, epoch, protocol_name, subset='development', purity=0.9,
              n_calls=20, n_random_starts=10, n_jobs=-1):
    """Tune peak detection

    Parameters
    ----------
    app : SpeakerChangeDetection
    epoch : int
        Epoch number.
    protocol_name : str
        E.g. 'Etape.SpeakerDiarization.TV'
    subset : {'train', 'development', 'test'}, optional
        Defaults to 'development'.
    purity : float, optional
        Target purity. Defaults to 0.95.
    n_calls : int, optional
        Number of trials for hyper-parameter optimization. Defaults to 20.
    n_random_starts : int, optional
        Number of trials with random initialization before being smart.
        Defaults to 10.
    n_jobs : int, optional
        Number of parallel job to use. Set to 1 to not use multithreading.
        Defaults to whichever is minimum between number of CPUs and number
        of items.


    Returns
    -------
    params : dict
        See Peak.tune
    coverage : float
        Best achieved coverage (at target purity)
    """

    # initialize protocol
    protocol = get_protocol(protocol_name, progress=False,
                            preprocessors=app.preprocessors_)

    # load model for epoch 'epoch'
    model = app.load_model(epoch)

    # initialize sequence labeling
    duration = app.config_['sequences']['duration']
    sequence_labeling = SequenceLabeling(
        model, app.feature_extraction_,
        duration, step=.9 * duration,
        source='annotated')
    sequence_labeling.cache_preprocessed_ = False

    # tune Peak parameters (alpha & min_duration)
    # with respect to coverage @ given purity
    peak_params, coverage = Peak.tune(
        getattr(protocol, subset)(), sequence_labeling.apply, purity=purity,
        n_calls=n_calls, n_random_starts=n_random_starts, n_jobs=n_jobs)

    return peak_params, coverage


class SpeakerChangeDetection(SpeechActivityDetection):

    def train(self, protocol_name, subset='train'):

        train_dir = self.TRAIN_DIR.format(
            experiment_dir=self.experiment_dir,
            protocol=protocol_name,
            subset=subset)

        # sequence batch generator
        batch_size = self.config_['sequences'].get('batch_size', 8192)
        duration = self.config_['sequences']['duration']
        step = self.config_['sequences']['step']
        balance = self.config_['sequences']['balance']
        batch_generator = ChangeDetectionBatchGenerator(
            self.feature_extraction_, duration=duration, step=step,
            balance=balance, batch_size=batch_size)
        batch_generator.cache_preprocessed_ = self.cache_preprocessed_

        protocol = get_protocol(protocol_name, progress=False,
                                preprocessors=self.preprocessors_)

        # total train duration
        train_total = protocol.stats(subset)['annotated']
        # number of batches per epoch
        steps_per_epoch = int(np.ceil((train_total / step) / batch_size))

        # input shape (n_frames, n_features)
        input_shape = batch_generator.shape

        # generator that loops infinitely over all training files
        train_files = getattr(protocol, subset)()
        generator = batch_generator(train_files, infinite=True)

        return SequenceLabeling.train(
            input_shape, self.architecture_, generator, steps_per_epoch, 1000,
            loss='binary_crossentropy', optimizer=SSMORMS3(),
            log_dir=train_dir)

        return labeling

    def validate_init(self, protocol_name, subset='development', purity=0.9):
        return {'purity': purity}

    def validate_epoch(self, epoch, protocol_name, subset='development',
                       validation_data=None):

        from pyannote.metrics.diarization import DiarizationPurityCoverageFMeasure
        from pyannote.audio.signal import Peak

        purity = validation_data['purity']

        peak_params, coverage = tune_peak(
            self, epoch, protocol_name, subset=subset, purity=purity)

        return {'DiarizationCoverage': {'minimize': False, 'value': coverage}}

    def tune(self, protocol_name, subset='development',
             start=None, end=None, at=None, purity=0.9):

        # FIXME -- make sure "subset" is not empty

        tune_dir = self.TUNE_DIR.format(
            train_dir=self.train_dir_,
            protocol=protocol_name,
            subset=subset)

        mkdir_p(tune_dir)

        epoch, first_epoch = self.get_number_of_epochs(self.train_dir_,
                                                       return_first=True)

        if at is None:
            if start is None:
                start = first_epoch
            if end is None:
                end = epoch - 1

        else:
            start = at
            end = at

        space = [skopt.space.Integer(start, end)]

        best_params = {}
        best_metric = {}

        tune_yml = self.TUNE_YML.format(tune_dir=tune_dir)
        tune_png = self.TUNE_PNG.format(tune_dir=tune_dir)

        def callback(res):

            # plot convergence
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import skopt.plots
            _ = skopt.plots.plot_convergence(res)
            plt.savefig(tune_png, dpi=75)
            plt.close()

            # save state
            params = {'status': {'epochs': epoch,
                                 'start': start,
                                 'end': end,
                                 'objective': float(res.fun)},
                      'epoch': int(res.x[0]),
                      'alpha': float(best_params[tuple(res.x)]['alpha']),
                      'min_duration': float(best_params[tuple(res.x)]['min_duration'])
                      }

            with io.open(tune_yml, 'w') as fp:
                yaml.dump(params, fp, default_flow_style=False)

        def objective_function(params):

            params = tuple(params)
            epoch, = params

            # do not rerun everything if epoch has already been tested
            if params in best_metric:
                return best_metric[params]

            # tune peak detection
            peak_params, coverage = tune_peak(
                self, epoch, protocol_name, subset=subset, purity=purity)

            # remember outcome of this trial
            best_params[params] = peak_params
            best_metric[params] = 1. - coverage

            return 1. - coverage

        res = skopt.gp_minimize(
            objective_function, space, random_state=1337,
            n_calls=20, n_random_starts=10, x0=[end],
            verbose=True, callback=callback)

        coverage = 1. - res.fun
        return {'epoch': res.x[0]}, coverage

    def apply(self, protocol_name, subset='test'):

        apply_dir = self.APPLY_DIR.format(tune_dir=self.tune_dir_)

        mkdir_p(apply_dir)

        # load tuning results
        tune_yml = self.TUNE_YML.format(tune_dir=self.tune_dir_)
        with io.open(tune_yml, 'r') as fp:
            self.tune_ = yaml.load(fp)

        # load model for epoch 'epoch'
        epoch = self.tune_['epoch']
        model = self.load_model(epoch)

        # initialize sequence labeling
        duration = self.config_['sequences']['duration']
        step = self.config_['sequences']['step']
        sequence_labeling = SequenceLabeling(
            model, self.feature_extraction_, duration,
            step=step)

        # initialize protocol
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)

        for i, item in enumerate(getattr(protocol, subset)()):

            predictions = sequence_labeling.apply(item)

            if i == 0:
                # create metadata file at root that contains
                # sliding window and dimension information
                path = Precomputed.get_config_path(apply_dir)
                f = h5py.File(path)
                f.attrs['start'] = predictions.sliding_window.start
                f.attrs['duration'] = predictions.sliding_window.duration
                f.attrs['step'] = predictions.sliding_window.step
                f.attrs['dimension'] = 2
                f.close()

            path = Precomputed.get_path(apply_dir, item)

            # create parent directory
            mkdir_p(dirname(path))

            f = h5py.File(path)
            f.attrs['start'] = predictions.sliding_window.start
            f.attrs['duration'] = predictions.sliding_window.duration
            f.attrs['step'] = predictions.sliding_window.step
            f.attrs['dimension'] = 2
            f.create_dataset('features', data=predictions.data)
            f.close()

        # initialize peak detection
        alpha = self.tune_['alpha']
        min_duration = self.tune_['min_duration']
        peak = Peak(alpha=alpha, min_duration=min_duration)

        precomputed = Precomputed(root_dir=apply_dir)

        writer = MDTMParser()
        path = self.HARD_MDTM.format(apply_dir=apply_dir,
                                     protocol=protocol_name,
                                     subset=subset)
        with io.open(path, mode='w') as gp:
            for item in getattr(protocol, subset)():
                predictions = precomputed(item)
                segmentation = peak.apply(predictions)
                writer.write(segmentation.to_annotation(),
                             f=gp, uri=item['uri'], modality='speaker')

def main():

    arguments = docopt(__doc__, version='Speaker change detection')

    db_yml = expanduser(arguments['--database'])
    protocol_name = arguments['<database.task.protocol>']
    subset = arguments['--subset']

    if arguments['train']:
        experiment_dir = arguments['<experiment_dir>']

        if subset is None:
            subset = 'train'

        application = SpeakerChangeDetection(experiment_dir, db_yml=db_yml)
        application.train(protocol_name, subset=subset)


    if arguments['validate']:
        train_dir = arguments['<train_dir>']

        if subset is None:
            subset = 'development'

        # start validating at this epoch (defaults to 0)
        start = arguments['--from']
        if start is None:
            start = 0
        else:
            start = int(start)

        # stop validating at this epoch (defaults to None)
        end = arguments['--to']
        if end is not None:
            end = int(end)

        # validate every that many epochs (defaults to 1)
        every = int(arguments['--every'])

        purity = float(arguments['--purity'])

        application = SpeakerChangeDetection.from_train_dir(
            train_dir, db_yml=db_yml)
        application.validate(protocol_name, subset=subset,
                             start=start, end=end, every=every,
                             purity=purity)

    if arguments['tune']:
        train_dir = arguments['<train_dir>']

        if subset is None:
            subset = 'development'

        # start tuning at this epoch (defaults to None)
        start = arguments['--from']
        if start is not None:
            start = int(start)

        # stop tuning at this epoch (defaults to None)
        end = arguments['--to']
        if end is not None:
            end = int(end)

        at = arguments['--at']
        if at is not None:
            at = int(at)

        purity = float(arguments['--purity'])

        application = SpeakerChangeDetection.from_train_dir(
            train_dir, db_yml=db_yml)
        application.tune(protocol_name, subset=subset,
                         start=start, end=end, at=at,
                         purity=purity)

    if arguments['apply']:
        tune_dir = arguments['<tune_dir>']

        if subset is None:
            subset = 'test'

        application = SpeakerChangeDetection.from_tune_dir(
            tune_dir, db_yml=db_yml)
        application.apply(protocol_name, subset=subset)
