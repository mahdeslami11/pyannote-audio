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
Feature extraction

Usage:
  feature_extraction [--robust --database=<db.yml>] <experiment_dir> <database.task.protocol>
  feature_extraction -h | --help
  feature_extraction --version

Options:
  <experiment_dir>           Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See "Configuration file"
                             section below for more details.
  <database.task.protocol>   Set evaluation protocol (e.g. "Etape.SpeakerDiarization.TV")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --robust                   When provided, skip files for which feature extraction fails.
  -h --help                  Show this screen.
  --version                  Show version.

Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.audio.util.FileFinder` docstring for more information
    on the expected format.

Configuration file:
    The configuration of each experiment is described in a file called
    <experiment_dir>/config.yml, that describes the feature extraction process
    (e.g. MFCCs).

    ................... <experiment_dir>/config.yml ...................
    feature_extraction:
       name: YaafeMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives
    ...................................................................

"""

import yaml
import h5py
import os.path
import itertools
import numpy as np
from docopt import docopt

import pyannote.core
import pyannote.database
from pyannote.database import get_database
from pyannote.database.util import FileFinder

from pyannote.audio.features.utils import PyannoteFeatureExtractionError


def extract(database_name, task_name, protocol_name, preprocessors, experiment_dir, robust=False):

    database = get_database(database_name, preprocessors=preprocessors)
    protocol = database.get_protocol(task_name, protocol_name, progress=True)

    if task_name == 'SpeakerDiarization':
        items = itertools.chain(protocol.train(),
                                protocol.development(),
                                protocol.test())

    elif task_name == 'SpeakerRecognition':
        items = itertools.chain(protocol.train(yield_name=False),
                                protocol.development_enroll(yield_name=False),
                                protocol.development_test(yield_name=False),
                                protocol.test_enroll(yield_name=False),
                                protocol.test_test(yield_name=False))

    # load configuration file
    config_yml = experiment_dir + '/config.yml'
    with open(config_yml, 'r') as fp:
        config = yaml.load(fp)

    feature_extraction_name = config['feature_extraction']['name']
    features = __import__('pyannote.audio.features.yaafe',
                          fromlist=[feature_extraction_name])
    FeatureExtraction = getattr(features, feature_extraction_name)
    feature_extraction = FeatureExtraction(
        **config['feature_extraction'].get('params', {}))

    features_h5 = experiment_dir + '/features.h5'
    with h5py.File(features_h5, mode='a', libver='latest') as fp:

        sliding_window = feature_extraction.sliding_window()
        fp.attrs['start'] = sliding_window.start
        fp.attrs['duration'] = sliding_window.duration
        fp.attrs['step'] = sliding_window.step

        dimension = feature_extraction.dimension()
        fp.attrs['dimension'] = dimension

        for item in items:

            wav = item['wav']
            if wav in fp:
                continue

            try:
                features = feature_extraction(wav)
            except PyannoteFeatureExtractionError as e:
                if robust:
                    msg = 'Feature extraction failedfor file "{wav}".'
                    msg = msg.format(wav=wav)
                    continue
                else:
                    raise e

            if features is None:
                msg = 'Feature extraction returned None for file "{wav}".'
                msg = msg.format(wav=wav)
                if not robust:
                    raise PyannoteFeatureExtractionError(msg)
                warnings.warn(msg)
                continue

            data = features.data

            if np.any(np.isnan(data)):
                msg = 'Feature extraction returned NaNs for file "{wav}".'
                msg = msg.format(wav=wav)
                if not robust:
                    raise PyannoteFeatureExtractionError(msg)
                warnings.warn(msg)
                continue

            fp.create_dataset(wav, data=data)


if __name__ == '__main__':

    arguments = docopt(__doc__, version='Feature extraction')

    db_yml = os.path.expanduser(arguments['--database'])
    preprocessors = {'wav': FileFinder(db_yml)}

    database_name, task_name, protocol_name = arguments['<database.task.protocol>'].split('.')
    experiment_dir = arguments['<experiment_dir>']
    robust = arguments['--robust']
    extract(database_name, task_name, protocol_name, preprocessors, experiment_dir, robust=robust)
