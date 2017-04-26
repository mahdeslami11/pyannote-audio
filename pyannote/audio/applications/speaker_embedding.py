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
 pyannote-speaker-embedding data [--database=<db.yml> --subset=<subset>] <data_dir> <database.task.protocol>
 pyannote-speaker-embedding train <train_dir>
 pyannote-speaker-embedding -h | --help
 pyannote-speaker-embedding --version

Options:
  <data_dir>                 Set data root directory. This script expects a
                             configuration file called "config.yml" to live in
                             this directory. See '"data" mode' section below
                             for more details.
  <train_dir>                Set experiment root directory. This script expects
                             a configuration file called "config.yml" to live
                             in this directory. See '"train" mode' section
                             for more details.
  <database.task.protocol>   Set evaluation protocol (e.g.
                             "Etape.SpeakerDiarization.TV")
  --database=<db.yml>        Path to database configuration file.
                             [default: ~/.pyannote/db.yml]
  --subset=<subset>          Set subset (train|developement|test).
                             In "data" mode, default subset is "train".
                             In "validate" mode, default subset is "development".
  -h --help                  Show this screen.
  --version                  Show version.


Database configuration file:
    The database configuration provides details as to where actual files are
    stored. See `pyannote.audio.util.FileFinder` docstring for more information
    on the expected format.

"data" mode:

    A file called <data_dir>/config.yml should exist, that describes the
    feature extraction process (e.g. MFCCs) and the sequence generator used for
    generating training sequences.

    ................... <data_dir>/config.yml ...................
    feature_extraction:
       name: YaafeMFCC
       params:
          e: False                   # this experiments relies
          De: True                   # on 11 MFCC coefficients
          DDe: True                  # with 1st and 2nd derivatives
          D: True                    # without energy, but with
          DD: True                   # energy derivatives

    generator:
       duration: 3.2                 # sliding windows of 3.2s
       step: 1.6                     # with a step of 1.6s
       heterogeneous: True           # allow heterogeneous sequences
    ...................................................................

    Using "data" mode will create the following directory that contains
    the pre-computed training sequences:

        <data_dir>/<database.task.protocol>.<subset>/<duration>+<step><heterogeneous>

    This means that the sequences were generated on the <subset> subset of the
    <database.task.protocol> protocol. By default, <subset> is "train".
    This directory is called <sequences_dir> in the subsequent modes.

"train" mode:

    The configuration of each experiment is described in a file called
    <train_dir>/config.yml, that describes which precomputed sequences to
    use, the architecture of the neural network, and the approach (e.g. triplet
    loss) used for sequence embedding

    ................... <train_dir>/config.yml ...................
    data: <sequences_dir>

    architecture:
       name: StackedLSTM
       params:                       # this experiments relies
         n_classes: 2                # on one LSTM layer (16 outputs)
         lstm: [16]                  # and one dense layer.
         mlp: [16]                   # LSTM is bidirectional
         bidirectional: True

    approach:
       name: TripletLoss
       params:
         per_label: 5
         per_fold: 20

    ...................................................................

    Using "train" mode will create a bunch of files in <train_dir>
    including the pre-trained neural network weights after each epoch.

"""

from os.path import dirname, isfile, expanduser
import numpy as np

from docopt import docopt

from pyannote.database import get_protocol
from pyannote.audio.util import mkdir_p
import h5py

from .base import Application

from pyannote.generators.fragment import SlidingLabeledSegments

class SpeakerEmbedding(Application):

    DATA_DIR = '{data_dir}/{protocol}.{subset}/{params}'
    DATA_H5 = '{data_dir}/sequences.h5'
    TRAIN_DIR = '{train_dir}'

    @classmethod
    def from_data_dir(cls, data_dir, db_yml=None):
        speaker_embedding = cls(data_dir, db_yml=db_yml)
        speaker_embedding.data_dir_ = data_dir
        return speaker_embedding

    @classmethod
    def from_train_dir(cls, train_dir, db_yml=None):
        """Initialize application from <train_dir>"""
        speaker_embedding = cls(train_dir, db_yml=db_yml)
        speaker_embedding.train_dir_ = train_dir
        return speaker_embedding

    @classmethod
    def from_tune_dir(cls, tune_dir, db_yml=None):
        """Initialize application from <tune_dir>"""
        train_dir = dirname(dirname(tune_dir))
        speaker_embedding = cls.from_train_dir(train_dir,
                                               db_yml=db_yml)
        speaker_embedding.tune_dir_ = tune_dir
        return speaker_embedding

    def __init__(self, train_dir, db_yml=None):

        super(SpeakerEmbedding, self).__init__(
            train_dir, db_yml=db_yml)

        # architecture
        if 'architecture' in self.config_:
            architecture_name = self.config_['architecture']['name']
            models = __import__('pyannote.audio.embedding.models',
                                fromlist=[architecture_name])
            Architecture = getattr(models, architecture_name)
            self.architecture_ = Architecture(
                **self.config_['architecture'].get('params', {}))

        # approach
        if 'approach' in self.config_:
            approach_name = self.config_['approach']['name']
            approaches = __import__('pyannote.audio.embedding.approaches',
                                    fromlist=[approach_name])
            Approach = getattr(approaches, approach_name)
            self.approach_ = Approach(
                **self.config_['approach'].get('params', {}))

    # (5, None, None, False) ==> '5'
    # (5, 1, None, False) ==> '1-5'
    # (5, None, 2, False) ==> '5+2'
    # (5, 1, 2, False) ==> '1-5+2'
    # (5, None, None, True) ==> '5x'
    @staticmethod
    def params_to_directory(duration=5.0, min_duration=None, step=None,
                            heterogeneous=False, skip_unlabeled=True,
                            **kwargs):
        if not skip_unlabeled:
            raise NotImplementedError('skip_unlabeled not supported yet.')

        DIRECTORY = '' if min_duration is None else '{min_duration:g}-'
        DIRECTORY += '{duration:g}'
        if step is not None:
            DIRECTORY += '+{step:g}'
        if heterogeneous:
            DIRECTORY += 'x'
        return DIRECTORY.format(duration=duration,
                                min_duration=min_duration,
                                step=step)

    # (5, None, None, False) <== '5'
    # (5, 1, None, False) <== '1-5'
    # (5, None, 2, False) <== '5+2'
    # (5, 1, 2, False) <== '1-5+2'
    @staticmethod
    def directory_to_params(directory):
        heterogeneous = False
        if directory[-1] == 'x':
            heterogeneous = True
            directory = directory[:-1]
        tokens = directory.split('+')
        step = float(tokens[1]) if len(tokens) == 2 else None
        tokens = tokens[0].split('-')
        min_duration = float(tokens[0]) if len(tokens) == 2 else None
        duration = float(tokens[0]) if len(tokens) == 1 else float(tokens[1])
        return duration, min_duration, step, heterogeneous

    def data(self, protocol_name, subset='train'):

        # labeled segment generator
        generator = SlidingLabeledSegments(source='annotated',
                                           **self.config_['generator'])
        duration = generator.duration

        # file generator
        protocol = get_protocol(protocol_name, progress=True,
                                preprocessors=self.preprocessors_)
        file_generator = getattr(protocol, subset)()

        data_dir = self.DATA_DIR.format(
            data_dir=self.data_dir_,
            protocol=protocol_name,
            subset=subset,
            params=self.params_to_directory(**self.config_['generator']))
        mkdir_p(data_dir)

        data_h5 = self.DATA_H5.format(data_dir=data_dir)
        with h5py.File(data_h5, mode='w') as fp:

            # initialize with a fixed number of sequences
            n_sequences = 1000

            # dataset meant to store the speaker identifier
            Y = fp.create_dataset(
                'y', shape=(n_sequences, ),
                dtype=h5py.special_dtype(vlen=bytes),
                maxshape=(None, ))

            # dataset meant to store the speech turn unique ID
            Z = fp.create_dataset(
                'z', shape=(n_sequences, ),
                dtype=np.int64,
                maxshape=(None, ))

            i = 0  # number of sequences
            z = 0  # speech turn identifier

            for item in file_generator:

                # feature extraction
                features = self.feature_extraction_(item)

                for segment, y in generator.from_file(item):

                    # extract feature sequence
                    x = features.crop(segment,
                                      mode='center',
                                      fixed=duration)

                    # create X dataset to store feature sequences
                    # this cannot be done before because we need
                    # the number of samples per sequence and the
                    # dimension of feature vectors.
                    if i == 0:
                        # get number of samples and feature dimension
                        # from the first sequence...
                        n_samples, n_features = x.shape

                        # create X dataset accordingly
                        X = fp.create_dataset(
                            'X', dtype=x.dtype, compression='gzip',
                            shape=(n_sequences, n_samples, n_features),
                            chunks=(1, n_samples, n_features),
                            maxshape=(None, n_samples, n_features))

                        # make sure the speech turn identifier
                        # will not be erroneously incremented
                        prev_y = y

                    # increase the size of the datasets when full
                    if i == n_sequences:
                        n_sequences = int(n_sequences * 1.1)
                        X.resize(n_sequences, axis=0)
                        Y.resize(n_sequences, axis=0)
                        Z.resize(n_sequences, axis=0)

                    # save current feature sequence and its label
                    X[i] = x
                    Y[i] = y

                    # a change of label indicates that a new speech turn has began.
                    # increment speech turn identifier (z) accordingly
                    if y != prev_y:
                        prev_y = y
                        z += 1

                    # save speech turn identifier
                    Z[i] = z

                    # increment number of sequences
                    i += 1

            X.resize(i-1, axis=0)
            Y.resize(i-1, axis=0)
            Z.resize(i-1, axis=0)    

def main():

    arguments = docopt(__doc__, version='Speaker embedding')

    if not arguments['train']:
        db_yml = expanduser(arguments['--database'])
        protocol_name = arguments['<database.task.protocol>']
        subset = arguments['--subset']

    if arguments['data']:
        data_dir = arguments['<data_dir>']
        if subset is None:
            subset = 'train'
        application = SpeakerEmbedding.from_data_dir(data_dir, db_yml=db_yml)
        application.data(protocol_name, subset=subset)
