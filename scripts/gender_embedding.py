import pyannote.core  # HACK
import sys
import yaml
import random
random.seed(1337)  # deterministic behavior

# BEFORE ANY OTHER LIBRARIES...
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from etape import Etape

# FEATURE EXTRACTION
from pyannote.audio.features.yaafe import YaafeMFCC

# TRAINING
from pyannote.audio.embedding.models import TripletLossSequenceEmbedding
from pyannote.audio.embedding.generator import YaafeTripletBatchGenerator

# TESTING
from pyannote.audio.embedding.callback import ValidationCheckpoint
from pyannote.audio.features.yaafe import YaafeBatchGenerator
from pyannote.generators.fragment import RandomSegmentPairs
import numpy as np


# -- PROTOCOL --
protocol = Etape('/vol/work1/bredin/corpus/odessa/etape/')
gender_txt = '/vol/work1/bredin/keras/gender/gender.txt'

with open(sys.argv[1], 'r') as fp:
    config = yaml.load(fp)

# -- FEATURE EXTRACTION --

# input sequence duration
duration = config['feature_extraction']['duration']
# MFCCs
feature_extractor = YaafeMFCC(**config['feature_extraction']['mfcc'])
# normalization
normalize = config['feature_extraction']['normalize']

# -- EMBEDDING STRUCTURE --

# embedding dimension
output_dim = config['embedding']['output_dim']
# internal embedding structure
lstm = config['embedding']['lstm']
dense = config['embedding']['dense']
# dropout
dropout = config['embedding']['dropout']
# bi-directional
bi_directional = config['embedding']['bi_directional']

# -- TRAINING --

alpha = config['training']['margin']
per_label = config['training']['per_label']
batch_size = config['training']['batch_size']
batch_per_epoch = config['training']['batch_per_epoch']
nb_epoch = config['training']['nb_epoch']

# -- TESTING --

per_label_test = config['testing']['per_label']
batch_size_test = config['testing']['batch_size']

# -- LOGS --

workdir = config['workdir']
checkpoint_h5 = workdir + '/weights.{epoch:03d}.{loss:.3f}.h5'
architecture_yml = workdir + '/architecture.yml'


class GenderMixin:

    def load_gender(self, gender_txt):
        genders = {}
        with open(gender_txt, 'r') as fp:
            for line in fp:
                person_name, gender = line.strip().split()
                genders[person_name] = gender
        return genders

    def to_gender(self, reference):
        return reference.translate(self.gender).subset(['male', 'female'])

    def preprocess(self, protocol_item, identifier=None):
        protocol_item = super(self.__class__, self).preprocess(protocol_item, identifier=identifier)
        wav, uem, reference = protocol_item
        gender = self.to_gender(reference)
        return wav, uem, gender


class PairBatchGenerator(YaafeBatchGenerator, GenderMixin):

    def __init__(self, extractor, gender_txt, duration=3.2,
                 normalize=False, per_label=40, batch_size=32):

        generator = RandomSegmentPairs(
            duration=duration,
            per_label=per_label,
            yield_label=False)

        super(PairBatchGenerator, self).__init__(
            extractor,
            generator,
            batch_size=batch_size,
            normalize=normalize)

        self.gender = self.load_gender(gender_txt)


class TripletBatchGenerator(YaafeTripletBatchGenerator, GenderMixin):

    def __init__(self, extractor, embedding, gender_txt, duration=3.2,
                 normalize=False, per_label=40, batch_size=32):

        super(TripletBatchGenerator, self).__init__(
            extractor,
            embedding,
            duration=duration,
            normalize=normalize,
            per_label=per_label,
            batch_size=batch_size)

        self.gender = self.load_gender(gender_txt)

# embedding
embedding = TripletLossSequenceEmbedding(
    output_dim, lstm=lstm, dense=dense,
    alpha=alpha, checkpoint=checkpoint_h5)

# pair generator for testing
pair_batch_generator = PairBatchGenerator(
    feature_extractor, gender_txt,
    duration=duration, normalize=normalize,
    per_label=per_label_test, batch_size=batch_size_test)

validation_checkpoint = ValidationCheckpoint(embedding, pair_batch_generator, protocol, checkpoint=workdir)

# triplet generator for training
triplet_batch_generator = TripletBatchGenerator(
    feature_extractor, embedding, gender_txt,
    duration=duration, normalize=normalize,
    per_label=per_label, batch_size=batch_size)

input_shape = triplet_batch_generator.get_shape()

# save model architecture
embedding.to_disk(architecture=architecture_yml,
                  input_shape=input_shape,
                  overwrite=True)

# training
samples_per_epoch = batch_size * batch_per_epoch
generator = triplet_batch_generator(protocol.train_iter, infinite=True)
embedding.fit(input_shape, generator,
              samples_per_epoch, nb_epoch,
              max_q_size=10, verbose=1,
              callbacks=[validation_checkpoint])
