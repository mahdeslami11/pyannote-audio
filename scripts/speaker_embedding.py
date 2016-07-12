import pyannote.core  # HACK
import os.path
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
from pyannote.audio.embedding.generator import TripletBatchGenerator

# TESTING
from pyannote.audio.embedding.callback import ValidationCheckpoint
from pyannote.audio.generators.speaker import SpeakerPairsBatchGenerator

config_path = sys.argv[1]
workdir = os.path.dirname(config_path)

with open(config_path, 'r') as fp:
    config = yaml.load(fp)

# -- PROTOCOL --
protocol = Etape(config['etape'])

# -- FEATURE EXTRACTION --

# input sequence duration
duration = config['feature_extraction']['duration']
# MFCCs
feature_extractor = YaafeMFCC(**config['feature_extraction']['mfcc'])
# normalization
normalize = config['feature_extraction']['normalize']

# -- EMBEDDING STRUCTURE --
# triplet loss margin
margin = config['embedding']['margin']
# embedding dimension
output_dim = config['embedding']['output_dim']
# internal embedding structure
lstm = config['embedding']['lstm']
dense = config['embedding']['dense']
# bi-directional
bidirectional = config['embedding']['bidirectional']
# final activation
space = config['embedding']['space'] # 'sphere' or 'quadrant'

# -- TRAINING --

# estimated number of labels in training set
# (used to estimate the number of samples in each epoch)
n_labels_estimate = config['training']['n_labels_estimate']
# number of labels in each group
n_labels = config['training']['triplet']['n_labels']
# number of samples per label
per_label = config['training']['triplet']['per_label']
# batch size
batch_size = config['training']['batch_size']
# number of epochs
nb_epoch = config['training']['nb_epoch']
# optimizer
optimizer = config['training']['optimizer']

# -- TESTING --

per_label_test = config['testing']['per_label']
batch_size_test = config['testing']['batch_size']

# -- LOGS --

checkpoint_h5 = workdir + '/weights.{epoch:03d}.{loss:.3f}.h5'
architecture_yml = workdir + '/architecture.yml'

# embedding
embedding = TripletLossSequenceEmbedding(
    output_dim, lstm=lstm, dense=dense,
    bidirectional=bidirectional, space=space,
    margin=margin, optimizer=optimizer,
    checkpoint=checkpoint_h5)

# pair generator for testing
pair_generator = SpeakerPairsBatchGenerator(
    feature_extractor,
    duration=duration, normalize=normalize,
    per_label=per_label_test, batch_size=batch_size_test)

validation_checkpoint = ValidationCheckpoint(
    embedding, pair_generator, protocol, checkpoint=workdir)

# triplet generator for training
file_generator = protocol.train_iter()
triplet_batch_generator = TripletBatchGenerator(
    feature_extractor, file_generator, embedding,
    duration=duration, overlap=0.0, normalize=normalize,
    n_labels=n_labels, per_label=per_label,
    batch_size=batch_size, forward_batch_size=n_labels * per_label)

input_shape = triplet_batch_generator.get_shape()

# save model architecture
embedding.to_disk(architecture=architecture_yml,
                  input_shape=input_shape,
                  overwrite=True)

# training

# number of (anchor, positive) pairs for one label
# multiplied by (estimate) number of labels
samples_per_epoch = per_label * (per_label - 1) * n_labels_estimate

embedding.fit(input_shape, triplet_batch_generator,
              samples_per_epoch, nb_epoch,
              max_q_size=10, verbose=1,
              callbacks=[validation_checkpoint])
