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
from keras.callbacks import Callback
from pyannote.audio.features.yaafe import YaafeBatchGenerator
from pyannote.generators.fragment import RandomSegmentPairs
from sklearn.metrics import precision_recall_curve
import numpy as np
import scipy.stats


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


class ValidationCheckpoint(Callback):

    def __init__(self, sequence_embedding, generator, protocol, checkpoint='/tmp/checkpoint'):
        super(ValidationCheckpoint, self).__init__()
        self.sequence_embedding = sequence_embedding
        self.generator = generator
        self.protocol = protocol
        self.checkpoint = checkpoint
        self.accuracy = {'dev': [], 'test': []}
        self.fscore = {'dev': [], 'test': []}
        self.loss = []

    def validation(self, protocol_iter):

        embedding = self.sequence_embedding.get_embedding(self.model)

        Y, Distance = [], []

        for batch in self.generator(protocol.dev_iter, infinite=False):
            (query, returned), y = batch
            Xq = embedding.predict_on_batch(query)
            Xr = embedding.predict_on_batch(returned)
            distance = np.sum((Xq - Xr) ** 2, axis=-1)
            Distance.append(distance)
            Y.append(y)
        y = np.hstack(Y)
        distance = np.hstack(Distance)

        precision, recall, thresholds = precision_recall_curve(
            y, -distance, pos_label=True)
        thresholds = -thresholds

        fscore = np.hstack([scipy.stats.hmean(np.vstack([precision, recall])[:,:-1], axis=0), [0]])
        accuracy = [np.mean(y == (distance < threshold))
                    for threshold in thresholds]

        return y, distance, thresholds, precision, recall, fscore, accuracy

    def on_epoch_end(self, epoch, logs={}):

        self.loss.append(logs['loss'])

        embedding = self.sequence_embedding.get_embedding(self.model)

        bins = np.arange(0, 2, 0.05)

        # development set
        y, distance, thresholds, precision, recall, fscore, accuracy = self.validation(protocol.dev_iter)

        # find threshold maximizing accuracy (on dev)
        A = np.argmax(accuracy)
        accuracy_threshold = thresholds[A]
        self.accuracy['dev'].append(accuracy[A])

        # find threshold maximizing f-score (on dev)
        F = np.argmax(fscore)
        fscore_threshold = thresholds[F]
        self.fscore['dev'].append(fscore[F])

        plt.figure(figsize=(12, 8))

        # plot inter- vs. intra-class distance distributions
        plt.subplot(2, 3, 1)
        plt.hist(distance[y], bins=bins, color='g', alpha=0.5, normed=True)
        plt.hist(distance[~y], bins=bins, color='r', alpha=0.5, normed=True)

        # plot precision / recall curve
        # show best operating point
        plt.subplot(2, 3, 2)
        plt.plot(recall, precision, 'b')
        plt.plot([recall[F]], [precision[F]], 'bo')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # plot fscore and accuracy curves
        # show best operating points
        plt.subplot(2, 3, 3)
        plt.plot(thresholds, fscore[:-1], 'b', label='f-score')
        plt.plot(thresholds, accuracy, 'g', label='accuracy')
        plt.xlabel('Threshold')
        plt.xlim(np.min(bins), np.max(bins))
        plt.ylim(0, 1)
        plt.plot([thresholds[F]], [fscore[F]], 'bo')
        plt.plot([thresholds[A]], [accuracy[A]], 'go')
        plt.legend(loc='lower right')

        # test set
        y, distance, thresholds, precision, recall, fscore, accuracy = self.validation(protocol.test_iter)

        # find threshold most similar to the ones selected on dev
        [a, f] = np.searchsorted(-thresholds, [-accuracy_threshold, -fscore_threshold])
        # evaluate accuracy with dev-optimized threshold
        self.accuracy['test'].append(accuracy[a])
        # evaluate fscore with dev-optimized threshold
        self.fscore['test'].append(fscore[f])

        # plot inter- vs. intra-class distance distributions
        plt.subplot(2, 3, 4)
        plt.hist(distance[y], bins=bins, color='g', alpha=0.5, normed=True)
        plt.hist(distance[~y], bins=bins, color='r', alpha=0.5, normed=True)

        # plot precision / recall
        # show dev-optimized operating point
        plt.subplot(2, 3, 5)
        plt.plot(recall, precision, 'b')
        plt.plot([recall[f]], [precision[f]], 'bo')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        # plot fscore and accuracy curves
        # show dev-optimized operating points
        plt.subplot(2, 3, 6)
        plt.plot(thresholds, fscore[:-1], 'b', label='f-score')
        plt.plot(thresholds, accuracy, 'g', label='accuracy')
        plt.xlabel('Threshold')
        plt.xlim(np.min(bins), np.max(bins))
        plt.ylim(0, 1)
        plt.plot([thresholds[f]], [fscore[f]], 'bo')
        plt.plot([thresholds[a]], [accuracy[a]], 'go')

        plt.tight_layout()
        plt.savefig(self.checkpoint + '/{epoch:03d}.png'.format(epoch=epoch), dpi=150)
        plt.savefig(self.checkpoint + '/{epoch:03d}.eps'.format(epoch=epoch))
        plt.close()

        plt.figure(figsize=(4, 8))

        plt.subplot(2, 1, 2)
        plt.plot(self.accuracy['dev'], 'b', label='Accuracy (dev)')
        plt.plot(self.accuracy['test'], 'b--', label='Accuracy (test)')
        plt.plot(self.fscore['dev'], 'g', label='FScore (dev)')
        plt.plot(self.fscore['test'], 'g--', label='FScore (test)')
        plt.title('Evaluation')
        plt.xlabel('Epoch')
        plt.ylim(0, 1)
        plt.legend(loc='lower right')

        plt.subplot(2, 1, 1)
        plt.plot(self.loss, 'b')
        plt.xlabel('Epoch')
        plt.title('Loss (train)')

        plt.tight_layout()

        plt.savefig(self.checkpoint + '/status.png', dpi=150)
        plt.savefig(self.checkpoint + '/status.eps')
        plt.close()

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
