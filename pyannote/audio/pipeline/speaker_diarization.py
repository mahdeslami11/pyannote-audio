
import numpy as np
from pyannote.audio.keras_utils import load_model
from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.signal import Binarize, Peak
from pyannote.audio.embedding.extraction import SequenceEmbedding
from pyannote.audio.embedding.clustering import Clustering
from pyannote.core import Annotation
from pyannote.audio.embedding.utils import l2_normalize
from pyannote.database import get_annotated


class SpeakerDiarization(object):

    def __init__(self, sad, scd, emb,
                 sad__onset=0.7, sad__offset=0.7, sad__dimension=1,
                 scd__alpha=0.5, scd__min_duration=1., scd__dimension=1,
                 cls__min_cluster_size=5, cls__min_samples=None,
                 cls__metric='cosine'):

        super(SpeakerDiarization, self).__init__()

        # speech activity detection hyper-parameters
        self.sad = sad
        self.sad__onset = sad__onset
        self.sad__offset = sad__offset
        self.sad__dimension = sad__dimension

        # speaker change detection hyper-parameters
        self.scd = scd
        self.scd__alpha = scd__alpha
        self.scd__min_duration = scd__min_duration
        self.scd__dimension = scd__dimension

        # embedding hyper-parameters
        self.emb = emb

        # clustering hyper-parameters
        self.cls__min_cluster_size = cls__min_cluster_size
        self.cls__min_samples = cls__min_samples
        self.cls__metric = cls__metric

        # initialize speech activity detection module
        self.sad_binarize_ = Binarize(onset=self.sad__onset,
                                      offset=self.sad__offset)

        # initialize speaker change detection module
        self.scd_peak_ = Peak(alpha=self.scd__alpha,
                              min_duration=self.scd__min_duration,
                              percentile=False)

        # initialize clustering module
        self.cls_ = Clustering(metric=self.cls__metric,
                               min_cluster_size=self.cls__min_cluster_size,
                               min_samples=self.cls__min_samples)


    def __call__(self, current_file, annotated=False):

        # speech activity detection
        soft_sad = self.sad(current_file)
        hard_sad = self.sad_binarize_.apply(
            soft_sad, dimension=self.sad__dimension)

        # speaker change detection
        soft_scd = self.scd(current_file)
        hard_scd = self.scd_peak_.apply(
            soft_scd, dimension=self.scd__dimension)

        # speech turns
        speech_turns = hard_scd.crop(hard_sad)

        if annotated:
            speech_turns = speech_turns.crop(
                get_annotated(current_file))

        hypothesis = Annotation(uri=current_file['uri'])
        if not speech_turns:
            return hypothesis

        # speech turns embedding
        emb = self.emb(current_file)

        fX_ = [np.sum(emb.crop(speech_turn, mode='loose'), axis=0)
               for speech_turn in speech_turns]
        fX = l2_normalize(np.vstack(fX_))

        # speech turn clustering
        cluster_labels = self.cls_.apply(fX)

        # build hypothesis from clustering results
        for speech_turn, label in zip(speech_turns, cluster_labels):
            hypothesis[speech_turn] = label
        return hypothesis
