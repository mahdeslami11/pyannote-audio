
from pyannote.audio.keras_utils import load_model
from pyannote.audio.labeling.base import SequenceLabeling
from pyannote.audio.signal import Binarize, Peak
from pyannote.audio.embedding.extraction import SequenceEmbedding
from pyannote.audio.embedding.clustering import Clustering


class SpeakerDiarization(object):

    def __init__(self, feature_extraction, sad__h5, scd__h5, emb__h5,
                 sad__onset=0.7, sad__offset=0.7, sad__dimension=1,
                 scd__alpha=0.5, scd__dimension=1,
                 emb__internal=False,
                 cls__min_cluster_size=5, cls__min_samples=None,
                 cls__metric='cosine'):

        super(SpeakerDiarization, self).__init__()

        self.feature_extraction = feature_extraction

        # speech activity detection hyper-parameters
        self.sad__h5 = sad__h5
        self.sad__onset = sad__onset
        self.sad__offset = sad__offset
        self.sad__dimension = sad__dimension

        # speaker change detection hyper-parameters
        self.scd__h5 = scd__h5
        self.scd__alpha = scd__alpha
        self.scd__min_duration = min_duration

        # embedding hyper-parameters
        self.emb__h5 = emb__h5
        self.emb__internal = emb__internal

        # clustering hyper-parameters
        self.cls__min_cluster_size = cls__min_cluster_size
        self.cls__min_samples = cls__min_samples
        self.cls__metric = cls__metric

        step = self.feature_extraction.sliding_window().step

        # initialize speech activity detection module
        sad_model = load_model(self.sad__h5, compile=False)
        sad_duration = step * sad_model.input_shape[1]
        self.sad_ = SequenceLabeling(sad_model, feature_extraction,
                                     sad_duration, step=sad_duration / 4,
                                     batch_size=32, source='audio')
        self.sad_binarize_ = Binarize(onset=self.sad__onset,
                                      offset=self.sad__offset)

        # initialize speaker change detection module
        scd_model = load_model(self.scd__h5, compile=False)
        scd_duration = step * scd_model.input_shape[1]
        self.scd_ = SequenceLabeling(scd_model, feature_extraction,
                                     scd_duration, step=scd_duration / 4,
                                     batch_size=32, source='audio')
        self.scd_peak_ = Peak(alpha=self.scd__alpha,
                              min_duration=self.scd__min_duration,
                              percentile=False)

        # initialize speech turn embedding module
        emb_model = load_model(self.emb__h5, compile=False)
        emb_duration = step * emb_model.input_shape[1]
        self.emb_ = SequenceEmbedding(emb_model, feature_extraction,
                                      emb_duration, step=emb_duration / 4,
                                      internal=self.emb__internal,
                                      batch_size=32, source='audio')

        # initialize clustering module
        self.cls_ = Clustering(metric=self.cls__metric,
                               min_cluster_size=self.cls__min_cluster_size,
                               min_samples=self.cls__min_samples)


    def __call__(self, current_file, return_timing=False):

        # speech activity detection
        soft_sad = self.sad_.apply(current_file)
        hard_sad = self.sad_binarize_.apply(
            soft_sad, dimension=self.sad__dimension)

        # speaker change detection
        soft_scd = self.scd_.apply(current_file)
        hard_scd = self.scd_peak_.apply(
            soft_scd, dimension=self.scd_dimension)

        # speech turns
        speech_turns = hard_scd.crop(hard_sad)

        # speech turns embedding
        emb = self.emb_.apply(current_file)
        fX = np.hstack([
            np.sum(emb.crop(speech_turn, mode='loose'), axis=1)
            for speech_turn in speech_turns
        ])

        # speech turn clustering
        cluster_labels = self.cls_.apply(fX)

        # build hypothesis from clustering results
        hypothesis = Annotation(uri=current_file['uri'])
        for speech_turn, label in zip(speech_turns, cluster_labels):
            hypothesis[speech_turn] = label
        hypothesis = hypothesis.support().rename_labels()

        return hypothesis
