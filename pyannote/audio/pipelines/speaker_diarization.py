import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from scipy.special import softmax

from pyannote.audio import Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.segmentation import Segmentation
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, Segment, SlidingWindowFeature
from pyannote.core.utils.hierarchy import pool
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform


class SegmentationBasedSpeakerDiarization(Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    segmentation : Inference or str, optional
        `Inference` instance used to extract raw segmentation scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "seg".
    embeddings : Inference or str, optional
        `Inference` instance used to extract speaker embeddings. When `str`,
        assumes that file already contains a corresponding key with precomputed
        embeddings. Defaults to "emb".

    Hyper-parameters
    ----------------
    clean_speech_threshold : float
    clustering_threshold : float
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        embedding: PipelineModel = "hbredin/SpeakerEmbedding-XVectorMFCC-VoxCeleb",
    ):

        super().__init__()

        self.segmentation = segmentation
        self.embedding = embedding

        segmentation_model: Model = get_model(segmentation)
        self.emb_model_: Model = get_model(embedding)
        self.emb_model_.eval()

        # send models to GPU (when GPUs are available and model is not already on GPU)
        cpu_models = [
            model
            for model in (segmentation_model, self.emb_model_)
            if model.device.type == "cpu"
        ]
        for cpu_model, gpu_device in zip(
            cpu_models, get_devices(needs=len(cpu_models))
        ):
            cpu_model.to(gpu_device)

        self.audio_ = self.emb_model_.audio

        self.segmentation_pipeline = Segmentation(
            segmentation=segmentation_model, return_activation=True
        )

        #  hyper-parameters
        self.clean_speech_threshold = Uniform(0, 1)
        self.clustering_threshold = Uniform(0, 2.0)

        # borrowed from self.segmentation_pipeline
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

    @staticmethod
    def pooling_func(
        u: int,
        v: int,
        C: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute average of newly merged cluster

        Parameters
        ----------
        u : int
            Cluster index.
        v : int
            Cluster index.
        C : (2 x n_observations - 1, dimension) np.ndarray
            Cluster embedding.

        Returns
        -------
        Cuv : (dimension, ) np.ndarray
            Embedding of newly formed cluster.
        """
        return C[u] + C[v]

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    def apply(self, file: AudioFile) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        activations = self.segmentation_pipeline(file)
        frames = activations.sliding_window
        num_frames_in_file, num_subclusters = activations.data.shape

        SEGMENTATION_POWER = 3
        SEGMENTATION_SCALE = 10
        data = activations.data ** SEGMENTATION_POWER
        data = data * softmax(SEGMENTATION_SCALE * data, axis=1) ** SEGMENTATION_POWER
        clean_speech = SlidingWindowFeature(data, frames)
        # clean_speech[f, k] is close to 1 when model is confident that kth sub-cluster
        # is active at fth frame and not overlapping with other speakers, and close to 0
        # in any other situation.
        # SHAPE (num_frames, num_subclusters)

        file["@diarization/clean_speech"] = clean_speech

        total_clean_speech = np.sum(clean_speech.data, axis=0) * frames.step
        # total_clean_speech[k] is the total duration of clean speech for kth sub-cluster
        # SHAPE (num_subclusters)
        has_enough_clean_speech = total_clean_speech > self.clean_speech_threshold
        # has_enough_clean_speech[k] indicates whether kth sub-cluster has enough clean
        # speech to be used in embedding-based clustering

        emb_device = self.emb_model_.device
        dimension = self.emb_model_.introspection.dimension
        embeddings = np.empty((num_subclusters, dimension))
        # SHAPE (num_subclusters, embedding_dimension)

        for k in range(num_subclusters):

            # first and last frame/time where sub-cluster is active
            start_frame, end_frame = np.where(activations.data[:, k] > 0)[0][[0, -1]]
            start_time, end_time = frames[start_frame].middle, frames[end_frame].middle

            chunk = Segment(start_time, end_time)
            waveforms = self.audio_.crop(file, chunk)[0].unsqueeze(0).to(emb_device)
            # SHAPE (1, 1, num_samples)

            weights = (
                torch.from_numpy(
                    clean_speech.crop(chunk)[:, k].reshape(1, -1),
                )
                .float()
                .to(emb_device)
            )
            # SHAPE (1, num_frames)

            with torch.no_grad():
                emb = (
                    F.normalize(
                        self.emb_model_(waveforms, weights=weights),
                        p=2.0,
                        dim=1,
                        eps=1e-12,
                    )
                    * weights.sum()
                ).squeeze()

                embeddings[k] = emb.cpu().numpy() * total_clean_speech[k]

        file["@diarization/embeddings"] = embeddings
        file["@diarization/total_clean_speech"] = total_clean_speech

        # TODO: better handle corner case with not enough clean embeddings
        if np.sum(has_enough_clean_speech) < 2:
            return Annotation(uri=file["uri"])

        # hierarchical agglomerative clustering with "pool" linkage
        embeddings_for_clustering = embeddings[has_enough_clean_speech]
        Z = pool(
            embeddings_for_clustering,
            metric="cosine",
            pooling_func=self.pooling_func,
            # cannot_link=cannot_link
        )

        file["@diarization/dendrogram"] = Z

        clean_clusters = (
            fcluster(Z, self.clustering_threshold, criterion="distance") - 1
        )

        # one representative embedding per clean cluster
        num_clusters = len(np.unique(clean_clusters))
        clean_cluster_embeddings = np.vstack(
            [
                np.sum(
                    embeddings[has_enough_clean_speech][clean_clusters == cluster],
                    axis=0,
                )
                for cluster in range(num_clusters)
            ]
        )
        noisy_clusters = np.argmin(
            cdist(
                clean_cluster_embeddings,
                embeddings[~has_enough_clean_speech],
                metric="cosine",
            ),
            axis=0,
        )

        aggregated = np.zeros((num_frames_in_file, num_clusters))

        for cluster, k in zip(clean_clusters, *np.where(has_enough_clean_speech)):
            aggregated[:, cluster] = np.maximum(
                aggregated[:, cluster], activations.data[:, k]
            )

        for cluster, k in zip(noisy_clusters, *np.where(~has_enough_clean_speech)):
            aggregated[:, cluster] = np.maximum(
                aggregated[:, cluster], activations.data[:, k]
            )

        clustered_activations = SlidingWindowFeature(aggregated, frames)

        file["@diarization/activations"] = clustered_activations
        diarization = self._binarize(clustered_activations)
        diarization.uri = file["uri"]
        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
