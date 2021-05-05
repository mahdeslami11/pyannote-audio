# MIT License
#
# Copyright (c) 2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster
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

        self.clustering_threshold = Uniform(0, 2.0)
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

    def get_weights(self, activations: SlidingWindowFeature) -> SlidingWindowFeature:
        # TODO: make those hyper-parameters?
        power: int = 3
        scale: float = 10.0
        pow_activations = pow(activations, power)
        return pow_activations * pow(softmax(scale * pow_activations), power)

    def get_embeddings(
        self, file: AudioFile, weights: SlidingWindowFeature
    ) -> np.ndarray:
        """

        Parameters
        ----------
        file : AudioFile
            Audio file
        weights : SlidingWindowFeature
            (num_frames, num_clusters) activations

        Returns
        -------
        embeddings : np.ndarray
            (num_clusters, num_dimension) embeddings.
        """

        num_clusters = weights.data.shape[1]
        model = self.emb_model_
        device = model.device
        duration = self.audio_.get_duration(file)

        embeddings = []

        for k in range(num_clusters):

            # find region where cluster is active
            first_frame, last_frame = np.where(weights.data[:, k] > 0)[0][[0, -1]]
            chunk = Segment(
                max(0, weights.sliding_window[first_frame].middle),
                min(duration, weights.sliding_window[last_frame].middle),
            )

            # extract corresponding audio
            waveforms = self.audio_.crop(file, chunk)[0].unsqueeze(0).to(device)
            # (1, 1, num_samples) tensor

            # extract corresponding weight
            cluster_weight = weights.crop(chunk)[:, k].reshape(1, -1)
            cluster_weight = torch.from_numpy(cluster_weight).float().to(device)
            # (1, num_frames) tensor

            # compute embedding
            with torch.no_grad():
                embedding = model(waveforms, weights=cluster_weight)
                embedding = F.normalize(embedding) * cluster_weight.sum()
                # (1, num_dimensions) tensor

            embeddings.append(embedding.cpu().numpy())

        return np.vstack(embeddings)
        # (num_clusters, num_dimensions) ndarray

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

        if num_subclusters < 2:
            diarization = self._binarize(activations)
            diarization.uri = file["uri"]
            return diarization

        weights: SlidingWindowFeature = self.get_weights(activations)
        file["@diarization/weights"] = weights

        embeddings: np.ndarray = self.get_embeddings(file, weights)
        file["@diarization/embeddings"] = embeddings

        # hierarchical agglomerative clustering with "pool" linkage
        Z = pool(
            embeddings,
            metric="cosine",
            pooling_func=self.pooling_func,
        )
        file["@diarization/dendrogram"] = Z

        clusters = fcluster(Z, self.clustering_threshold, criterion="distance") - 1
        num_clusters = len(clusters)

        aggregated = np.zeros((num_frames_in_file, num_clusters))
        for k, cluster in enumerate(clusters):
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
