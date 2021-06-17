# The MIT License (MIT)
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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Speaker diarization pipelines"""

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from scipy.special import softmax

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.hierarchy import pool
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform


class SpeakerDiarization(Pipeline):
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
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        embedding: PipelineModel = "pyannote/embedding",
    ):

        super().__init__()

        self.segmentation = segmentation
        self.embedding = embedding

        self.seg_model_: Model = get_model(segmentation)
        self.emb_model_: Model = get_model(embedding)
        self.emb_model_.eval()

        # send models to GPU (when GPUs are available and model is not already on GPU)
        cpu_models = [
            model
            for model in (self.seg_model_, self.emb_model_)
            if model.device.type == "cpu"
        ]
        for cpu_model, gpu_device in zip(
            cpu_models, get_devices(needs=len(cpu_models))
        ):
            cpu_model.to(gpu_device)

        self._segmentation_inference = Inference(self.seg_model_, skip_aggregation=True)

        # hyper-parameters
        self.active_threshold = Uniform(0.05, 0.95)
        self.alone_threshold = Uniform(0.05, 0.95)
        self.cluster_threshold = Uniform(0.0, 2.0)
        self.min_duration_on = 0.0
        self.min_duration_off = 0.0

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
            onset=0.5,
            offset=0.5,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    @staticmethod
    def get_pooling_weights(segmentation: np.ndarray) -> np.ndarray:
        """Overlap-aware weights

        Parameters
        ----------
        segmentation: np.ndarray
            (num_frames, num_speakers) segmentation scores

        Returns
        -------
        weights: np.ndarray
            (num_frames, num_speakers) overlap-aware weights
        """

        power: int = 3
        scale: float = 10.0
        pow_segmentation = pow(segmentation, power)
        return pow_segmentation * pow(softmax(scale * pow_segmentation, axis=1), power)

    @staticmethod
    def get_embedding(
        file: AudioFile,
        chunk: Segment,
        model: Model,
        pooling_weights: np.ndarray = None,
    ):
        """Extract embedding from a chunk

        Parameters
        ----------
        file : AudioFile
        chunk : Segment
        model : Model
            Pretrained embedding model.
        pooling_weights : np.ndarray, optional
            (num_frames, num_speakers) pooling weights

        Returns
        -------
        embeddings : np.ndarray
            (1, dimension) if pooling_weights is None, else (num_speakers, dimension)
        """

        if pooling_weights is None:
            num_speakers = 1

        else:
            _, num_speakers = pooling_weights.shape
            pooling_weights = (
                torch.from_numpy(pooling_weights).float().T.to(model.device)
            )
            # (num_speakers, num_frames)

        waveforms = (
            model.audio.crop(file, chunk)[0]
            .unsqueeze(0)
            .expand(num_speakers, -1, -1)
            .to(model.device)
        )
        # (num_speakers, num_channels == 1, num_samples)

        with torch.no_grad():
            if pooling_weights is None:
                embeddings = model(waveforms)
            else:
                embeddings = model(waveforms, weights=pooling_weights)

        return embeddings.cpu().numpy()

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

        frames: SlidingWindow = self._segmentation_inference.model.introspection.frames
        segmentations: SlidingWindowFeature = self._segmentation_inference(file)
        num_chunks, num_frames, num_speakers = segmentations.data.shape

        embeddings = []
        active = []
        alone = []

        for c, (chunk, segmentation) in enumerate(segmentations):

            pooling_weights = self.get_pooling_weights(segmentation)
            # (num_frames, num_speakers)

            try:
                chunk_embeddings = self.get_embedding(
                    file, chunk, self.emb_model_, pooling_weights=pooling_weights
                )
                # num_speakers, dimension
            except ValueError:
                # happens with last chunk that is too long for audio...
                continue

            # remember if speaker is active
            active.append(np.any(segmentation > self.active_threshold, axis=0))

            # remember if speaker is active without overlap
            alone.append(np.mean(pooling_weights, axis=0) > self.alone_threshold)

            old_norm = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
            # (num_speakers, 1)

            new_norm = np.mean(segmentation, axis=0, keepdims=True).T
            # (num_speakers, 1)

            embeddings.append(chunk_embeddings * (new_norm / old_norm))

        active = np.hstack(active)
        alone = np.hstack(alone)
        embeddings = np.vstack(embeddings)

        # clusters[chunk_id x num_speakers + speaker_id] = ...
        # -1 if speaker is inactive
        # k if speaker is active and is assigned to cluster k
        clusters = -np.ones(len(embeddings), dtype=np.int)

        # hierarchical agglomerative clustering with "pool" linkage

        if np.sum(alone * active) < 2:
            clusters[active] = 0
            num_clusters = 1
        else:
            Z = pool(
                embeddings[alone * active],
                metric="cosine",
                pooling_func=self.pooling_func,
            )
            file["@diarization/clustering"] = Z

            clusters[alone * active] = (
                fcluster(Z, self.cluster_threshold, criterion="distance") - 1
            )
            num_clusters = np.max(clusters) + 1

            centroids = np.vstack(
                [
                    np.sum(
                        embeddings[active * alone][clusters[active * alone] == k],
                        axis=0,
                    )
                    for k in range(num_clusters)
                ]
            )
            file["@diarization/centroids"] = centroids

            # assign remaining chunks to closest centroid
            clusters[active * ~alone] = np.argmin(
                cdist(centroids, embeddings[active * ~alone], metric="cosine"), axis=0
            )

        clusters = clusters.reshape(-1, num_speakers)

        clustered_segmentations = np.zeros((num_chunks, num_frames, num_clusters))
        for c, (cluster, (chunk, segmentation)) in enumerate(
            zip(clusters, segmentations)
        ):
            for k in range(num_speakers):
                if cluster[k] == -1:
                    pass
                clustered_segmentations[c, :, cluster[k]] = segmentation[:, k]

        speaker_activations = Inference.aggregate(
            SlidingWindowFeature(clustered_segmentations, segmentations.sliding_window),
            frames,
        )
        file["@diarization/activations"] = speaker_activations

        active_speaker_count = Inference.aggregate(
            np.sum(segmentations > self.active_threshold, axis=-1, keepdims=True),
            frames,
        )
        active_speaker_count.data = np.round(active_speaker_count)
        file["@diarization/speaker_count"] = active_speaker_count

        sorted_speakers = np.argsort(-speaker_activations, axis=-1)
        binarized = np.zeros_like(speaker_activations.data)
        for t, ((_, count), speakers) in enumerate(
            zip(active_speaker_count, sorted_speakers)
        ):
            # TODO: find a way to stop clustering early enough to avoid num_clusters < count
            count = min(num_clusters, int(count.item()))
            for i in range(count):
                binarized[t, speakers[i]] = 1.0

        diarization = self._binarize(SlidingWindowFeature(binarized, frames))
        diarization.uri = file["uri"]

        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
