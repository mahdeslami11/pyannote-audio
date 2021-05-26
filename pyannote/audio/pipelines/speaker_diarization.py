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

from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster
from scipy.special import softmax

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.activations import warmup_activations
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.hierarchy import pool
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import LogUniform, Uniform


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

        self.audio_ = self.emb_model_.audio

        self.warmup_ratio = Uniform(0.0, 0.2)
        self.duration_threshold = Uniform(0.0, 1.0)
        self.duration_scale = LogUniform(1e-2, 1e2)
        # sigmoid(scale * (d - threshold))
        self.distance_threshold = Uniform(0.0, 2.0)

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

        duration = self.seg_model_.specifications.duration
        step = duration - 2 * self.warmup_ratio * duration
        self._segmentation_inference = Inference(
            self.seg_model_, duration=duration, step=step, skip_aggregation=True
        )

        self._binarize = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=0.1,
            min_duration_off=0.2,
        )

    def get_weights(self, activations: SlidingWindowFeature) -> SlidingWindowFeature:
        # TODO: make those hyper-parameters?
        power: int = 3
        scale: float = 10.0
        pow_activations = pow(activations, power)
        return pow_activations * pow(softmax(scale * pow_activations, axis=1), power)

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

        duration = self.seg_model_.specifications.duration
        segmentations = warmup_activations(
            segmentations, warm_up=self.warmup_ratio * duration
        )

        num_chunks, num_frames, num_speakers = segmentations.data.shape

        embeddings = []
        cannot_link = []
        device = self.emb_model_.device
        for c, (chunk, segmentation) in enumerate(segmentations):

            waveforms: torch.Tensor = (
                self.audio_.crop(file, chunk)[0]
                .unsqueeze(0)
                .expand(num_speakers, -1, -1)
                .to(device)
            )
            # num_speakers, num_channels == 1, num_samples

            embedding_weights: torch.Tensor = (
                torch.from_numpy(self.get_weights(segmentation)).float().T.to(device)
            )
            # (num_speakers, num_frames)

            clustering_weights: torch.Tensor = F.sigmoid(
                self.duration_scale
                * (
                    torch.mean(embedding_weights, dim=1, keepdim=True)
                    - self.duration_threshold
                )
            )
            # num_speakers, 1

            with torch.no_grad():
                chunk_embeddings = clustering_weights * F.normalize(
                    self.emb_model_(waveforms, weights=embedding_weights)
                )
                # num_speakers, dimension

            embeddings.append(chunk_embeddings.cpu().numpy())

            for i, j in combinations(
                range(num_speakers * c, num_speakers * (c + 1)), 2
            ):
                cannot_link.append((i, j))

        embeddings = np.vstack(embeddings)

        # FIXME -- why do we need this +100 ?
        num_frames_in_file = (
            frames.samples(self.audio_.get_duration(file), mode="center") + 100
        )

        # hierarchical agglomerative clustering with "pool" linkage
        dendrogram = pool(
            embeddings,
            metric="cosine",
            pooling_func=self.pooling_func,
            cannot_link=cannot_link,
        )
        file["@diarization/dendrogram"] = dendrogram

        clusters = (
            fcluster(dendrogram, self.distance_threshold, criterion="distance") - 1
        )
        num_clusters = np.max(clusters) + 1

        aggregated = np.zeros((num_frames_in_file, num_clusters))

        for k, cluster in enumerate(clusters):

            chunk_idx = k // num_speakers
            chunk = segmentations.sliding_window[chunk_idx]

            speaker_idx = k % num_speakers
            activation = segmentations[chunk_idx][:, speaker_idx]

            start_frame = frames.closest_frame(chunk.start)
            end_frame = start_frame + len(activation)

            aggregated[start_frame:end_frame, cluster] = np.maximum(
                activation, aggregated[start_frame:end_frame, cluster]
            )

        activations = SlidingWindowFeature(aggregated, frames)

        file["@diarization/activations"] = activations

        diarization = self._binarize(activations)
        diarization.uri = file["uri"]
        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
