# The MIT License (MIT)
#
# Copyright (c) 2017-2021 CNRS
#
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


import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from scipy.special import softmax

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.hierarchy import pool
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform

# import networkx as nx


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

        # output frames as SlidingWindow instances
        self.seg_frames_: SlidingWindow = segmentation_model.introspection.frames

        # prepare segmentation model for inference
        self.seg_inference_ = Inference(
            segmentation_model,
            step=segmentation_model.specifications.duration * 0.1,
            skip_aggregation=True,
        )

        #  hyper-parameters
        self.clean_speech_threshold = Uniform(0, 1)
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

        # graph = nx.Graph()

        SEGMENTATION_POWER = 3
        SEGMENTATION_SCALE = 10

        emb_device = self.emb_model_.device

        # output of segmentation model on each chunk
        segmentations: SlidingWindowFeature = self.seg_inference_(file)
        # SHAPE (num_chunks, num_frames, num_speakers)

        (
            num_chunks,
            num_frames_per_chunk,
            num_speakers_per_chunk,
        ) = segmentations.data.shape

        is_active = np.max(segmentations.data, axis=1) > self.onset
        # is_active[c, k] indicates whether kth speaker in cth chunk is active
        # SHAPE (num_chunks, num_speakers_per_chunk)

        data = segmentations.data ** SEGMENTATION_POWER
        clean_speech = (
            data * softmax(SEGMENTATION_SCALE * data, axis=2) ** SEGMENTATION_POWER
        )
        # clean_speech[c, f, k] should be close to 1 when model is confident that
        # kth speaker in cth chunk is active at fth frame and not overlapping with
        # other speakers, and should be close to 0 in any other situation.
        # SHAPE (num_chunks, num_frames, num_speakers_per_chunk)

        average_clean_speech = np.mean(clean_speech, axis=1)
        # average_clean_speech[c, k] contains the ratio of clean (i.e. confident and
        # non-overlapping) speech over chunk duration for kth speaker in cth chunk.
        # SHAPE (num_chunks, num_speakers_per_chunk)

        has_enough_clean_speech = average_clean_speech > self.clean_speech_threshold
        # has_enough_clean_speech[c, k] indicates whether kth speaker in cth chunk
        # has enough clean speech to be used in embedding-based clustering
        # SHAPE (num_chunks, num_speakers_per_chunk)

        embeddings = np.empty(
            (
                num_chunks,
                num_speakers_per_chunk,
                self.emb_model_.introspection.dimension,
            )
        )
        # SHAPE (num_chunks, num_speakers_per_chunk, embedding_dimension)

        hamming = np.hamming(num_frames_per_chunk)

        for c, (chunk, segmentation) in enumerate(segmentations):

            # compute embeddings
            with torch.no_grad():

                # read audio chunk
                waveforms = (
                    self.audio_.crop(file, chunk)[0]
                    .unsqueeze(0)
                    .repeat(num_speakers_per_chunk, 1, 1)
                    .to(emb_device)
                )
                # SHAPE (num_speakers_per_chunk, 1, num_samples)

                # give more weights to regions where speaker is "clean"
                weights = torch.from_numpy(clean_speech[c].T).to(emb_device)
                # SHAPE (num_speakers_per_chunk, num_frames)

                # extract embedding and make its norm proportional to the amount of "clean" speech
                emb = F.normalize(
                    self.emb_model_(waveforms, weights=weights), p=2.0, dim=1, eps=1e-12
                ).cpu()
                emb *= average_clean_speech[c].reshape(-1, 1)
                # SHAPE (num_speakers_per_chunk, embedding_dimension)

            embeddings[c] = emb

        # TODO: better handle corner case with not enough clean embeddings
        if np.sum(has_enough_clean_speech) < 2:
            return Annotation(uri=file["uri"])

        # hierarchical agglomerative clustering with "pool" linkage
        Z = pool(
            embeddings[has_enough_clean_speech],
            metric="cosine",
            pooling_func=self.pooling_func,
            # cannot_link=cannot_link
        )
        clean_clusters = (
            fcluster(Z, self.clustering_threshold, criterion="distance") - 1
        )

        # one representative embedding per cluster (computed as the same of )
        num_clusters = len(np.unique(clean_clusters))
        cluster_embeddings = np.vstack(
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
                cluster_embeddings,
                embeddings[is_active & ~has_enough_clean_speech],
                metric="cosine",
            ),
            axis=0,
        )

        # number of frames in the whole file
        num_frames_in_file = self.seg_frames_.samples(
            self.audio_.get_duration(file), mode="center"
        )

        aggregated = np.zeros((num_frames_in_file, num_clusters))
        overlapped = np.zeros((num_frames_in_file, num_clusters))

        chunks = segmentations.sliding_window

        for cluster, c, k in zip(clean_clusters, *np.where(has_enough_clean_speech)):

            start_frame = self.seg_frames_.closest_frame(chunks[c].start)
            aggregated[start_frame : start_frame + num_frames_per_chunk, cluster] += (
                segmentations.data[c, :, k] * hamming
            )

            # remember how many chunks were added on this particular speaker
            overlapped[
                start_frame : start_frame + num_frames_per_chunk, cluster
            ] += hamming

        for cluster, c, k in zip(
            noisy_clusters, *np.where(is_active & ~has_enough_clean_speech)
        ):

            start_frame = self.seg_frames_.closest_frame(chunks[c].start)
            aggregated[start_frame : start_frame + num_frames_per_chunk, cluster] += (
                segmentations.data[c, :, k] * hamming
            )

            # remember how many chunks were added on this particular speaker
            overlapped[
                start_frame : start_frame + num_frames_per_chunk, cluster
            ] += hamming

        speaker_activations = SlidingWindowFeature(
            aggregated / (overlapped + 1e-12), self.seg_frames_
        )

        diarization = self._binarize(speaker_activations)
        diarization.uri = file["uri"]
        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
