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
from scipy.spatial.distance import squareform
from scipy.special import softmax
from spectralcluster import (
    ICASSP2018_REFINEMENT_SEQUENCE,
    AutoTune,
    RefinementOptions,
    SpectralClusterer,
    ThresholdType,
    constraint,
)

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.distance import pdist
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Categorical, Uniform


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
        self.onset = Uniform(0.05, 0.95)
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

        self.use_auto_tune = Categorical([True, False])
        self.use_refinement = Categorical([True, False])
        self.use_constraints = Categorical([True, False])

        self.use_overlap_aware_embedding = Categorical([True, False])

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        if self.use_auto_tune:
            self._autotune = AutoTune(
                p_percentile_min=0.60,
                p_percentile_max=0.95,
                init_search_step=0.01,
                search_level=3,
            )
        else:
            self._autotune = None

        if self.use_refinement:
            self._refinement_options = RefinementOptions(
                gaussian_blur_sigma=1,
                p_percentile=0.95,
                thresholding_soft_multiplier=0.01,
                thresholding_type=ThresholdType.RowMax,
                refinement_sequence=ICASSP2018_REFINEMENT_SEQUENCE,
            )
        else:
            self._refinement_options = None

        if self.use_constraints:
            ConstraintName = constraint.ConstraintName
            self._constraint_options = constraint.ConstraintOptions(
                constraint_name=ConstraintName.ConstraintPropagation,
                apply_before_refinement=True,
                constraint_propagation_alpha=0.6,
            )
        else:
            self._constraint_options = None

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
    ) -> np.ndarray:
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

    CACHED_SEGMENTATION = "@speaker_diarization/segmentation"
    CACHED_EMBEDDING = "@speaker_diarization/embedding"

    def apply(self, file: AudioFile, expected_num_speakers: int = None) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.
        expected_num_speakers : int, optional
            Expected number of speakers. Defaults to estimate it automatically.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, num_speakers)
        if (not self.training) or (
            self.training and self.CACHED_SEGMENTATION not in file
        ):
            file[self.CACHED_SEGMENTATION] = self._segmentation_inference(file)
        segmentations: SlidingWindowFeature = file[self.CACHED_SEGMENTATION]
        num_chunks, num_frames, num_speakers = segmentations.data.shape

        # extract embeddings (only if needed)
        # output shape is (num_valid_chunks x num_speakers, embedding_dimension)
        if (not self.training) or (self.training and self.CACHED_EMBEDDING not in file):

            embeddings = []

            for c, (chunk, segmentation) in enumerate(segmentations):

                if self.use_overlap_aware_embedding:
                    pooling_weights: np.ndarray = self.get_pooling_weights(segmentation)
                    # (num_frames, num_speakers)
                else:
                    pooling_weights: np.ndarray = segmentation
                    # (num_frames, num_speakers)

                try:
                    chunk_embeddings: np.ndarray = self.get_embedding(
                        file, chunk, self.emb_model_, pooling_weights=pooling_weights
                    )
                    # (num_speakers, dimension)

                except ValueError:
                    if c + 1 == num_chunks:
                        # it might happen that one cannot extract embeddings from
                        # the very last chunk because of audio duration.
                        continue
                    else:
                        # however, if we fail in the middle of the file, something
                        # bad has happened and we should not go any further...
                        raise ValueError()

                embeddings.append(chunk_embeddings)

            embeddings = np.vstack(embeddings)
            # (num_valid_chunks x num_speakers, dimension)

            # unit-normalize embeddings
            old_norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            new_norm = np.ones((len(embeddings), 1))
            embeddings = embeddings * (new_norm / old_norm)

            # TODO. add support for duration-weighted normalization
            # new_norm = f(np.mean(segmentations, axis=1, keepdims=True))

            file[self.CACHED_EMBEDDING] = embeddings

        embeddings = file[self.CACHED_EMBEDDING]
        # update number of chunks (only those with embeddings)
        num_chunks = int(embeddings.shape[0] / num_speakers)
        segmentations.data = segmentations.data[:num_chunks]

        frames: SlidingWindow = self._segmentation_inference.model.introspection.frames
        # frame resolution (e.g. duration = step = 17ms)

        # active.data[c, k] indicates whether kth speaker is active in cth chunk
        active: np.ndarray = np.any(segmentations > self.onset, axis=1).data
        # (num_chunks, num_speakers)
        num_active = np.sum(active)

        # clusters[chunk_id x num_speakers + speaker_id] = ...
        # -1 if speaker is inactive
        # k if speaker is active and is assigned to cluster k
        clusters = -np.ones(len(embeddings), dtype=np.int)

        if num_active < 2:
            clusters[active.reshape(-1)] = 0
            num_clusters = 1

        else:
            active_embeddings = embeddings[active.reshape(-1)]

            if expected_num_speakers is None:
                max_active_in_same_chunk = np.max(np.sum(active, axis=1))
                min_clusters = max(1, max_active_in_same_chunk)
                max_clusters = 20
            else:
                min_clusters = expected_num_speakers
                max_clusters = expected_num_speakers

            clusterer = SpectralClusterer(
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                autotune=self._autotune,
                laplacian_type=None,
                refinement_options=self._refinement_options,
                constraint_options=self._constraint_options,
                custom_dist="cosine",
            )

            if self.use_constraints:

                chunk_idx = np.broadcast_to(
                    np.arange(num_chunks), (num_speakers, num_chunks)
                ).T.reshape(-1)[active.reshape(-1)]

                cannot_link = squareform(-1.0 * pdist(chunk_idx, metric="equal"))
                clusters[active.reshape(-1)] = clusterer.predict(
                    active_embeddings, cannot_link
                )

            else:
                clusters[active.reshape(-1)] = clusterer.predict(active_embeddings)

            num_clusters = np.max(clusters) + 1

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
            np.sum(segmentations > self.onset, axis=-1, keepdims=True),
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
