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

"""Segmentation pipeline"""


import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.permutation import mad_cost_func, mse_cost_func, permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature, Timeline
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform


class Segmentation(Pipeline):
    """Segmentation pipeline

    Parameters
    ----------
    segmentation :
    consistency_metric : {"mse", "mad"}

    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        consistency_metric: str = "mse",
    ):
        super().__init__()

        self.segmentation = segmentation
        self.consistency_metric = consistency_metric

        model: Model = get_model(segmentation)
        if model.device.type == "cpu":
            (gpu_device,) = get_devices(needs=1)
            model.to(gpu_device)
        self._segmentation_inference = Inference(model, skip_aggregation=True)

        self._warm_up = (
            self._segmentation_inference.step,
            self._segmentation_inference.step,
        )
        self._binarize = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=0.0,
            min_duration_off=0.0,
        )

        # hyper-parameters

        self.activity_threshold = Uniform(0.05, 0.95)

        if consistency_metric == "mse":
            self._cost_func = mse_cost_func
            self.consistency_threshold = Uniform(0.0, 1.0)
        elif consistency_metric == "mad":
            self._cost_func = mad_cost_func
            self.consistency_threshold = Uniform(0.0, 1.0)
        else:
            raise ValueError('"consistency_metric" must be one of {"mse", "mad"}.')

    def apply(self, file: AudioFile) -> Annotation:

        frames: SlidingWindow = self._segmentation_inference.model.introspection.frames
        segmentations: SlidingWindowFeature = self._segmentation_inference(file)
        chunks: SlidingWindow = segmentations.sliding_window
        num_chunks, num_frames, num_speakers = segmentations.data.shape

        # instantaneous speaker count
        speaker_count = Inference.aggregate(
            np.sum(segmentations > self.activity_threshold, axis=-1, keepdims=True),
            frames,
            warm_up=self._warm_up,
        )
        speaker_count.data = np.round(speaker_count)
        file["@segmentation/speaker_count"] = speaker_count
        # TODO: apply binarize on each chunk separatly to benefit from onset/offset
        # TODO: maybe np.floor is better? <== optimize that as well

        # compute consistency between each pair of adjacent chunks
        # and permutate speakers in order to maximize consistency
        consistency = np.ones((num_chunks, num_chunks))
        for c in range(num_chunks):

            chunk = chunks[c]
            frame_index = frames.closest_frame(chunk.start)

            if c == 0:
                previous_frame_index = frame_index
                continue

            segmentation = segmentations[c].copy()
            frame_shift = frame_index - previous_frame_index

            _, (permutation,), (cost,) = permutate(
                segmentations[
                    np.newaxis,
                    c - 1,
                    2 * frame_shift : num_frames - frame_shift,
                ],
                segmentation[frame_shift : num_frames - 2 * frame_shift],
                cost_func=self._cost_func,
                return_cost=True,
            )

            for prev_i, i in enumerate(permutation):
                segmentations.data[c, :, prev_i] = segmentation[:, i]

            consistency[c - 1, c] = max(
                cost[prev_i, i] for prev_i, i in enumerate(permutation)
            )

            previous_frame_index = frame_index

        # group and aggregate consistent adjacent chunks
        consistency = squareform(consistency, checks=False)
        Z = linkage(consistency, method="single", metric="precomputed")
        clusters = fcluster(Z, self.consistency_threshold, criterion="distance")
        num_clusters = np.max(clusters)

        activations = np.zeros((num_chunks, num_frames, num_speakers * num_clusters))
        for c, (k, (_, permutated_segmentation)) in enumerate(
            zip(clusters, segmentations)
        ):
            activations[
                c, :, num_speakers * (k - 1) : num_speakers * k
            ] = permutated_segmentation

        activations = SlidingWindowFeature(activations, chunks)
        activations = Inference.aggregate(activations, frames, warm_up=self._warm_up)
        file["@segmentation/activations"] = activations

        # use speaker count to only keep as many speakers as needed
        sorted_speakers = np.argsort(-activations, axis=-1)
        binary_activations = np.zeros_like(activations.data)
        for t, ((_, count), speakers) in enumerate(zip(speaker_count, sorted_speakers)):
            count = int(count.item())
            for i in range(count):
                binary_activations[t, speakers[i]] = 1.0

        # turn binary activations into hard Annotation
        final_segmentation = self._binarize(
            SlidingWindowFeature(binary_activations, frames)
        )
        final_segmentation.uri = file["uri"]

        return final_segmentation

    def loss(self, file: AudioFile, hypothesis: Annotation) -> float:

        metric = GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
        duration = self._segmentation_inference.duration
        for segment in file["annotated"]:
            chunks = SlidingWindow(duration=2.0 * duration, step=0.5 * duration)
            for chunk in chunks(segment):
                _ = metric(
                    file["annotation"], hypothesis, uem=Timeline(segments=[chunk])
                )
        return abs(metric)
