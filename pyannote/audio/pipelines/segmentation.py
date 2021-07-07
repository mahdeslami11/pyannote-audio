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

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature, Timeline
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform


class Segmentation(Pipeline):
    """Segmentation pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
    ):
        super().__init__()

        self.segmentation = segmentation

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
        self.consistency_threshold = Uniform(0.0, 1.0)

    def cost_func(self, Y: np.ndarray, y: np.ndarray) -> float:
        return np.mean(
            np.mean(
                (Y > self.activity_threshold) != (y > self.activity_threshold), axis=-1
            ),
            axis=-1,
        )

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
        # group and aggregate consistent adjacent chunks
        # permutate speakers in order to maximize consistency
        clusters = np.zeros((num_chunks,))
        current_cluster = 0
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
                cost_func=self.cost_func,
                return_cost=True,
            )

            for prev_i, i in enumerate(permutation):
                segmentations.data[c, :, prev_i] = segmentation[:, i]

            consistency = np.mean(
                [cost[prev_i, i] for prev_i, i in enumerate(permutation)]
            )

            if consistency > self.consistency_threshold:
                current_cluster += 1

            clusters[c] = current_cluster

            previous_frame_index = frame_index

        num_clusters = current_cluster + 1

        activations = np.zeros((num_chunks, num_frames, num_speakers * num_clusters))
        for c, (k, (_, permutated_segmentation)) in enumerate(
            zip(clusters, segmentations)
        ):
            activations[
                c, :, num_speakers * k : num_speakers * (k + 1)
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
