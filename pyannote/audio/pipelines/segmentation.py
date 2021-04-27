# MIT License
#
# Copyright (c) 2020-2021 CNRS
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

"""Segmentation pipelines"""

import networkx as nx
import numpy as np

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform


class OracleSegmentation(Pipeline):
    """Oracle segmentation pipeline"""

    def apply(self, file: AudioFile) -> Annotation:
        """Return groundtruth segmentation

        Parameter
        ---------
        file : AudioFile
            Must provide a "annotation" key.

        Returns
        -------
        hypothesis : `pyannote.core.Annotation`
            Segmentation
        """

        return file["annotation"].relabel_tracks(generator="string")


class Segmentation(Pipeline):
    """Segmentation pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    return_activation : bool, optional
        Return soft speaker activation instead of hard segmentation.
        Defaults to False (i.e. hard segmentation).
    inference_kwargs : dict, optional
        Keywords arguments passed to Inference (e.g. batch_size, progress_hook).

    Hyper-parameters
    ----------------
    onset, offset : float
        Onset/offset detection thresholds
    min_duration_on : float
        Remove speaker turn shorter than that many seconds.
    min_duration_off : float
        Fill same-speaker gaps shorter than that many seconds.
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        return_activation: bool = False,
        **inference_kwargs,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.return_activation = return_activation

        # load model and send it to GPU (when available and not already on GPU)
        model: Model = get_model(segmentation)
        if model.device.type == "cpu":
            (device,) = get_devices(needs=1)
            model.to(device)

        inference_kwargs["duration"] = model.specifications.duration
        inference_kwargs["step"] = model.specifications.duration * 0.1
        inference_kwargs["skip_aggregation"] = True

        self.segmentation_inference_ = Inference(model, **inference_kwargs)

        # a speaker is active if its activation is greater than `onset`
        # for at least one frame within a chunk`
        self.activity_threshold = Uniform(0.0, 1.0)

        # TODO: add confidence threshold to only map chunks on which
        # segmentation model is confident
        # self.confidence_threshold = Uniform(0.0, 1.0)

        # mapped speakers activations between two consecutive chunks
        # are said to be consistent if
        self.consistency_threshold = Uniform(0.0, 1.0)

        if not self.return_activation:

            #  hyper-parameter used for hysteresis thresholding
            # (in combination with onset := activity_threshold)
            self.offset = Uniform(0.0, 1.0)

            # hyper-parameters used for post-processing i.e. removing short speech turns
            # or filling short gaps between speech turns of one speaker
            self.min_duration_on = Uniform(0.0, 1.0)
            self.min_duration_off = Uniform(0.0, 1.0)

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        if not self.return_activation:

            self._binarize = Binarize(
                onset=self.activity_threshold,
                offset=self.offset,
                min_duration_on=self.min_duration_on,
                min_duration_off=self.min_duration_off,
            )

    def apply(self, file: AudioFile) -> Annotation:
        """Apply segmentation

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        segmentation : `pyannote.core.Annotation`
            Segmentation
        """

        # TODO: read it from model?
        # TODO: hardcode it as f(step) to make sure we have enough data for aggregation
        WARMUP_RATIO = 0.1

        raw_activations: SlidingWindowFeature = self.segmentation_inference_(file)

        num_chunks, raw_num_frames_per_chunk, num_speakers = raw_activations.data.shape
        duration: float = self.segmentation_inference_.duration
        step: float = self.segmentation_inference_.step
        file_frames: SlidingWindow = (
            self.segmentation_inference_.model.introspection.frames
        )

        # remove warm-up regions from raw activation
        warmup_num_frames: int = round(WARMUP_RATIO * raw_num_frames_per_chunk)
        warmup_duration: float = WARMUP_RATIO * duration
        data: np.ndarray = raw_activations.data[
            :, warmup_num_frames : raw_num_frames_per_chunk - warmup_num_frames
        ]
        num_frames_per_chunk: int = raw_num_frames_per_chunk - 2 * warmup_num_frames
        num_frames_per_step: int = file_frames.duration_to_samples(step)

        raw_chunks: SlidingWindow = raw_activations.sliding_window
        chunks = SlidingWindow(
            start=raw_chunks.start + warmup_duration,
            duration=raw_chunks.duration - 2 * warmup_duration,
            step=raw_chunks.step,
        )
        activations = SlidingWindowFeature(data, chunks)

        # build (chunk, speaker) consistency graph
        #   - (c, s) node indicates that sth speaker of cth chunk is active
        #   - (c, s) == (c+1, s') edge indicates that sth speaker of cth chunk
        #     is mapped to s'th speaker of (c+1)th chunk

        consistency_graph = nx.Graph()
        for c, (chunk, activation) in enumerate(activations):

            chunk_frames = SlidingWindow(
                start=chunk.start, step=file_frames.step, duration=file_frames.duration
            )
            activation = SlidingWindowFeature(activation, chunk_frames)

            if c < 1:
                previous_activation = activation
                continue

            # map speakers from chunks c-1 and c
            intersection = previous_activation.extent & activation.extent
            previous_data = previous_activation.crop(intersection)
            data = activation.crop(intersection)
            _, (permutation,) = permutate(previous_data[np.newaxis], data)
            for previous_s, s in enumerate(permutation):

                # if speaker is active in previous chunk, add it to the graph
                previous_active = (
                    np.max(previous_data[:, previous_s]) > self.activity_threshold
                )
                if previous_active:
                    consistency_graph.add_node((c - 1, previous_s))

                # if speaker is active in current chunk, add it to the graph
                active = np.max(data[:, s]) > self.activity_threshold
                if active:
                    consistency_graph.add_node((c, s))

                # if speaker is active in both chunks and its activations
                # are consistent enough, add edge to the graph
                consistent = (
                    np.mean(np.abs(previous_data[:, previous_s] - data[:, s]))
                    < self.consistency_threshold
                )
                if previous_active and active and consistent:
                    consistency_graph.add_edge((c - 1, previous_s), (c, s))

            # current chunk becomes previous chunk
            previous_activation = activation

        # aggregate speaker activation scores based on consistency graph
        # connected components
        num_frames_in_file = file_frames.duration_to_samples(chunk.end)
        connected_components = list(nx.connected_components(consistency_graph))
        aggregated = np.zeros((num_frames_in_file, len(connected_components)))
        overlapped = np.zeros((num_frames_in_file, len(connected_components)))

        start_frame_in_chunk = num_frames_per_chunk

        for k, component in enumerate(connected_components):

            for c, s in component:

                # corner case for very first chunk
                if c == 0:
                    end_frame = file_frames.closest_frame(0.5 * duration + 0.5 * step)
                    end_frame_in_chunk = end_frame
                    aggregated[:end_frame, k] += raw_activations.data[
                        0, :end_frame_in_chunk, s
                    ]
                    overlapped[:end_frame, k] += 1

                # corner case for very last chunk
                elif c + 1 == num_chunks:
                    start_frame = file_frames.closest_frame(
                        chunks[c].middle - 0.5 * step
                    )
                    end_frame = file_frames.closest_frame(chunks[c].end)
                    aggregated[start_frame:end_frame, k] += raw_activations.data[
                        c, raw_num_frames_per_chunk - (end_frame - start_frame) :, s
                    ]
                    overlapped[start_frame:end_frame, k] += 1

                else:

                    start_frame = file_frames.closest_frame(
                        chunks[c].middle - 0.5 * step
                    )
                    start_frame_in_chunk = file_frames.closest_frame(
                        0.5 * (duration - 2 * warmup_duration) - 0.5 * step
                    )

                    aggregated[
                        start_frame : start_frame + num_frames_per_step, k
                    ] += activations.data[
                        c,
                        start_frame_in_chunk : start_frame_in_chunk
                        + num_frames_per_step,
                        s,
                    ]
                    overlapped[start_frame : start_frame + num_frames_per_step, k] += 1

        aggregated_activations = SlidingWindowFeature(
            aggregated / (overlapped + 1e-12), file_frames
        )
        file["@segmentation/activations"] = aggregated_activations

        if self.return_activation:
            return aggregated_activations

        segmentation = self._binarize(aggregated_activations)
        segmentation.uri = file["uri"]
        return segmentation

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        return "minimize"
