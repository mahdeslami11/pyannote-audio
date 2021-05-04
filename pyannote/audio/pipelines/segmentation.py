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

import math

import networkx as nx
import numpy as np

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.activations import split_activations, warmup_activations
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
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

        self.audio_ = model.audio

        inference_kwargs["duration"] = model.specifications.duration
        inference_kwargs["step"] = model.specifications.duration * 0.1
        inference_kwargs["skip_aggregation"] = True

        self.segmentation_inference_ = Inference(model, **inference_kwargs)

        # a speaker is active if its activation is greater than `onset`
        # for at least one frame within a chunk`
        self.activity_threshold = Uniform(0.0, 1.0)

        # mapped speakers activations between two consecutive chunks
        # are said to be consistent if
        self.consistency_threshold = Uniform(0.0, 1.0)

        self.warmup_ratio = Uniform(0.0, 0.4)

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

        frames: SlidingWindow = self.segmentation_inference_.model.introspection.frames
        raw_activations: SlidingWindowFeature = self.segmentation_inference_(file)
        raw_duration = raw_activations.sliding_window.duration

        activations = warmup_activations(
            raw_activations, warm_up=self.warmup_ratio * raw_duration
        )
        chunks = activations.sliding_window
        activations = split_activations(activations)

        file["@segmentation/raw_activations"] = activations

        num_overlapping_chunks = math.floor(0.5 * chunks.duration / chunks.step)

        # build (chunk, speaker) consistency graph
        #   - (c, s) node indicates that sth speaker of cth chunk is active
        #   - (c, s) == (c+1, s') edge indicates that sth speaker of cth chunk
        #     is mapped to s'th speaker of (c+1)th chunk

        consistency_graph = nx.Graph()

        for current_chunk, current_activation in enumerate(activations):

            for past_chunk in range(
                max(0, current_chunk - num_overlapping_chunks), current_chunk
            ):
                past_activation = activations[past_chunk]
                intersection = past_activation.extent & current_activation.extent

                current_data = current_activation.crop(intersection)
                past_data = past_activation.crop(intersection)
                _, (permutation,) = permutate(past_data[np.newaxis], current_data)

                for past_speaker, current_speaker in enumerate(permutation):

                    # if past speaker is active in the intersection, add it to the graph
                    past_active = (
                        np.max(past_data[:, past_speaker]) > self.activity_threshold
                    )
                    if past_active:
                        consistency_graph.add_node((past_chunk, past_speaker))

                    # if current speaker is active in the intersectin, add it to the graph
                    current_active = (
                        np.max(current_data[:, current_speaker])
                        > self.activity_threshold
                    )
                    if current_active:
                        consistency_graph.add_node((current_chunk, current_speaker))

                    # TO EXPERIMENT: consistency at chunk level (all speakers must agree)

                    # if current speaker is active in both chunks and its warm_activations
                    # are consistent enough, add edge to the graph
                    consistent = (
                        np.mean(
                            np.abs(
                                past_data[:, past_speaker]
                                - current_data[:, current_speaker]
                            )
                        )
                        < self.consistency_threshold
                    )
                    if past_active and current_active and consistent:
                        consistency_graph.add_edge(
                            (past_chunk, past_speaker), (current_chunk, current_speaker)
                        )

        # bipartite clique graph
        bipartite = nx.algorithms.clique.make_clique_bipartite(consistency_graph)
        is_speaker = nx.get_node_attributes(bipartite, "bipartite")
        cliques = [node for node in bipartite.nodes() if not is_speaker[node]]

        # remove clique smaller than num_overlapping chunks
        for clique in cliques:
            if len(bipartite[clique]) < num_overlapping_chunks:
                bipartite.remove_node(clique)

        # TO EXPERIMENT: remove clique that are not made out of consecutive chunks

        # group chunks if they share at least one clique
        components = [
            [n for n in c if is_speaker[n]] for c in nx.connected_components(bipartite)
        ]
        num_components = len(components)
        num_frames_in_file = frames.samples(
            self.audio_.get_duration(file), mode="center"
        )
        # FIXME -- why do we need this +100 ?
        aggregated = np.zeros((num_frames_in_file + 100, num_components))
        overlapped = np.zeros((num_frames_in_file + 100, num_components))

        for k, component in enumerate(components):

            if len(component) < 3:
                continue

            # aggregate chunks if they belong to the same component
            # remove outermost chunks of each component as their
            for i, (chunk, speaker) in enumerate(sorted(component)[1:-1]):
                chunk_activations = activations[chunk]
                speaker_activations: np.ndarray = chunk_activations.data[:, speaker]
                start_frame = frames.closest_frame(chunk_activations.extent.start)
                end_frame = start_frame + len(speaker_activations)
                aggregated[start_frame:end_frame, k] += speaker_activations
                overlapped[start_frame:end_frame, k] += 1.0

        aggregated_activations = SlidingWindowFeature(
            aggregated / (overlapped + 1e-12), frames
        )

        file["@segmentation/activations"] = aggregated_activations

        if self.return_activation:
            return aggregated_activations

        segmentation = self._binarize(aggregated_activations)
        segmentation.uri = file["uri"]
        return segmentation
