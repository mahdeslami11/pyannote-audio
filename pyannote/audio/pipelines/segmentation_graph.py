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
from itertools import combinations, product
from typing import List, Tuple

import networkx as nx
import numpy as np

from pyannote.audio import Inference, Model, Pipeline
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.activations import split_activations, warmup_activations
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.generators import pairwise
from pyannote.pipeline.parameter import Uniform

ChunkIndex = int
SpeakerIndex = int
LocalSpeaker = Tuple[ChunkIndex, SpeakerIndex]
Clique = List[LocalSpeaker]


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
                _, (permutation,), (cost,) = permutate(
                    past_data[np.newaxis], current_data, returns_cost=True
                )

                permutation_cost = np.sum(
                    [
                        cost[past_speaker, current_speaker]
                        for past_speaker, current_speaker in enumerate(permutation)
                    ]
                )

                for past_speaker, current_speaker in enumerate(permutation):

                    # if past speaker is active in the intersection, add it to the graph
                    past_active = (
                        np.max(past_data[:, past_speaker]) > self.activity_threshold
                    )
                    if past_active:
                        consistency_graph.add_node((past_chunk, past_speaker))

                    # if current speaker is active in the intersection, add it to the graph
                    current_active = (
                        np.max(current_data[:, current_speaker])
                        > self.activity_threshold
                    )
                    if current_active:
                        consistency_graph.add_node((current_chunk, current_speaker))

                    # if current speaker is active in both chunks and all chunk activations
                    # are consistent enough, add edge to the graph
                    if (
                        past_active
                        and current_active
                        and permutation_cost < self.consistency_threshold
                    ):
                        consistency_graph.add_edge(
                            (past_chunk, past_speaker), (current_chunk, current_speaker)
                        )

        # FIXME -- why do we need this +100 ?
        num_frames_in_file = (
            frames.samples(self.audio_.get_duration(file), mode="center") + 100
        )

        aggregated = []
        overlapped = []

        outliers_aggregated = []
        outliers_overlapped = []

        for c, component in enumerate(nx.connected_components(consistency_graph)):

            bipartite = nx.algorithms.clique.make_clique_bipartite(
                consistency_graph.subgraph(component).copy()
            )
            is_speaker = nx.get_node_attributes(bipartite, "bipartite")

            complete_cliques: List[Clique] = sorted(
                map(
                    lambda clique: sorted(bipartite[clique]),
                    filter(
                        lambda clique: (not is_speaker[clique])
                        and (len(bipartite[clique]) == num_overlapping_chunks + 1),
                        bipartite.nodes(),
                    ),
                )
            )

            strong_consistency_graph = nx.Graph()
            for clique1, clique2 in pairwise(complete_cliques):

                for n, m in combinations(clique1, 2):
                    strong_consistency_graph.add_edge(n, m)

                if len(set(clique1) & set(clique2)) == num_overlapping_chunks:
                    for n, m in product(clique1, clique2):
                        strong_consistency_graph.add_edge(n, m)

            strong_components = sorted(
                sorted(strong_component)
                for strong_component in nx.connected_components(
                    strong_consistency_graph
                )
            )
            num_strong_components = len(strong_components)

            # outliers are chunks that are not part of any complete cliques
            # they should be handled with care...
            outliers = component - set(strong_consistency_graph.nodes())

            # ignore outliers that are connected to more than one strong component
            for outlier in list(outliers):
                if (
                    sum(
                        1
                        if len(set(consistency_graph[outlier]) & set(strong_component))
                        > 0
                        else 0
                        for strong_component in strong_components
                    )
                    > 1
                ):
                    outliers.remove(outlier)
                    print(f"connected to more than one strong component: {outlier}")

            # ignore outliers that are connected to zero strong component
            for outlier in list(outliers):
                if (
                    sum(
                        1
                        if len(set(consistency_graph[outlier]) & set(strong_component))
                        > 0
                        else 0
                        for strong_component in strong_components
                    )
                    == 0
                ):
                    outliers.remove(outlier)
                    print(f"connected to zero strong component: {outlier}")

            print(outliers)

            num_outliers = len(outliers)

            if num_strong_components:
                sub_aggregated = np.zeros((num_frames_in_file, num_strong_components))
                sub_overlapped = np.zeros((num_frames_in_file, num_strong_components))

                for k, strong_component in enumerate(strong_components):

                    # aggregate chunks if they belong to the same strong component
                    for chunk, speaker in strong_component:
                        chunk_activations = activations[chunk]
                        speaker_activations: np.ndarray = chunk_activations.data[
                            :, speaker
                        ]
                        start_frame = frames.closest_frame(
                            chunk_activations.extent.start
                        )
                        end_frame = start_frame + len(speaker_activations)
                        sub_aggregated[start_frame:end_frame, k] += speaker_activations
                        sub_overlapped[start_frame:end_frame, k] += 1.0

                most_central = np.argmax(sub_overlapped, axis=1)
                for k in range(num_strong_components):
                    sub_aggregated[most_central != k, k] = 0.0
                    sub_overlapped[most_central != k, k] = 0.0

                aggregated.append(sub_aggregated)
                overlapped.append(sub_overlapped)

            if num_outliers:

                # if outlier is connected to more than one strong component, ignore it.
                # if outlier is connected to exactly one strong component, only keep the part
                # that is not covered

                sub_aggregated = np.zeros((num_frames_in_file, num_outliers))
                sub_overlapped = np.zeros((num_frames_in_file, num_outliers))

                for k, (chunk, speaker) in enumerate(outliers):
                    chunk_activations = activations[chunk]
                    speaker_activations: np.ndarray = chunk_activations.data[:, speaker]
                    start_frame = frames.closest_frame(chunk_activations.extent.start)
                    end_frame = start_frame + len(speaker_activations)
                    sub_aggregated[start_frame:end_frame, k] += speaker_activations
                    sub_overlapped[start_frame:end_frame, k] += 1.0

                outliers_aggregated.append(sub_aggregated)
                outliers_overlapped.append(sub_overlapped)

        aggregated_activations = SlidingWindowFeature(
            np.hstack(aggregated) / np.hstack(overlapped), frames
        )

        outliers_aggregated_activations = SlidingWindowFeature(
            np.hstack(outliers_aggregated) / np.hstack(outliers_overlapped), frames
        )

        file["@segmentation/activations"] = aggregated_activations

        if self.return_activation:
            return aggregated_activations, outliers_aggregated_activations

        segmentation = self._binarize(aggregated_activations)
        segmentation.uri = file["uri"]
        return segmentation
