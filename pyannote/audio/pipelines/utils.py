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

import itertools
import math
from copy import deepcopy
from typing import Any, Mapping, Optional, Text, Tuple, Union

import networkx as nx
import numpy as np
import torch
from pyannote.core import Annotation, Segment, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import DiarizationErrorRate
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.config import from_dict as augmentation_from_dict

from pyannote.audio import Audio, Inference, Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.utils.permutation import mae_cost_func, permutate
from pyannote.audio.utils.signal import Binarize, binarize

PipelineModel = Union[Model, Text, Mapping]


def get_model(model: PipelineModel) -> Model:
    """Load pretrained model and set it into `eval` mode.

    Parameter
    ---------
    model : Model, str, or dict
        When `Model`, returns `model` as is.
        When `str`, assumes that this is either the path to a checkpoint or the name of a
        pretrained model on Huggingface.co and loads with `Model.from_pretrained(model)`
        When `dict`, loads with `Model.from_pretrained(**model)`.

    Returns
    -------
    model : Model
        Model in `eval` mode.

    Examples
    --------
    >>> model = get_model("hbredin/VoiceActivityDetection-PyanNet-DIHARD")
    >>> model = get_model("/path/to/checkpoint.ckpt")
    >>> model = get_model({"checkpoint": "hbredin/VoiceActivityDetection-PyanNet-DIHARD",
    ...                    "map_location": torch.device("cuda")})

    See also
    --------
    pyannote.audio.core.model.Model.from_pretrained

    """

    if isinstance(model, Model):
        pass

    elif isinstance(model, Text):
        model = Model.from_pretrained(model, strict=False)

    elif isinstance(model, Mapping):
        model = Model.from_pretrained(**model)

    else:
        raise TypeError(
            f"Unsupported type ({type(model)}) for loading model: "
            f"expected `str` or `dict`."
        )

    model.eval()
    return model


PipelineInference = Union[Inference, Model, Text, Mapping]


def get_inference(inference: PipelineInference) -> Inference:
    """Load inference

    Parameter
    ---------
    inference : Inference, Model, str, or dict
        When `Inference`, returns `inference` as is.
        When `Model`, wraps it in `Inference(model)`.
        When `str`, assumes that this is either the path to a checkpoint or the name of a
        pretrained model on Huggingface.co and loads with `Inference(checkpoint)`.
        When `dict`, loads with `Inference(**inference)`.

    Returns
    -------
    inference : Inference
        Inference.

    Examples
    --------
    >>> inference = get_inference("hbredin/VoiceActivityDetection-PyanNet-DIHARD")
    >>> inference = get_inference("/path/to/checkpoint.ckpt")
    >>> inference = get_inference({"model": "hbredin/VoiceActivityDetection-PyanNet-DIHARD",
    ...                            "window": "sliding"})

    See also
    --------
    pyannote.audio.core.inference.Inference

    """

    if isinstance(inference, Inference):
        return inference

    if isinstance(inference, (Model, Text)):
        return Inference(inference)

    if isinstance(inference, Mapping):
        return Inference(**inference)

    raise TypeError(
        f"Unsupported type ({type(inference)}) for loading inference: "
        f"expected `Model`, `str` or `dict`."
    )


PipelineAugmentation = Union[BaseWaveformTransform, Mapping]


def get_augmentation(augmentation: PipelineAugmentation) -> BaseWaveformTransform:
    """Load augmentation

    Parameter
    ---------
    augmentation : BaseWaveformTransform, or dict
        When `BaseWaveformTransform`, returns `augmentation` as is.
        When `dict`, loads with `torch_audiomentations`'s `from_config` utility function.

    Returns
    -------
    augmentation : BaseWaveformTransform
        Augmentation.
    """

    if augmentation is None:
        return None

    if isinstance(augmentation, BaseWaveformTransform):
        return augmentation

    if isinstance(augmentation, Mapping):
        return augmentation_from_dict(augmentation)

    raise TypeError(
        f"Unsupported type ({type(augmentation)}) for loading augmentation: "
        f"expected `BaseWaveformTransform`, or `dict`."
    )


def get_devices(needs: int = None):
    """Get devices that can be used by the pipeline

    Parameters
    ----------
    needs : int, optional
        Number of devices needed by the pipeline

    Returns
    -------
    devices : list of torch.device
        List of available devices.
        When `needs` is provided, returns that many devices.
    """

    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        devices = [torch.device("cpu")]
        if needs is None:
            return devices
        return devices * needs

    devices = [torch.device(f"cuda:{index:d}") for index in range(num_gpus)]
    if needs is None:
        return devices
    return [device for _, device in zip(range(needs), itertools.cycle(devices))]


def logging_hook(key: Text, value: Any, file: Optional[Mapping] = None):
    file[key] = deepcopy(value)


# TODO: move to dedicated module
class SpeakerDiarizationMixin:
    """Defines a bunch of methods common to speaker diarization pipelines"""

    @staticmethod
    def set_num_speakers(
        num_speakers: int = None,
        min_speakers: int = None,
        max_speakers: int = None,
    ):
        """Validate number of speakers

        Parameters
        ----------
        num_speakers : int, optional
            Number of speakers.
        min_speakers : int, optional
            Minimum number of speakers.
        max_speakers : int, optional
            Maximum number of speakers.

        Returns
        -------
        num_speakers : int or None
        min_speakers : int
        max_speakers : int or np.inf
        """

        # override {min|max}_num_speakers by num_speakers when available
        min_speakers = num_speakers or min_speakers or 1
        max_speakers = num_speakers or max_speakers or np.inf

        if min_speakers > max_speakers:
            raise ValueError(
                f"min_speakers must be smaller than (or equal to) max_speakers "
                f"(here: min_speakers={min_speakers:g} and max_speakers={max_speakers:g})."
            )
        if min_speakers == max_speakers:
            num_speakers = min_speakers

        return num_speakers, min_speakers, max_speakers

    @staticmethod
    def optimal_mapping(
        reference: Union[Mapping, Annotation], hypothesis: Annotation
    ) -> Annotation:
        """Find the optimal bijective mapping between reference and hypothesis labels

        Parameters
        ----------
        reference : Annotation or Mapping
            Reference annotation. Can be an Annotation instance or
            a mapping with an "annotation" key.
        hypothesis : Annotation

        Returns
        -------
        mapped : Annotation
            Hypothesis mapped to reference speakers.

        """
        if isinstance(reference, Mapping):
            reference = reference["annotation"]
            annotated = reference["annotated"] if "annotated" in reference else None
        else:
            annotated = None

        mapping = DiarizationErrorRate().optimal_mapping(
            reference, hypothesis, uem=annotated
        )
        return hypothesis.rename_labels(mapping=mapping)

    # TODO: get rid of onset/offset (binarization should be applied before calling speaker_count)
    # TODO: get rid of warm-up parameter (trimming should be applied before calling speaker_count)
    @staticmethod
    def speaker_count(
        segmentations: SlidingWindowFeature,
        onset: float = 0.5,
        offset: float = None,
        warm_up: Tuple[float, float] = (0.1, 0.1),
        frames: SlidingWindow = None,
    ) -> SlidingWindowFeature:
        """Estimate frame-level number of instantaneous speakers

        Parameters
        ----------
        segmentations : SlidingWindowFeature
            (num_chunks, num_frames, num_classes)-shaped scores.
        onset : float, optional
           Onset threshold. Defaults to 0.5
        offset : float, optional
           Offset threshold. Defaults to `onset`.
        warm_up : (float, float) tuple, optional
            Left/right warm up ratio of chunk duration.
            Defaults to (0.1, 0.1), i.e. 10% on both sides.
        frames : SlidingWindow, optional
            Frames resolution. Defaults to estimate it automatically based on
            `segmentations` shape and chunk size. Providing the exact frame
            resolution (when known) leads to better temporal precision.

        Returns
        -------
        count : SlidingWindowFeature
            (num_frames, 1)-shaped instantaneous speaker count
        """

        binarized: SlidingWindowFeature = binarize(
            segmentations, onset=onset, offset=offset, initial_state=False
        )
        trimmed = Inference.trim(binarized, warm_up=warm_up)
        count = Inference.aggregate(
            np.sum(trimmed, axis=-1, keepdims=True),
            frames=frames,
            hamming=True,
            missing=0.0,
        )
        count.data = np.rint(count.data).astype(np.uint8)

        return count

    @staticmethod
    def get_stitching_graph(segmentations: SlidingWindowFeature) -> nx.Graph:
        """Build stitching graph

        Each active speaker is represented by a (chunk_idx, speaker_idx) node.

        Any pair of (overlapping) chunks go through the process of finding the
        optimal mapping of their respective speakers. Edges between speakers from
        two different chunks indicate that they were matched in the process.
        Those edges are weighted by the number of active frames in common.

        Edges between speakers from the same chunk indicate that they are overlapping.
        Those edges are weighted by the (negative) number of overlapping frames.
        We use negative values to indicate that those speakers are most likely two
        different speakers (according to the segmentation).

        Parameters
        ----------
        segmentations : (num_chunks, num_frames, num_speakers)-shaped SlidingWindowFeature
            (Raw or binarized) output of segmentation model.

        Returns
        -------
        stitching_graph : nx.Graph
            Stitching graph (see description above).

        """

        chunks = segmentations.sliding_window
        num_chunks, num_frames, num_speakers = segmentations.data.shape
        max_lookahead = math.floor(chunks.duration / chunks.step - 1)
        lookahead = 2 * (max_lookahead,)

        stitching_graph = nx.Graph()

        # for each chunk
        for C, (chunk, segmentation) in enumerate(segmentations):

            # for each adjacent chunk
            for c in range(
                max(0, C - lookahead[0]), min(num_chunks, C + lookahead[1] + 1)
            ):

                # speakers from the same chunk
                if c == C:

                    # connect each pair of speakers by an edge weighted by how much they overlap
                    for this, that in itertools.combinations(range(num_speakers), 2):
                        num_overlapping_frames = np.sum(
                            segmentation[:, this] * segmentation[:, that]
                        )
                        stitching_graph.add_edge(
                            (C, this), (c, that), cost=-num_overlapping_frames
                        )

                # speakers from adjacent chunks
                else:

                    # extract temporal support common to both chunks
                    shift = round((C - c) * num_frames * chunks.step / chunks.duration)

                    if shift < 0:
                        this_segmentations = segmentation[-shift:]
                        that_segmentations = segmentations[c, : num_frames + shift]
                    else:
                        this_segmentations = segmentation[: num_frames - shift]
                        that_segmentations = segmentations[c, shift:]

                    # find the optimal bijective mapping between their respective speakers
                    _, (permutation,) = permutate(
                        this_segmentations[np.newaxis],
                        that_segmentations,
                        cost_func=mae_cost_func,
                        return_cost=False,
                    )

                    # connect each pair of mapped speakers by an edge weighted by how much
                    # active frames they have in common
                    for this, that in enumerate(permutation):
                        num_matching_frames = np.sum(
                            this_segmentations[:, this] * that_segmentations[:, that]
                        )
                        stitching_graph.add_edge(
                            (C, this), (c, that), cost=num_matching_frames
                        )

        return stitching_graph

    @staticmethod
    def to_annotation(
        discrete_diarization: SlidingWindowFeature,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
    ) -> Annotation:
        """

        Parameters
        ----------
        discrete_diarization : SlidingWindowFeature
            (num_frames, num_speakers)-shaped discrete diarization
        min_duration_on : float, optional
            Defaults to 0.
        min_duration_off : float, optional
            Defaults to 0.

        Returns
        -------
        continuous_diarization : Annotation
            Continuous diarization
        """

        binarize = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off,
        )

        return binarize(discrete_diarization)

    @staticmethod
    def to_diarization(
        segmentations: SlidingWindowFeature,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        """Build diarization out of preprocessed segmentation and precomputed speaker count

        Parameters
        ----------
        segmentations : SlidingWindowFeature
            (num_chunks, num_frames, num_speakers)-shaped segmentations
        count : SlidingWindow_feature
            (num_frames, 1)-shaped speaker count

        Returns
        -------
        discrete_diarization : SlidingWindowFeature
            Discrete (0s and 1s) diarization.
        """

        # TODO: investigate alternative aggregation
        activations = Inference.aggregate(
            segmentations,
            frames=count.sliding_window,
            hamming=False,
            missing=0.0,
            skip_average=True,
        )

        _, num_speakers = activations.data.shape
        count.data = np.minimum(count.data, num_speakers)

        extent = activations.extent & count.extent
        activations = activations.crop(extent, return_data=False)
        count = count.crop(extent, return_data=False)

        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations.data)

        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            for i in range(c.item()):
                binary[t, speakers[i]] = 1.0

        return SlidingWindowFeature(binary, activations.sliding_window)

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1


def oracle_segmentation(
    file: AudioFile,
    window: SlidingWindow,
    frames: Union[SlidingWindow, float],
    num_speakers: int = None,
) -> SlidingWindowFeature:
    """Oracle speaker segmentation

    Simulates inference based on an (imaginary) oracle segmentation model:

    >>> oracle = Model.from_pretrained("oracle")
    >>> assert frames == oracle.introspection.frames
    >>> inference = Inference(oracle, duration=window.duration, step=window.step, skip_aggregation=True)
    >>> oracle_segmentation = inference(file)

    Parameters
    ----------
    file : AudioFile
        Audio file with "annotation".
    window : SlidingWindow
        Sliding window used for inference (see above)
    frames : SlidingWindow or float
        Output resolution of the oracle model (see above)
    num_speakers : int, optional
        Override the number of speakers returned by the oracle segmentation model
        Defaults to the actual number of speakers in the whole file

    Returns
    -------
    oracle_segmentation : (num_chunks, num_frames, num_speakers) SlidingWindowFeature
        Oracle segmentation.
    """

    if "duration" not in file:
        duration = Audio().get_duration(file)
    else:
        duration: float = file["duration"]
    reference: Annotation = file["annotation"]

    if not isinstance(frames, SlidingWindow):
        frames = SlidingWindow(start=0.0, step=frames, duration=frames)

    labels = reference.labels()
    actual_num_speakers = len(labels)
    if num_speakers is None:
        num_speakers = actual_num_speakers

    if num_speakers > actual_num_speakers:
        num_missing = num_speakers - actual_num_speakers
        labels += [
            f"FakeSpeakerForOracleSegmentationInference{i:d}"
            for i in range(num_missing)
        ]

    window = SlidingWindow(start=0.0, duration=window.duration, step=window.step)

    segmentations = []
    for chunk in window(Segment(0.0, duration)):
        chunk_segmentation: SlidingWindowFeature = reference.discretize(
            chunk,
            resolution=frames,
            labels=labels,
            duration=window.duration,
        )

        if num_speakers < actual_num_speakers:
            # keep `num_speakers` most talkative speakers
            most_talkative_index = np.argsort(-np.sum(chunk_segmentation, axis=0))[
                :num_speakers
            ]
            chunk_segmentation = chunk_segmentation[:, most_talkative_index]

        segmentations.append(chunk_segmentation)

    return SlidingWindowFeature(np.float32(np.stack(segmentations)), window)
