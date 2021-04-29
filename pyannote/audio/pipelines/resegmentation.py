# MIT License
#
# Copyright (c) 2018-2021 CNRS
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

"""Resegmentation pipeline"""

from typing import Text

import numpy as np

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.activations import warm_up
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Uniform


class Resegmentation(Pipeline):
    """Resegmentation pipeline

    This pipeline relies on a pretrained segmentation model to improve an existing diarization
    hypothesis. Resegmentation is done locally by sliding the segmentation model over the whole
    file. For each position of the sliding window, we find the optimal mapping between the input
    diarization and the output of the segmentation model and permutate the latter accordingly.
    Permutated local segmentations scores are then aggregated over time and postprocessed using
    hysteresis thresholding.

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model. Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    diarization : str, optional
        File key to use as input diarization. Defaults to "diarization".
    inference_kwargs : dict, optional
        Keywords arguments passed to Inference.


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
        diarization: Text = "diarization",
        **inference_kwargs,
    ):

        super().__init__()

        self.segmentation = segmentation
        self.diarization = diarization

        # load model and send it to GPU (when available and not already on GPU)
        model = get_model(segmentation)
        if model.device.type == "cpu":
            (segmentation_device,) = get_devices(needs=1)
            model.to(segmentation_device)

        self.audio_ = model.audio

        # number of speakers in output of segmentation model
        self.num_frames_in_chunk_, self.seg_num_speakers_ = model.introspection(
            round(model.specifications.duration * model.hparams.sample_rate)
        )

        # prepare segmentation model for inference
        inference_kwargs["window"] = "sliding"
        inference_kwargs["skip_aggregation"] = True
        self.segmentation_inference_ = Inference(model, **inference_kwargs)

        #  hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short speech turns
        # or filling short gaps between speech turns of one speaker
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

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

        frames: SlidingWindow = self.segmentation_inference_.model.introspection.frames

        raw_activations: SlidingWindowFeature = self.segmentation_inference_(file)
        duration = raw_activations.sliding_window.duration
        step = raw_activations.sliding_window.step

        step_activations = warm_up(raw_activations, warm_up=0.5 * (duration - step))
        step_chunks = step_activations.sliding_window
        num_chunks, num_frames_per_chunk, _ = step_activations.data.shape

        # number of frames in the whole file
        num_frames_in_file = frames.samples(
            self.audio_.get_duration(file), mode="center"
        )

        # turn input diarization into binary (0 or 1) activations
        labels = file[self.diarization].labels()
        num_clusters = len(labels)
        y_original = np.zeros(
            (num_frames_in_file, len(labels)), dtype=raw_activations.data.dtype
        )
        for k, label in enumerate(labels):
            segments = file[self.diarization].label_timeline(label)
            for start, stop in frames.crop(segments, mode="center", return_ranges=True):
                y_original[start:stop, k] += 1
        y_original = np.minimum(y_original, 1, out=y_original)
        diarization = SlidingWindowFeature(y_original, frames)
        file["@resegmentation/diarization"] = diarization

        aggregated = np.zeros((num_frames_in_file, num_clusters))
        overlapped = np.zeros((num_frames_in_file, num_clusters))

        for c, (chunk, segmentation) in enumerate(raw_activations):

            # only consider active speakers in `segmentation`
            active = np.max(segmentation, axis=0) > self.onset
            if np.sum(active) == 0:
                continue
            segmentation = segmentation[:, active]

            local_diarization = diarization.crop(chunk)[
                np.newaxis, : self.num_frames_in_chunk_
            ]
            _, (permutation,) = permutate(local_diarization, segmentation)

            if c == 0:
                data = step_activations.leftmost["data"]
                start_frame, end_frame = 0, len(data)
            elif c + 1 == num_chunks:
                data = step_activations.rightmost["data"]
                start_frame = frames.closest_frame(step_activations.rightmost["start"])
                end_frame = start_frame + len(data)
            else:
                data = step_activations[c]
                start_frame = frames.closest_frame(step_chunks[c].start)
                end_frame = start_frame + num_frames_per_chunk

            aggregated[start_frame:start_frame:end_frame] += data[permutation]
            overlapped[start_frame:start_frame:end_frame] += 1.0

        speaker_activations = SlidingWindowFeature(
            aggregated / (overlapped + 1e-12), frames, labels=labels
        )

        file["@resegmentation/activations"] = speaker_activations

        diarization = self._binarize(speaker_activations)
        diarization.uri = file["uri"]
        return diarization

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
