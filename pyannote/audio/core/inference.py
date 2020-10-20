# MIT License
#
# Copyright (c) 2020 CNRS
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

import math
import warnings
from typing import List, Optional, Text, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Scale
from pyannote.core import Segment, SlidingWindow
from pytorch_lightning.utilities.memory import is_oom_error


class Inference:
    """Inference

    Parameters
    ----------
    model : Model
        Model. Will be automatically set to eval() mode and moved to `device` when provided.
    window : {"sliding", "whole"}, optional
        Use a "sliding" window and aggregate the corresponding outputs (default)
        or just one (potentially long) window covering the "whole" file or chunk.
    duration : float, optional
        Chunk duration, in seconds. Defaults to duration used for training the model.
        Has no effect when `window` is "whole".
    step : float, optional
        Step between consecutive chunks, in seconds. Defaults to 10% of duration.
        Has no effect when `window` is "whole".
    batch_size : int, optional
        Batch size. Larger values make inference faster. Defaults to 32.
    device : torch.device, optional
        Device used for inference. Defaults to `model.device`.
        In case `device` and `model.device` are different, model is sent to device.
    """

    def __init__(
        self,
        model: Model,
        window: Text = "sliding",
        device: torch.device = None,
        duration: float = None,
        step: float = None,
        batch_size: int = 32,
        slide: bool = True,
    ):

        self.model = model

        if window not in ["sliding", "whole"]:
            raise ValueError('`window` must be "sliding" or "whole".')

        scale = self.model.hparams.task_specifications.scale
        if scale == Scale.FRAME and window == "whole":
            warnings.warn(
                'Using "whole" `window` inference with a frame-based model might lead to bad results '
                'and huge memory consumption: it is recommended to set `window` to "sliding".'
            )

        self.window = window

        if device is None:
            device = self.model.device
        self.device = device

        self.model.eval()
        self.model.to(self.device)

        # chunk duration used during training
        training_duration = self.model.hparams.task_specifications.duration
        if duration is None:
            duration = training_duration
        elif training_duration != duration:
            warnings.warn(
                f"Model was trained with {training_duration:g}s chunks, and you requested "
                f"{duration:g}s chunks for inference: this might lead to suboptimal results."
            )
        self.duration = duration

        #  step between consecutive chunks
        if step is None:
            step = 0.1 * self.duration
        if step > self.duration:
            raise ValueError(
                f"Step between consecutive chunks is set to {step:g}s, while chunks are "
                f"only {self.duration:g}s long, leading to gaps between consecutive chunks. "
                f"Either decrease step or increase duration."
            )
        self.step = step

        self.batch_size = batch_size

    def slide(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> Tuple[np.ndarray, SlidingWindow]:

        file_duration = len(waveform) / sample_rate

        # prepare sliding audio chunks
        num_samples, num_channels = waveform.shape
        window_size: int = math.floor(self.duration * sample_rate)

        # corner case: waveform is shorter than chunk duration
        if num_samples <= window_size:

            warnings.warn(
                f"Waveform is shorter than requested sliding window ({self.duration}s): "
                f"this might lead to inconsistant results."
            )

            with torch.no_grad():
                outputs = self.model(waveform[None, :].to(self.device)).cpu().numpy()

            if self.model.hparams.task_specifications.scale == Scale.CHUNK:
                frames = SlidingWindow(
                    start=0.0, duration=self.duration, step=self.step
                )
                return outputs, frames

            _, num_frames, dimension = outputs.shape
            frames = SlidingWindow(
                start=0,
                duration=file_duration / num_frames,
                step=file_duration / num_frames,
            )
            return outputs[0], frames

        step_size: int = math.floor(self.step * sample_rate)
        chunks: torch.Tensor = rearrange(
            waveform.unfold(0, window_size, step_size),
            "chunk channel frame -> chunk frame channel",
        )
        num_chunks, _, _ = chunks.shape

        # prepare last (right-aligned) audio chunk
        if (num_samples - window_size) % step_size > 0:
            last_start = num_samples - window_size
            last_chunk: torch.Tensor = waveform[last_start:]
        else:
            last_chunk = None

        outputs: Union[List[np.ndarray], np.ndarray] = list()

        # slide over audio chunks in batch
        for c in np.arange(0, num_chunks, self.batch_size):
            batch: torch.Tensor = chunks[c : c + self.batch_size]

            with torch.no_grad():
                try:
                    output: torch.Tensor = self.model(batch.to(self.device))
                # catch "out of memory" errors
                except RuntimeError as exception:
                    if is_oom_error(exception):
                        raise MemoryError(
                            f"batch_size ({self.batch_size: d}) is probably too large. "
                            f"Try with a smaller value until memory error disappears."
                        )
                    else:
                        raise exception

            outputs.append(output.cpu().numpy())

        outputs = np.vstack(outputs)

        # if model outputs just one vector per chunk, return the outputs as they are
        if self.model.hparams.task_specifications.scale == Scale.CHUNK:
            frames = SlidingWindow(start=0.0, duration=self.duration, step=self.step)
            return outputs, frames

        # process orphan last chunk
        if last_chunk is not None:
            with torch.no_grad():
                last_output: np.ndarray = (
                    self.model(last_chunk[None, :].to(self.device))[0].cpu().numpy()
                )

        # estimate total number of aggregated output frames
        _, num_frames_per_chunk, dimension = outputs.shape
        num_frames = math.ceil(num_frames_per_chunk * file_duration / self.duration)

        # aggregated_output[i] will be used to store the sum of all predictions for frame #i
        aggregated_output: np.ndarray = np.zeros(
            (num_frames, dimension), dtype=np.float32
        )

        # overlapping_chunk_count[i] will be used to store the number of chunks that
        # overlap with frame #i
        overlapping_chunk_count: np.ndarray = np.zeros((num_frames, 1), dtype=np.int32)

        # loop on the outputs of sliding chunks
        for c, output in enumerate(outputs):
            start_frame = round(c * self.step / file_duration * num_frames)
            stop_frame = start_frame + num_frames_per_chunk
            aggregated_output[start_frame:stop_frame] += output
            overlapping_chunk_count[start_frame:stop_frame] += 1

        # process last (right-aligned) chunk separately
        if last_chunk is not None:
            start_frame = round((last_start / sample_rate) / file_duration * num_frames)
            stop_frame = start_frame + last_output.shape[0]
            aggregated_output[start_frame:stop_frame] += last_output
            overlapping_chunk_count[start_frame:stop_frame] += 1

        aggregated_output /= np.maximum(overlapping_chunk_count, 1)

        frames = SlidingWindow(
            start=0,
            duration=file_duration / num_frames,
            step=file_duration / num_frames,
        )
        return aggregated_output, frames

    def __call__(
        self, file: AudioFile
    ) -> Union[np.ndarray, Tuple[np.ndarray, SlidingWindow]]:
        """Run inference on a whole file

        Parameters
        ----------
        file : AudioFile
            Audio file.

        Returns
        -------
        output : np.ndarray
            Output.
        frames : SlidingWindow, optional
            Only returned for "sliding" window.
        """

        waveform, sample_rate = self.model.audio(file)
        # TODO remove this conversion if/when we switch to torchaudio IO
        waveform = torch.tensor(waveform, requires_grad=False)

        if self.window == "sliding":
            return self.slide(waveform, sample_rate)

        with torch.no_grad():
            return self.model(waveform[None, :].to(self.device))[0].cpu().numpy()

    def crop(
        self,
        file: AudioFile,
        chunk: Segment,
        fixed: Optional[float] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, SlidingWindow]]:
        """Run inference on a chunk

        Parameters
        ----------
        file : AudioFile
            Audio file.
        chunk : pyannote.core.Segment
            Chunk.
        fixed : float, optional
            Enforce chunk duration (in seconds). This is a hack to avoid rounding
            errors that may result in a different number of audio samples for two
            chunks of the same duration.

        # TODO: document "fixed" better in pyannote.audio.core.io.Audio

        Returns
        -------
        output : np.ndarray
            Output.
        frames : SlidingWindow, optional
            Only returned for "sliding" window.
        """

        waveform, sample_rate = self.model.audio.crop(file, chunk, fixed=fixed)
        # TODO remove this conversion if/when we switch to torchaudio IO
        waveform = torch.tensor(waveform, requires_grad=False)

        if self.window == "sliding":
            output, frames = self.slide(waveform, sample_rate)
            shifted_frames = SlidingWindow(
                start=chunk.start, duration=frames.duration, step=frames.step
            )
            return output, shifted_frames

        with torch.no_grad():
            return self.model(waveform[None, :].to(self.device))[0].cpu().numpy()

    # TODO: add a way to process a stream (to allow for online processing)
