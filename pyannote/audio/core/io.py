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

from typing import Union, Optional, Text
from pathlib import Path
from pyannote.database import ProtocolFile
import soundfile as sf

import warnings
import numpy as np

import librosa

from pyannote.core import Segment, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.types import Alignment

AudioFile = Union[Path, Text, ProtocolFile, dict]
""" 
Audio files can be provided to the Audio class using different types:
    - a "str" instance: "/path/to/audio.wav"
    - a "Path" instance: Path("/path/to/audio.wav")
    - a ProtocolFile (or regular dict) with an "audio" key: 
        {"audio": Path("/path/to/audio.wav")}
    - a ProtocolFile (or regular dict) with both "waveform" and "sample_rate" key:
        {"waveform": (time, channel) numpy array, "sample_rate": 44100}

For last two options, an additional "channel" key can be provided as a zero-indexed
integer to load a specific channel: 
        {"audio": Path("/path/to/stereo.wav"), "channel": 0}
"""


class Audio:
    """Audio IO

    Parameters
    ----------
    sample_rate: int, optional
        Target sampling rate. Defaults to using native sampling rate.
    mono : int, optional
        Convert multi-channel to mono. Defaults to True.

    Usage
    -----
    >>> audio = Audio(sample_rate=16000, mono=True)
    >>> waveform, sample_rate = audio({"audio": "/path/to/audio.wav"})
    >>> assert sample_rate == 16000

    >>> two_seconds_stereo = np.random.rand(44100 * 2, 2, dtype=np.float32)
    >>> waveform, sample_rate = audio({"waveform": two_seconds_stereo, "sample_rate": 44100})
    >>> assert sample_rate == 16000
    >>> assert waveform.shape[1] == 1
    """

    @staticmethod
    def get_duration(file: AudioFile) -> float:
        """Get audio file duration

        Parameters
        ----------
        file : AudioFile
            Audio file.
        
        Returns
        -------
        duration : float
            Duration in seconds.
        """

        if isinstance(file, (ProtocolFile, dict)):
            audio = file["audio"]
        else:
            audio = file

        if isinstance(audio, Path):
            audio = str(audio)

        with sf.SoundFile(audio, "r") as f:
            return float(f.frames) / f.samplerate

    @staticmethod
    def is_valid(file: AudioFile) -> bool:

        if isinstance(file, (ProtocolFile, dict)):

            if "waveform" in file:

                waveform = file["waveform"]
                if len(waveform.shape) != 2 or waveform.shape[0] < waveform.shape[1]:
                    raise ValueError(
                        "'waveform' must be provided as a (time, channel) numpy array."
                    )

                sample_rate = file.get("sample_rate", None)
                if sample_rate is None:
                    raise ValueError(
                        "'waveform' must be provided with their 'sample_rate'."
                    )

                return True

            elif "audio" in file:
                audio = file["audio"]

            else:
                # TODO improve error message
                raise ValueError("either 'audio' or 'waveform' key must be provided.")

        else:
            audio = file

        #  should we check here that "audio" file exists?
        #  this will slow things down and will fail later anyway.

        return True

    def __init__(self, sample_rate=None, mono=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

    def downmix_and_resample(
        self, waveform: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Downmix and resample 

        Parameters
        ----------
        waveform : (time, channel) np.ndarray
            Waveform.
        sample_rate : int
            Sample rate.

        Returns
        -------
        waveform : (time, channel) np.ndarray
            Remixed and resampled waveform
        sample_rate : int
            New sample rate
        """

        # downmix to mono
        if self.mono and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1, keepdims=True)

        # resample
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            if self.mono:
                # librosa expects mono audio to be of shape (n,), but we have (n, 1).
                waveform = librosa.core.resample(
                    waveform[:, 0], sample_rate, self.sample_rate
                )[:, None]
            else:
                waveform = librosa.core.resample(
                    waveform.T, sample_rate, self.sample_rate
                ).T
            sample_rate = self.sample_rate

        return waveform, sample_rate

    def __call__(self, file: AudioFile):
        """Obtain waveform

        Parameters
        ----------
        file : AudioFile

        Returns
        -------
        waveform : `pyannote.core.SlidingWindowFeature`
            Waveform.

        See also
        --------
        AudioFile
        """

        self.is_valid(file)

        if isinstance(file, (ProtocolFile, dict)):

            if "waveform" in file:
                audio = None
                waveform = file["waveform"]
                sample_rate = file.get("sample_rate", None)

            elif "audio" in file:
                audio = file["audio"]
                waveform = None
                sample_rate = None

            else:
                pass

            channel = file.get("channel", None)

        else:
            audio = file
            waveform = None
            sample_rate = None
            channel = None

        if isinstance(audio, Path):
            audio = str(audio)

        if waveform is None:
            waveform, sample_rate = sf.read(audio, dtype="float32", always_2d=True)

        if channel is not None:
            waveform = waveform[:, channel - 1 : channel]

        waveform = self.downmix_and_resample(waveform, sample_rate)

        sliding_window = SlidingWindow(
            start=-0.5 / sample_rate, duration=1.0 / sample_rate, step=1.0 / sample_rate
        )

        return SlidingWindowFeature(waveform, sliding_window)

    def crop(
        self,
        file: AudioFile,
        segment: Segment,
        mode: Alignment = "center",
        fixed: Optional[float] = None,
    ) -> np.ndarray:
        """Fast version of self(file).crop(segment, **kwargs)

        Parameters
        ----------
        file : AudioFile
            Audio file.
        segment : `pyannote.core.Segment`
            Temporal segment to load.
        mode : {'loose', 'strict', 'center'}, optional
            In 'strict' mode, only samples fully included in 'segment' are
            returned. In 'loose' mode, any intersecting frames are returned. In
            'center' mode, first and last frames are chosen to be the ones
            whose centers are the closest to 'focus' start and end times.
            Defaults to 'center'.
        fixed : float, optional
            Overrides `Segment` 'focus' duration and ensures that the number of
            returned frames is fixed (which might otherwise not be the case
            because of rounding errors). Has no effect in 'strict' or 'loose'
            modes.

        Returns
        -------
        waveform : (time, channel) numpy array
            Waveform
        sample_rate : int
            Sample rate

        TODO: remove support for "mode" option. It is always "center" anyway.

        See also
        --------
        `pyannote.core.SlidingWindowFeature.crop`
        """

        self.is_valid(file)

        if isinstance(file, (ProtocolFile, dict)):

            if "waveform" in file:
                audio = None
                waveform = file["waveform"]
                sample_rate = file.get("sample_rate", None)
                frames = len(waveform)

            elif "audio" in file:
                audio = file["audio"]
                waveform = None

            else:
                pass

            channel = file.get("channel", None)

        else:
            audio = file
            waveform = None
            channel = None

        if isinstance(audio, Path):
            audio = str(audio)

        # read sample rate and number of frames
        if waveform is None:
            with sf.SoundFile(audio, "r") as f:
                sample_rate = f.samplerate
                frames = f.frames

        # infer which samples to load from sample rate and requested chunk
        #  TODO: compute start directly instead of using a sliding window
        samples = SlidingWindow(
            start=-0.5 / sample_rate, duration=1.0 / sample_rate, step=1.0 / sample_rate
        )
        ((start, stop),) = samples.crop(
            segment, mode=mode, fixed=fixed, return_ranges=True
        )

        if start < 0 or stop > frames:
            raise ValueError(
                f"requested chunk [{segment.start:.6f}, {segment.end:.6f}] "
                f"lies outside of file bounds [0., {frames / sample_rate:.6f}]."
            )

        if waveform is not None:
            data = waveform[start:stop]

        else:

            with sf.SoundFile(audio, "r") as f:

                try:
                    f.seek(start)
                    data = f.read(stop - start, dtype="float32", always_2d=True)
                except RuntimeError:
                    msg = (
                        f"SoundFile failed to seek-and-read in "
                        f"{audio}: loading the whole file..."
                    )
                    warnings.warn(msg)
                    return self(audio).crop(segment, mode=mode, fixed=fixed)

        if channel is not None:
            data = data[:, channel - 1 : channel]

        return self.downmix_and_resample(data, sample_rate)


def normalize(wav):
    """Normalize waveform"""
    return wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)
