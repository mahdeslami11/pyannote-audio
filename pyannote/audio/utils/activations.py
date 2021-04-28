# MIT License
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


from numbers import Number
from typing import Tuple

import numpy as np

from pyannote.core import SlidingWindow, SlidingWindowFeature


def warm_up(
    activations: SlidingWindowFeature,
    warm_up: Tuple[float, float] = None,
) -> SlidingWindowFeature:
    """Remove warm-up parts of activation scores

    Parameters
    ----------
    activations : SlidingWindowFeature
        (num_chunks, num_frames, num_classes) activations.
    warm_up : float or (float, float), optional
        Remove that many seconds on the left- and rightmost parts of each chunk,
        to only keep the central part. Defaults to 10% of chunk duration.

    Returns
    -------
    cropped_activations : SlidingWindowFeature
        Activations with warm-up parts removed.
        Both "data" and "sliding_window" attributes are updated accordingly:
        * "data" has shape (num_chunks, num_frames - warm_up_frames, num_classes)
        * "sliding_window" has shorter "duration" and delayed "start" time.

    Notes
    -----
    For convenience, "leftmost" and "rightmost" attributes are added (as dict) to
    the returned "cropped_activations" and provide "start", "duration", and "data"
    keys where either the beginning (for "leftmost") or the end (for "rightmost")
    warm-up regions are not removed. This is to ensure we can still "cover" the
    whole extent of the original activations.
    """

    num_chunks, old_num_frames, _ = activations.data.shape
    old_window: SlidingWindow = activations.sliding_window
    old_start = old_window.start
    old_duration = old_window.duration
    old_step = old_window.step

    if warm_up is None:
        warm_up = (0.1 * old_duration, 0.1 * old_duration)

    elif isinstance(warm_up, Number):
        warm_up = (warm_up, warm_up)

    warm_up_ratio: Tuple[float, float] = tuple(w / old_duration for w in warm_up)
    warm_up_frames: Tuple[int, int] = tuple(
        round(wr * old_num_frames) for wr in warm_up_ratio
    )

    new_data: np.ndarray = activations.data[
        :, warm_up_frames[0] : old_num_frames - warm_up_frames[1]
    ]
    new_window = SlidingWindow(
        start=old_start + warm_up[0],
        duration=old_duration - warm_up[0] - warm_up[1],
        step=old_step,
    )
    new_activations = SlidingWindowFeature(new_data, new_window)
    new_activations.leftmost = {
        "start": old_start,
        "duration": old_duration - warm_up[1],
        "data": activations[0][: old_num_frames - warm_up_frames[1]],
    }
    new_activations.rightmost = {
        "start": old_window[num_chunks].start + warm_up[0],
        "duration": old_duration - warm_up[0],
        "data": activations[num_chunks - 1][warm_up_frames[0] :],
    }

    return new_activations
