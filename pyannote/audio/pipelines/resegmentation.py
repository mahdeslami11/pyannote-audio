# MIT License
#
# Copyright (c) 2018-2020 CNRS
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

import tempfile
from copy import deepcopy
from types import MethodType
from typing import Text

from pytorch_lightning import Trainer
from pytorch_lightning.core.memory import ModelSummary
from torch.optim import SGD
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio import Inference
from pyannote.audio.core.callback import GraduallyUnfreeze
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineAugmentation,
    PipelineInference,
    get_augmentation,
    get_inference,
)
from pyannote.audio.tasks import SpeakerTracking
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.pipeline.parameter import Categorical, Integer, LogUniform


class Resegmentation(Pipeline):
    """Self-supervised resegmentation (aka AdaptiveSpeakerTracking)

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model.
        Defaults to "pyannote/Segmentation-PyanNet-DIHARD".
    augmentation : BaseWaveformTransform, or dict, optional
        torch_audiomentations waveform transform, used during fine-tuning.
        Defaults to no augmentation.
    diarization : str, optional
        File key to use as input diarization. Defaults to "diarization".
    confidence : str, optional
        File key to use as confidence. Defaults to not use any confidence estimation.

    Hyper-parameters
    ----------------
    num_epochs : int
        Number of epochs (where one epoch = going through the file once) between
        each gradual unfreezing step.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.

    See also
    --------
    pyannote.audio.pipelines.utils.get_inference
    """

    def __init__(
        self,
        segmentation: PipelineInference = "pyannote/Segmentation-PyanNet-DIHARD",
        augmentation: PipelineAugmentation = None,
        diarization: Text = "diarization",
        confidence: Text = None,
        fscore: bool = False,
    ):
        super().__init__()

        # base pretrained segmentation model
        self.segmentation: Inference = get_inference(segmentation)
        self.augmentation: BaseWaveformTransform = get_augmentation(augmentation)

        self.diarization = diarization
        self.confidence = confidence

        self.num_epochs = Integer(0, 20)
        self.batch_size = Categorical([1, 2, 4, 8, 16, 32])
        self.learning_rate = LogUniform(1e-6, 1)

    def apply(self, file: AudioFile) -> Annotation:

        # create a copy of file
        file = dict(file)

        # do not fine tune the model if num_epochs is zero
        if self.num_epochs == 0:
            return file[self.diarization]

        # create a dummy train-only protocol where `file` is the only training file
        class DummyProtocol(SpeakerDiarizationProtocol):
            name = "DummyProtocol"

            def train_iter(self):
                yield file

        spk = SpeakerTracking(
            DummyProtocol(),
            duration=self.segmentation.duration,
            balance=None,
            weight=self.confidence,
            batch_size=self.batch_size,
            num_workers=None,
            pin_memory=False,
            augmentation=self.augmentation,
        )

        callback = GraduallyUnfreeze(patience=self.num_epochs)
        max_epochs = (
            len(ModelSummary(self.segmentation.model, mode="top").named_modules)
            * self.num_epochs
        )

        model = deepcopy(self.segmentation.model)
        model.task = spk

        def configure_optimizers(model):
            return SGD(model.parameters(), lr=self.learning_rate)

        model.configure_optimizers = MethodType(configure_optimizers, model)

        with tempfile.TemporaryDirectory() as default_root_dir:
            trainer = Trainer(
                max_epochs=max_epochs,
                gpus=1 if self.segmentation.device.type == "cuda" else 0,
                callbacks=[callback],
                checkpoint_callback=False,
                default_root_dir=default_root_dir,
            )
            trainer.fit(model)

        inference = Inference(
            model,
            device=self.segmentation.device,
            batch_size=self.segmentation.batch_size,
            progress_hook=self.segmentation.progress_hook,
        )

        speakers_probability = inference(file)
        sliding_window = speakers_probability.sliding_window

        binarize = Binarize()

        diarization = Annotation(uri=file.get("uri", None))
        for i, data in enumerate(speakers_probability.data.T):
            speaker_probability = SlidingWindowFeature(
                data.reshape(-1, 1), sliding_window
            )
            for speaker_turn in binarize(speaker_probability):
                diarization[speaker_turn, i] = i

        return diarization
