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


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pyannote.core.utils.generators import pairwise

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict


class PyanNet(Model):
    """PyanNet segmentation model

    SincNet > LSTM > Feed forward > Classifier

    Parameters
    ----------
    sample_rate : int, optional
        Audio sample rate. Defaults to 16kHz (16000).
    num_channels : int, optional
        Number of channels. Defaults to mono (1).
    sincnet : dict, optional
        Keyword arugments passed to the SincNet block.
        Defaults to {"stride": 1}.
    lstm : dict, optional
        Keyword arguments passed to the LSTM layer.
        Defaults to {"hidden_size": 128, "num_layers": 2, "bidirectional": True},
        i.e. two bidirectional layers with 128 units each.
        Set "monolithic" to False to split monolithic multi-layer LSTM into multiple mono-layer LSTMs.
        This may proove useful for probing LSTM internals.
    linear : dict, optional
        Keyword arugments used to initialize linear layers
        Defaults to {"hidden_size": 128, "num_layers": 2},
        i.e. two linear layers with 128 units each.
    """

    SINCNET_DEFAULTS = {"stride": 10}

    TRANSFORMER_DEFAULTS = {
        "dim_feedforward": 2048,
        "nhead": 2,
        "norm_first": False,
        "num_layers": 2,
    }

    LSTM_DEFAULTS = {
        "hidden_size": 128,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.0,
    }

    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}

    def __init__(
        self,
        sincnet: dict = None,
        transformer: dict = None,
        lstm: dict = None,
        linear: dict = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate

        transformer = merge_dict(self.TRANSFORMER_DEFAULTS, transformer)
        transformer["batch_first"] = True

        lstm = merge_dict(self.LSTM_DEFAULTS, lstm)
        lstm["batch_first"] = True
        _ = lstm.pop("monolithic")

        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.save_hyperparameters("sincnet", "transformer", "lstm", "linear")

        self.sincnet = SincNet(**self.hparams.sincnet)
        dimension = 60

        if self.hparams.transformer["num_layers"] > 0:
            # TODO: change input dimension with a linear layer
            num_layers = self.hparams.transformer.pop("num_layers")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=60, **self.hparams.transformer
            )
            self.hparams.transformer["num_layers"] = num_layers

            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
            # dimension = self.hparams.transformer["dim_feedforward"]

        if self.hparams.lstm["num_layers"] > 0:
            self.lstm = nn.LSTM(dimension, **self.hparams.lstm)
            dimension = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        if self.hparams.linear["num_layers"] > 0:
            self.linear = nn.ModuleList(
                [
                    nn.Linear(in_features, out_features)
                    for in_features, out_features in pairwise(
                        [
                            dimension,
                        ]
                        + [self.hparams.linear["hidden_size"]]
                        * self.hparams.linear["num_layers"]
                    )
                ]
            )

    def build(self):

        if self.hparams.linear["num_layers"] > 0:
            dimension = self.hparams.linear["hidden_size"]

        elif self.hparams.lstm["num_layers"] > 0:
            dimension = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        elif self.hparams.transformer["num_layers"] > 0:
            dimension = self.hparams.transformer["dim_feedforward"]

        else:
            dimension = 60

        self.classifier = nn.Linear(dimension, len(self.specifications.classes))
        self.activation = self.default_activation()

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """

        outputs = self.sincnet(waveforms)

        outputs = rearrange(outputs, "batch feature frame -> batch frame feature")

        if self.hparams.transformer["num_layers"] > 0:
            outputs = self.transformer(outputs)

        if self.hparams.lstm["num_layers"] > 0:
            outputs, _ = self.lstm(outputs)

        if self.hparams.linear["num_layers"] > 0:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        return self.activation(self.classifier(outputs))
