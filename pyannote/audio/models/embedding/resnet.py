# MIT License
#
# Copyright (c) 2021 CNRS
# Copyright (c) 2020-present NAVER Corp. (from https://github.com/clovaai/voxceleb_trainer)
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


from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task


class PreEmphasis(nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pre-emphasize

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of (mono) waveforms with shape (batch, 1, sample)

        Returns
        -------
        pre_emphasized : torch.Tensor
            Batch of pre-emphasized waveforms with shape (batch, sample).
        """

        _, num_channels, _ = waveforms.shape
        assert num_channels == 1, f"{self.__class__.__name__} only supports mono audio."
        waveforms = F.pad(waveforms, (1, 0), "reflect")
        return F.conv1d(waveforms, self.flipped_filter).squeeze(1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNetSE34V2(Model):
    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
        n_mels: int = 64,
        dimension: int = 256,
        num_filters: List[int] = [32, 64, 128, 256],
        layers: List[int] = [3, 4, 6, 3],
        encoder_type: str = "ASP",
    ):

        assert (
            sample_rate == 16000
        ), f"{self.__class__.__name__} only supports audio sampled at {sample_rate}Hz."
        assert num_channels == 1, f"{self.__class__.__name__} only supports mono audio."

        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        self.save_hyperparameters(
            "n_mels", "dimension", "num_filters", "layers", "encoder_type"
        )

        self.inplanes = self.hparams.num_filters[0]

        # (handcrafted) feature extraction
        self.preemphasis = PreEmphasis(coef=0.97)
        self.mel_spectrogram = MelSpectrogram(
            sample_rate=self.hparams.sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn=torch.hamming_window,
            n_mels=self.hparams.n_mels,
        )
        self.instancenorm = nn.InstanceNorm1d(self.hparams.n_mels)

        # initial convolutional layer
        self.conv1 = nn.Conv2d(
            1, self.hparams.num_filters[0], kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.hparams.num_filters[0])

        # blocks
        self.layer1 = self._make_layer(
            SEBasicBlock, self.hparams.num_filters[0], self.hparams.layers[0], stride=1
        )
        self.layer2 = self._make_layer(
            SEBasicBlock,
            self.hparams.num_filters[1],
            self.hparams.layers[1],
            stride=(2, 2),
        )
        self.layer3 = self._make_layer(
            SEBasicBlock,
            self.hparams.num_filters[2],
            self.hparams.layers[2],
            stride=(2, 2),
        )
        self.layer4 = self._make_layer(
            SEBasicBlock,
            self.hparams.num_filters[3],
            self.hparams.layers[3],
            stride=(2, 2),
        )

        outmap_size = int(self.hparams.n_mels / 8)
        self.attention = nn.Sequential(
            nn.Conv1d(num_filters[3] * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, num_filters[3] * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.hparams.encoder_type == "SAP":
            out_dim = num_filters[3] * outmap_size
        elif self.hparams.encoder_type == "ASP":
            out_dim = num_filters[3] * outmap_size * 2
        else:
            raise ValueError("Undefined encoder")

        self.embedding = nn.Linear(out_dim, self.hparams.dimension)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, waveforms: torch.Tensor):
        """

        Parameters
        ----------
        waveforms : torch.Tensor
            Batch of (mono) waveforms with shape (batch, 1, sample)

        Returns
        -------
        embeddings : torch.Tensor
            Batch of embeddings with shape (batch, dimension)
        """

        batch_size, num_channels, num_samples = waveforms.shape
        assert num_channels == 1, f"{self.__class__.__name__} only supports mono audio"

        # (handcrafted) feature extraction
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.preemphasis(waveforms)
                outputs = self.mel_spectrogram(outputs) + 1e-6
                outputs = outputs.log()
                outputs = self.instancenorm(outputs).unsqueeze(1)

        # initial convolutional layer
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.bn1(outputs)

        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)

        outputs = outputs.reshape(outputs.size()[0], -1, outputs.size()[-1])

        attention = self.attention(outputs)

        if self.hparams.encoder_type == "SAP":
            outputs = torch.sum(outputs * attention, dim=2)
        elif self.hparams.encoder_type == "ASP":
            mu = torch.sum(outputs * attention, dim=2)
            sg = torch.sqrt(
                (torch.sum((outputs ** 2) * attention, dim=2) - mu ** 2).clamp(min=1e-5)
            )
            outputs = torch.cat((mu, sg), 1)

        outputs = outputs.view(outputs.size()[0], -1)
        return self.embedding(outputs)


ResNet = ResNetSE34V2
