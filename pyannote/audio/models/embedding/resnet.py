# Most of this code was shamelessly copied from the following repository by Jan Profant
# with Creative Commons Attribution-NonCommercial 4.0 International Public License
# https://github.com/phonexiaresearch/VBx-training-recipe

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from asteroid_filterbanks import Encoder, MelGramFB

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task


class _Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        m_channels=32,
        feat_dim=40,
        embed_dim=128,
    ):
        super().__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.conv1 = nn.Conv2d(
            1, m_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(m_channels)
        self.layer1 = self._make_layer(m_channels, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(m_channels * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(m_channels * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(m_channels * 8, num_blocks[3], stride=2)
        self.embedding = nn.Linear(
            int(feat_dim / 8) * m_channels * 16 * _Bottleneck.expansion, embed_dim
        )

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(_Bottleneck(self.in_planes, planes, stride))
            self.in_planes = planes * _Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze_(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        pooling_mean = torch.mean(out, dim=-1)
        meansq = torch.mean(out * out, dim=-1)
        pooling_std = torch.sqrt(meansq - pooling_mean ** 2 + 1e-10)
        out = torch.cat(
            (
                torch.flatten(pooling_mean, start_dim=1),
                torch.flatten(pooling_std, start_dim=1),
            ),
            1,
        )

        embedding = self.embedding(out)
        return embedding


class ResNet101(Model):

    MELGRAMFB_DEFAULTS_8KHZ = dict(
        n_filters=128,
        kernel_size=128,
        sample_rate=16000,
        n_mels=64,
        fmin=20.0,
        fmax=7700,
        norm="slaney",
    )

    MELGRAMFB_DEFAULTS_16KHZ = dict(
        n_filters=128,
        kernel_size=128,
        sample_rate=8000,
        n_mels=64,
        fmin=20.0,
        fmax=3700,
        norm="slaney",
    )

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if sample_rate == 8000:
            self.hparams.melgram_fb = dict(**self.MELGRAMFB_DEFAULTS_8KHZ)

        elif sample_rate == 16000:
            self.hparams.melgram_fb = dict(**self.MELGRAMFB_DEFAULTS_16KHZ)

        else:
            msg = f"'sample_rate' must be one of (8000, 16000) (is {sample_rate})."
            raise ValueError(msg)

        self.melgram_fb = Encoder(MelGramFB(**self.hparams.melgram_fb))
        self.resnet = _ResNet(
            [3, 4, 23, 3], feat_dim=self.hparams.melgram_fb["n_mels"], embed_dim=256
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = self.melgram_fb(waveforms)
        return self.resnet(outputs)
