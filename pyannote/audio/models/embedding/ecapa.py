# Shamelessly borrowed from https://github.com/lawlict/ECAPA-TDNN

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MFCC

from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.models.blocks.sincnet import SincNet
from pyannote.audio.utils.params import merge_dict


class Res2Conv1dReluBn(nn.Module):
    """Res2Conv1d + BatchNorm1d + ReLU

    in_channels == out_channels == channels
    """

    def __init__(
        self,
        channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        scale=4,
    ):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(
                    self.width,
                    self.width,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    bias=bias,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out


class Conv1dReluBn(nn.Module):
    """Conv1d + BatchNorm1d + ReLU"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    """The SE connection of 1D case"""

    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    """SE-Res2Block.

    Note: residual connection is implemented in the ECAPA_TDNN model, not here.
    """

    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels),
    )


class AttentiveStatsPool(nn.Module):
    """Attentive weighted mean and standard deviation pooling"""

    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Conv1d(
            in_dim, bottleneck_dim, kernel_size=1
        )  # equals W and b in the paper
        self.linear2 = nn.Conv1d(
            bottleneck_dim, in_dim, kernel_size=1
        )  # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class EcapaMFCC(Model):
    """Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation,
    because it brings little improvment but significantly increases model parameters.
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
    """

    MFCC_DEFAULTS = {"n_mfcc": 40, "dct_type": 2, "norm": "ortho", "log_mels": False}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        mfcc: dict = None,
        channels: int = 512,
        dimension: int = 192,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        mfcc = merge_dict(self.MFCC_DEFAULTS, mfcc)
        mfcc["sample_rate"] = sample_rate

        self.save_hyperparameters("mfcc", "dimension", "channels")

        self.mfcc = MFCC(**self.hparams.mfcc)
        in_channels = self.hparams.mfcc["n_mfcc"]

        self.layer1 = Conv1dReluBn(
            in_channels, self.hparams.channels, kernel_size=5, padding=2
        )
        self.layer2 = SE_Res2Block(
            self.hparams.channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
        )
        self.layer3 = SE_Res2Block(
            self.hparams.channels,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
        )
        self.layer4 = SE_Res2Block(
            self.hparams.channels,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
        )

        cat_channels = self.hparams.channels * 3
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        self.pooling = AttentiveStatsPool(cat_channels, 128)
        self.bn1 = nn.BatchNorm1d(cat_channels * 2)
        self.linear = nn.Linear(cat_channels * 2, self.hparams.dimension)
        self.bn2 = nn.BatchNorm1d(self.hparams.dimension)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:

        features = self.mfcc(waveforms).squeeze(dim=1)
        # batch_size, num_filters, num_frames

        out1 = self.layer1(features)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))

        return out


class EcapaSincNet(Model):
    """Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation,
    because it brings little improvment but significantly increases model parameters.
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
    """

    SINCNET_DEFAULTS = {"stride": 10}

    def __init__(
        self,
        sample_rate: int = 16000,
        num_channels: int = 1,
        sincnet: dict = None,
        channels: int = 512,
        dimension: int = 192,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        sincnet = merge_dict(self.SINCNET_DEFAULTS, sincnet)
        sincnet["sample_rate"] = sample_rate

        self.save_hyperparameters("sincnet", "dimension", "channels")

        self.sincnet = SincNet(**self.hparams.sincnet)
        in_channels = 60

        self.layer1 = Conv1dReluBn(
            in_channels, self.hparams.channels, kernel_size=5, padding=2
        )
        self.layer2 = SE_Res2Block(
            self.hparams.channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            scale=8,
        )
        self.layer3 = SE_Res2Block(
            self.hparams.channels,
            kernel_size=3,
            stride=1,
            padding=3,
            dilation=3,
            scale=8,
        )
        self.layer4 = SE_Res2Block(
            self.hparams.channels,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            scale=8,
        )

        cat_channels = self.hparams.channels * 3
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        self.pooling = AttentiveStatsPool(cat_channels, 128)
        self.bn1 = nn.BatchNorm1d(cat_channels * 2)
        self.linear = nn.Linear(cat_channels * 2, self.hparams.dimension)
        self.bn2 = nn.BatchNorm1d(self.hparams.dimension)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:

        features = self.sincnet(waveforms)
        # batch_size, num_filters, num_frames

        out1 = self.layer1(features)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))

        return out
