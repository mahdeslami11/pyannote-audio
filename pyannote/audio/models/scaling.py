#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

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

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr

import torch
import torch.nn as nn

try:
    from typing import Literal
except ImportError as e:
    from typing_extensions import Literal


class Scaling(nn.Module):
    """Scale feature vectors

    Parameters
    ----------
    n_features : int
        Number of input features.
    method : {"unit", "logistic"}, optional
        Defaults to no scaling.
    """

    def __init__(self, n_features: int, method: Literal["fixed", "logistic"] = None):
        super().__init__()
        self.n_features = n_features
        self.method = method

        if self.method == "logistic":
            self.batch_norm = nn.BatchNorm1d(
                1, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True
            )
            self.activation = nn.Sigmoid()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Scale features

        Parameters
        ----------
        features : torch.Tensor
            Input features of shape (batch_size, *, n_features)

        Returns
        -------
        scaled : torch.Tensor
            Scaled features of shape (batch_size, *, n_features)
        """

        if self.method is None:
            return features

        norm = features.norm(p=2, dim=-1, keepdim=True)

        if self.method == "unit":
            new_norm = 1.0

        if self.method == "logistic":
            new_norm = self.activation(self.batch_norm(norm))

        return new_norm / (norm + 1e-6) * features

    @property
    def dimension(self):
        return self.n_features
