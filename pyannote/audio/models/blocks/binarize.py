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


import torch
import torch.nn as nn
from torch.autograd import Function


class BinarizeFunction(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.zeros_like(input)
        output[input > 0.0] = 1.0
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return (grad_output * torch.sigmoid(input) * torch.sigmoid(-input),)


binarize = BinarizeFunction.apply


class Binarize(nn.Module):
    """Binarization made differentiable with sigmoid surrogate gradient"""

    def __init__(self, in_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.bias = bias
        if self.bias:
            self.threshold = nn.Parameter(torch.zeros(in_features))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        scores : torch.Tensor
            (batch_size, *, in_features) scores in [0, 1] range.

        Returns
        -------
        binarized : torch.Tensor
            (batch_size, *, in_features) binarized (i.e. 0 or 1) scores.

        """

        if self.bias:
            return binarize(scores - torch.sigmoid(self.threshold))
        else:
            return binarize(scores - 0.5)
