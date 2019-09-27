from __future__ import absolute_import
from typing import Tuple, Union

import torch
import torch.nn as nn


class NonMaximaSuppression2d(nn.Module):
    r"""Applies non maxima suppression to filter.
    """

    def __init__(self, kernel_size):
        super(NonMaximaSuppression2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = self._compute_zero_padding2d(kernel_size)
        self.max_pool2d = nn.MaxPool2d(kernel_size, stride=1,
                                       padding=self.padding)

    @staticmethod
    def _compute_zero_padding2d(
            kernel_size):
        assert isinstance(kernel_size, tuple), type(kernel_size)
        assert len(kernel_size) == 2, kernel_size

        def pad(x):
            return (x - 1) // 2  # zero padding function

        ky, kx = kernel_size     # we assume a cubic kernel
        return (pad(ky), pad(kx))

    def forward(self, x):  # type: ignore
        assert len(x.shape) == 4, x.shape
        # find local maximum values
        x_max = \
            self.max_pool2d(x)

        # create mask for maximums in the original map
        x_mask = torch.where(
            x == x_max, torch.ones_like(x), torch.zeros_like(x))

        return x * x_mask  # return original masked by local max


# functiona api


def non_maxima_suppression2d(
        input, kernel_size):
    r"""Applies non maxima suppression to filter.

    See :class:`~kornia.feature.NonMaximaSuppression2d` for details.
    """
    return NonMaximaSuppression2d(kernel_size)(input)
