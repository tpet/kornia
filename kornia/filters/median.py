from __future__ import absolute_import
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.kernels import get_binary_kernel2d


def _compute_zero_padding(kernel_size):
    r"""Utility function that computes zero padding tuple."""
    computed = tuple([(k - 1) // 2 for k in kernel_size])
    return computed[0], computed[1]


class MedianBlur(nn.Module):
    r"""Blurs an image using the median filter.

    Args:
        kernel_size (Tuple[int, int]): the blurring kernel size.

    Returns:
        torch.Tensor: the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = kornia.filters.MedianBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
    """

    def __init__(self, kernel_size):
        super(MedianBlur, self).__init__()
        self.kernel = get_binary_kernel2d(kernel_size)
        self.padding = _compute_zero_padding(kernel_size)

    def forward(self, input):  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel = self.kernel.to(input.device).to(input.dtype)
        kernel = tmp_kernel.repeat(c, 1, 1, 1)

        # map the local window to single vector
        features = F.conv2d(
            input, kernel, padding=self.padding, stride=1, groups=c)
        features = features.view(b, c, -1, h, w)  # BxCx(K_h * K_w)xHxW

        # compute the median along the feature axis
        median = torch.median(features, dim=2)[0]
        return median


# functiona api


def median_blur(input,
                kernel_size):
    r"""Blurs an image using the median filter.

    See :class:`~kornia.filters.MedianBlur` for details.
    """
    return MedianBlur(kernel_size)(input)
