from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters.kernels import get_spatial_gradient_kernel2d
from kornia.filters.kernels import normalize_kernel2d


class SpatialGradient(nn.Module):
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self,
                 mode = 'sobel',
                 order = 1,
                 normalized = True):
        super(SpatialGradient, self).__init__()
        self.normalized = normalized
        self.order = order
        self.mode = mode
        self.kernel = get_spatial_gradient_kernel2d(mode, order)
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)
        return

    def __repr__(self):
        return self.__class__.__name__ + '('\
            'order=' + str(self.order) + ', ' + \
            'normalized=' + str(self.normalized) + ', ' + \
            'mode=' + self.mode + ')'

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
        kernel = tmp_kernel.repeat(c, 1, 1, 1, 1)

        # convolve input tensor with sobel kernel
        kernel_flip = kernel.flip(-3)
        # Pad with "replicate for spatial dims, but with zeros for channel dim
        padded_inp = F.pad(F.pad(input, [1, 1, 1, 1], 'replicate')[:, :, None],  # noqa
                                         [0, 0, 0, 0, 1, 1], 'constant', 0)
        return F.conv3d(padded_inp, kernel_flip, padding=0, groups=c)


class Sobel(nn.Module):
    r"""Computes the Sobel operator and returns the magnitude per channel.

    Return:
        torch.Tensor: the sobel edge gradient maginitudes map.

    Args:
        normalized (bool): if True, L1 norm of the kernel is set to 1.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = kornia.filters.Sobel()(input)  # 1x3x4x4
    """

    def __init__(self,
                 normalized = True):
        super(Sobel, self).__init__()
        self.normalized = normalized

    def __repr__(self):
        return self.__class__.__name__ + '('\
            'normalized=' + str(self.normalized) + ')'

    def forward(self, input):  # type: ignore
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # comput the x/y gradients
        edges = spatial_gradient(input,
                                               normalized=self.normalized)

        # unpack the edges
        gx = edges[:, :, 0]
        gy = edges[:, :, 1]

        # compute gradient maginitude
        magnitude = torch.sqrt(gx * gx + gy * gy)
        return magnitude


# functiona api


def spatial_gradient(input,
                     mode = 'sobel',
                     order = 1,
                     normalized = True):
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    See :class:`~kornia.filters.SpatialGradient` for details.
    """
    return SpatialGradient(mode, order, normalized)(input)


def sobel(input, normalized = True):
    r"""Computes the Sobel operator and returns the magnitude per channel.

    See :class:`~kornia.filters.Sobel` for details.
    """
    return Sobel(normalized)(input)
