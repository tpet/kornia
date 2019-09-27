from __future__ import absolute_import
import torch
import torch.nn as nn


class RgbToBgr(nn.Module):
    r"""Convert image from RGB to BGR.

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to BGR

    returns:
        torch.Tensor: BGR version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self):
        super(RgbToBgr, self).__init__()

    def forward(self, image):  # type: ignore
        return rgb_to_bgr(image)


def rgb_to_bgr(image):
    r"""Convert a RGB image to BGR.

    See :class:`~kornia.color.RgbToBgr` for details.

    Args:
        image (torch.Tensor): RGB Image to be converted to BGR.

    Returns:
        torch.Tensor: BGR version of the image.
    """

    return bgr_to_rgb(image)


class BgrToRgb(nn.Module):
    r"""Convert image from BGR to RGB.

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): BGR image to be converted to RGB.

    returns:
        torch.Tensor: RGB version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self):
        super(BgrToRgb, self).__init__()

    def forward(self, image):  # type: ignore
        return bgr_to_rgb(image)


def bgr_to_rgb(image):
    r"""Convert a BGR image to RGB.

    See :class:`~kornia.color.BgrToRgb` for details.

    Args:
        input (torch.Tensor): BGR Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    # flip image channels
    out = image.flip(-3)

    return out
