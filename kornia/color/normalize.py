from __future__ import division
from __future__ import absolute_import
import torch
import torch.nn as nn


class Normalize(nn.Module):
    r"""Normalize a tensor image or a batch of tensor images
    with mean and standard deviation. Input must be a tensor of shape (C, H, W)
    or a batch of tensors :math:`(*, C, H, W)`.

    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels,
    this transform will normalize each channel of the input ``torch.Tensor``
    i.e. ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (torch.Tensor): Mean for each channel.
        std (torch.Tensor): Standard deviation for each channel.
    """

    def __init__(self, mean, std):

        super(Normalize, self).__init__()

        self.mean = mean
        self.std = std

    def forward(self, input):  # type: ignore
        return normalize(input, self.mean, self.std)

    def __repr__(self):
        repr = '(mean={0}, std={1})'.format(self.mean, self.std)
        return self.__class__.__name__ + repr


def normalize(data, mean,
              std):
    r"""Normalise the image with channel-wise mean and standard deviation.

    See :class:`~kornia.color.Normalize` for details.

    Args:
        data (torch.Tensor): The image tensor to be normalised.
        mean (torch.Tensor): Mean for each channel.
        std (torch.Tensor): Standard deviations for each channel.

        Returns:
            torch.Tensor: The normalised image tensor.
    """

    if not torch.is_tensor(data):
        raise TypeError('data should be a tensor. Got {}'.format(type(data)))

    if not torch.is_tensor(mean):
        raise TypeError('mean should be a tensor. Got {}'.format(type(mean)))

    if not torch.is_tensor(std):
        raise TypeError('std should be a tensor. Got {}'.format(type(std)))

    if mean.shape[0] != data.shape[-3] and mean.shape[:2] != data.shape[:2]:
        raise ValueError('mean lenght and number of channels do not match')

    if std.shape[0] != data.shape[-3] and std.shape[:2] != data.shape[:2]:
        raise ValueError('std lenght and number of channels do not match')

    mean = mean[..., :, None, None].to(data.device)
    std = std[..., :, None, None].to(data.device)

    out = (data - mean) / std

    return out
