from __future__ import division
from __future__ import absolute_import
from typing import Union

import torch
import torch.nn as nn

from kornia.geometry.transform.imgwarp import (
    warp_affine, get_rotation_matrix2d,
)

__all__ = [
    "affine",
    "scale",
    "rotate",
    "translate",
    "shear",
    "Scale",
    "Rotate",
    "Translate",
    "Shear",
]

# utilities to compute affine matrices


def _compute_tensor_center(tensor):
    """Computes the center of tensor plane."""
    height, width = tensor.shape[-2:]
    center_x = float(width - 1) / 2
    center_y = float(height - 1) / 2
    center = torch.tensor(
        [center_x, center_y],
        device=tensor.device, dtype=tensor.dtype)
    return center


def _compute_rotation_matrix(angle,
                             center):
    """Computes a pure affine rotation matrix."""
    scale = torch.ones_like(angle)
    matrix = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_translation_matrix(translation):
    """Computes affine matrix for translation."""
    matrix = torch.eye(
        3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_scaling_matrix(scale,
                            center):
    """Computes affine matrix for scaling."""
    angle = torch.zeros_like(scale)
    matrix = get_rotation_matrix2d(center, angle, scale)
    return matrix


def _compute_shear_matrix(shear):
    """Computes affine matrix for shearing."""
    matrix = torch.eye(3, device=shear.device, dtype=shear.dtype)
    matrix = matrix.repeat(shear.shape[0], 1, 1)

    shx, shy = torch.chunk(shear, chunks=2, dim=-1)
    matrix[..., 0, 1:2] += shx
    matrix[..., 1, 0:1] += shy
    return matrix


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L166

def affine(tensor, matrix):
    r"""Apply an affine transformation to the image.

    Args:
        tensor (torch.Tensor): The image tensor to be warped.
        matrix (torch.Tensor): The 2x3 affine transformation matrix.

    Returns:
        torch.Tensor: The warped image.
    """
    # warping needs data in the shape of BCHW
    is_unbatched = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    warped = warp_affine(tensor, matrix, (height, width))

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped


# based on:
# https://github.com/anibali/tvl/blob/master/src/tvl/transforms.py#L185

def rotate(tensor, angle,
           center = None):
    r"""Rotate the image anti-clockwise about the centre.

    See :class:`~kornia.Rotate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(angle):
        raise TypeError("Input angle type is not a torch.Tensor. Got {}"
                        .format(type(angle)))
    if center is not None and not torch.is_tensor(angle):
        raise TypeError("Input center type is not a torch.Tensor. Got {}"
                        .format(type(center)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the rotation center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    angle = angle.expand(tensor.shape[0])
    center = center.expand(tensor.shape[0], -1)
    rotation_matrix = _compute_rotation_matrix(angle, center)

    # warp using the affine transform
    return affine(tensor, rotation_matrix[..., :2, :3])


def translate(tensor, translation):
    r"""Translate the tensor in pixel units.

    See :class:`~kornia.Translate` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(translation):
        raise TypeError("Input translation type is not a torch.Tensor. Got {}"
                        .format(type(translation)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the translation matrix
    translation_matrix = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3])


def scale(tensor, scale_factor,
          center = None):
    r"""Scales the input image.

    See :class:`~kornia.Scale` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(scale_factor):
        raise TypeError("Input scale_factor type is not a torch.Tensor. Got {}"
                        .format(type(scale_factor)))

    # compute the tensor center
    if center is None:
        center = _compute_tensor_center(tensor)

    # compute the rotation matrix
    # TODO: add broadcasting to get_rotation_matrix2d for center
    center = center.expand(tensor.shape[0], -1)
    scale_factor = scale_factor.expand(tensor.shape[0])
    scaling_matrix = _compute_scaling_matrix(scale_factor, center)

    # warp using the affine transform
    return affine(tensor, scaling_matrix[..., :2, :3])


def shear(tensor, shear):
    r"""Shear the tensor.

    See :class:`~kornia.Shear` for details.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(shear):
        raise TypeError("Input shear type is not a torch.Tensor. Got {}"
                        .format(type(shear)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the translation matrix
    shear_matrix = _compute_shear_matrix(shear)

    # warp using the affine transform
    return affine(tensor, shear_matrix[..., :2, :3])


class Rotate(nn.Module):
    r"""Rotate the tensor anti-clockwise about the centre.

    Args:
        angle (torch.Tensor): The angle through which to rotate. The tensor
          must have a shape of (B), where B is batch size.
        center (torch.Tensor): The center through which to rotate. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.
    Returns:
        torch.Tensor: The rotated tensor.
    """

    def __init__(self, angle,
                 center = None):
        super(Rotate, self).__init__()
        self.angle = angle
        self.center = center

    def forward(self, input):  # type: ignore
        return rotate(input, self.angle, self.center)


class Translate(nn.Module):
    r"""Translate the tensor in pixel units.

    Args:
        translation (torch.Tensor): tensor containing the amount of pixels to
          translate in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains dx dy.

    Returns:
        torch.Tensor: The translated tensor.
    """

    def __init__(self, translation):
        super(Translate, self).__init__()
        self.translation = translation

    def forward(self, input):  # type: ignore
        return translate(input, self.translation)


class Scale(nn.Module):
    r"""Scale the tensor by a factor.

    Args:
        scale_factor (torch.Tensor): The scale factor apply. The tensor
          must have a shape of (B), where B is batch size.
        center (torch.Tensor): The center through which to scale. The tensor
          must have a shape of (B, 2), where B is batch size and last
          dimension contains cx and cy.

    Returns:
        torch.Tensor: The scaled tensor.
    """

    def __init__(self, scale_factor,
                 center = None):
        super(Scale, self).__init__()
        self.scale_factor = scale_factor
        self.center = center

    def forward(self, input):  # type: ignore
        return scale(input, self.scale_factor, self.center)


class Shear(nn.Module):
    r"""Shear the tensor.

    Args:
        tensor (torch.Tensor): The image tensor to be skewed.
        shear (torch.Tensor): tensor containing the angle to shear
          in the x and y direction. The tensor must have a shape of
          (B, 2), where B is batch size, last dimension contains shx shy.

    Returns:
        torch.Tensor: The skewed tensor.
    """

    def __init__(self, shear):
        super(Shear, self).__init__()
        self.shear = shear

    def forward(self, input):  # type: ignore
        return shear(input, self.shear)
