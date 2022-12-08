"""
Random augmentation functions.
"""

from __future__ import annotations
from typing import Tuple
from ... import nn


def random_frame_drop(source: nn.Tensor, in_spatial_dim: nn.Dim, drop_prob: float) -> Tuple[nn.Tensor, nn.Dim]:
    """
    Randomly drop some frames.
    """
    mask_shape = in_spatial_dim.dyn_size_ext.dim_tags + (in_spatial_dim,)
    mask = nn.random_uniform(mask_shape, minval=0.0, maxval=1.0) < drop_prob
    return nn.boolean_mask(source, mask=mask, in_spatial_dim=in_spatial_dim)
