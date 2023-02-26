"""
Utilities operating on the targets for some loss.
"""

from __future__ import annotations
from typing import Tuple
from ... import nn


def prev_target_seq(
    targets: nn.Tensor, *, spatial_dim: nn.Dim, bos_idx: int, out_one_longer: bool
) -> Tuple[nn.Tensor, nn.Dim]:
    """
    shift by one

    :param targets: e.g. [B,S]. assumes S>0 if same_length
    :param spatial_dim: e.g. S
    :param bos_idx: e.g. 0
    :param out_one_longer:
      If False, the output will be of the same shape as the targets, i.e. have the same length.
      This means that we cut off the last symbol (via slicing).
      If True, the output will be one longer than the targets.
    :return: targets with BOS prepended, e.g. [B,S+1] or [B,S] depending on out_one_longer; and out_spatial_dim
    """
    batch_dims = targets.remaining_dims(spatial_dim)
    if out_one_longer:
        y, dim_ = targets, spatial_dim
    else:
        y, dim_ = nn.slice(targets, axis=spatial_dim, slice_end=-1)
    pad_dim = nn.SpatialDim("bos-prefix", 1)
    pad_value = nn.constant(value=bos_idx, shape=[pad_dim], dtype=targets.dtype, sparse_dim=targets.feature_dim)
    y, dim__ = nn.concat((pad_value, pad_dim), (y, dim_), allow_broadcast=True)
    if out_one_longer:
        y.verify_out_shape(set(batch_dims) | {dim__, targets.feature_dim})
        return y, dim__
    else:
        y, _ = nn.replace_dim(y, in_dim=dim__, out_dim=spatial_dim)
        y.verify_out_shape(set(batch_dims) | {spatial_dim, targets.feature_dim})
        return y, spatial_dim
