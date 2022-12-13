"""
Random augmentation functions.
"""

from __future__ import annotations
from typing import Tuple
from ... import nn


def random_frame_drop(source: nn.Tensor,
                      in_spatial_dim: nn.Dim,
                      drop_prob: float,
                      min_keep_num: int = 1
                      ) -> Tuple[nn.Tensor, nn.Dim]:
    """
    Randomly drop some frames.
    However, also make sure that we do not drop all frames.
    """
    mask_shape = in_spatial_dim.dyn_size_ext.dim_tags + (in_spatial_dim,)
    keep_mask = nn.random_uniform(mask_shape, minval=0.0, maxval=1.0) >= drop_prob
    # Now to make sure we keep at least one frame, pick some randomly. Pick the topk of this noise.
    noise = nn.random_uniform(mask_shape, minval=0.0, maxval=10.0)
    noise = nn.seq_len_mask(noise, axis=in_spatial_dim, mask_value=-1.0)  # do not take them outside the seq
    noise_top_values, _, k_dim = nn.top_k(noise, axis=in_spatial_dim, k=min_keep_num)
    noise_top_values = nn.reduce(noise_top_values, axis=k_dim, mode="min")
    keep_mask = keep_mask | (noise >= noise_top_values)
    return nn.boolean_mask(source, mask=keep_mask, in_spatial_dim=in_spatial_dim)
