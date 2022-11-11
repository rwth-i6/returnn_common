"""
Connectionist temporal classification (CTC) utilities.
"""

from __future__ import annotations
from typing import Optional, Tuple
from ... import nn


def ctc_greedy_decode(logits: nn.Tensor, *,
                      in_spatial_dim: nn.Dim,
                      feature_dim: Optional[nn.Dim] = None,
                      blank_index: int = -1
                      ) -> Tuple[nn.Tensor, nn.Dim]:
  """
  Also see :func:`nn.ctc_loss`.

  :param logits: non-normalized (or actually does not matter, as we will take argmax). for example [B,T,D]
  :param in_spatial_dim:
  :param feature_dim:
  :param blank_index:
  :return: (greedy_decoded, out_spatial_dim). for example [B,T'] -> D.
  """
  if feature_dim is None:
    feature_dim = logits.feature_dim
  if blank_index < 0:
    blank_index += feature_dim.dimension
  assert 0 <= blank_index < feature_dim.dimension
  argmax = nn.reduce(logits, axis=feature_dim, mode="argmax")
  shift_right = nn.shift_axis(argmax, axis=in_spatial_dim, amount=1, pad_value=-1, adjust_size_info=False)
  unique_mask = argmax != shift_right
  non_blank_mask = argmax != blank_index
  mask = unique_mask & non_blank_mask
  decoded, out_spatial_dim = nn.gather_by_mask(argmax, mask=mask, in_spatial_dim=in_spatial_dim)
  decoded_sparse_dim = feature_dim.sub_left(1) if blank_index == 0 else feature_dim - 1
  decoded = nn.reinterpret_set_sparse_dim(decoded, decoded_sparse_dim)
  return decoded, out_spatial_dim
