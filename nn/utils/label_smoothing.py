"""
Label smoothing
"""

from __future__ import annotations
from typing import Optional, Union
from ... import nn


def label_smoothing(prob: nn.Tensor, smoothing: Union[nn.Tensor, float],
                    *, axis: Optional[nn.Dim] = None) -> nn.Tensor:
  """
  Label smoothing, often used for cross entropy.

  In case of sparse data, it will become dense (via :func:`smooth_one_hot`)
  and the target label will get probability (1 - smoothing).
  """
  if not axis:
    assert prob.feature_dim
    axis = prob.feature_dim
  if prob.data.sparse:
    assert prob.data.sparse_dim == axis
    return nn.smooth_one_hot(prob, label_prob=1. - smoothing)
  else:
    assert axis in prob.shape
    # Make it consistent to the sparse case.
    # Value of 1.0 should result in (1 - smoothing).
    # Value of 0.0 should result in smoothing / (dim - 1).
    # Sum over all should still remain 1.0.
    dim = prob.data.sparse_dim.dimension
    floor_prob = smoothing / (dim - 1)
    factor = 1. - dim * floor_prob
    # Case for source[i] == 0 is clear.
    # Case for source[i] == 1: 1 - dim * floor_prob + floor_prob = 1 + (1 - dim) * floor_prob = 1 - smoothing
    # Sum over all: 1 - dim * floor_prob + flor_prob * dim = 1
    return prob * factor + floor_prob


def smooth_one_hot(source: nn.Tensor, *, label_prob: Union[nn.Tensor, float]) -> nn.Tensor:
  """
  Smooth variant of :func:`one_hot`.
  Uses ``label_prob`` for the labels and ``(1 - label_prob) / (dim - 1)`` for the remaining values.
  This is used for label smoothing.
  """
  assert source.data.sparse
  if source.data.sparse_dim.dimension is None:
    raise NotImplementedError(f"smooth_one_hot({source}) not implemented for dynamic dims")
  return nn.sparse_to_dense(
    source, label_value=label_prob, other_value=(1. - label_prob) / (source.data.sparse_dim.dimension - 1))
