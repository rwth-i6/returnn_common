"""
Label smoothing
"""

from __future__ import annotations
from typing import Optional, Union, Sequence
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
    dim = axis.dimension
    floor_prob = smoothing / (dim - 1)
    factor = 1. - dim * floor_prob
    # Case for prob[i] == 0 is clear.
    # Case for prob[i] == 1: 1 - dim * floor_prob + floor_prob = 1 + (1 - dim) * floor_prob = 1 - smoothing
    # Sum over all: 1 - dim * floor_prob + floor_prob * dim = 1
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


def label_smoothed_log_prob_gradient(log_prob: nn.Tensor, smoothing: Union[nn.Tensor, float],
                                     *,
                                     axis: Optional[nn.Dim] = None,
                                     exclude_labels: Optional[Sequence[int]] = None,
                                     ) -> nn.Tensor:
  """
  :param log_prob: shape [...,D] (not necessarily the same as loss)
  :param smoothing: smoothing factor, for :func:`label_smoothing`
  :param axis: label axis. uses feature_dim by default
  :param exclude_labels: list of labels to exclude from smoothing (e.g. blank)

  Assume some cross-entropy-like loss:

    loss = - sum_i target_prob[i] * log_prob[i] .

  The sum is over the label indices i (corresponding to the ``axis`` argument).
  Then the gradient of loss w.r.t. log_prob[i] is:

    grad_logprob[i] loss = -target_prob[i] .

  We assume that the negative gradient is a probability distribution, and apply :func:`label_smoothing` on it.
  More specifically, we apply the same scale and shift as in the :func:`label_smoothing` function
  via :func:`scaled_gradient`.

  Just as a side remark: assume

    log_prob = log_softmax(z) .

  The gradient of log_softmax is:

    grad_z[j] log_prob[i] = delta(i==j) - softmax(z)[j] .

  Then the gradient w.r.t. z[j] is:

    grad_z[j] loss = sum_i (grad_logprob[i] loss) (grad_z[j] logprob[i])
                   = sum_i -target_prob[i] delta(i==j) + target_prob[i] softmax(z)[j]
                   = -target_prob[j] + (sum_i target_prob[i]) softmax(z)[j]
                   = softmax(z)[j] - target_prob[j]    # assuming (sum_i target_prob[i]) == 1
                                                    .

  """
  if not axis:
    assert log_prob.feature_dim
    axis = log_prob.feature_dim
  # See formula above for label_smoothing.
  dim = axis.dimension
  floor_prob = smoothing / (dim - 1)
  factor = 1. - dim * floor_prob
  if exclude_labels:
    indices = nn.range_over_dim(axis)
    mask = True
    for label in exclude_labels:
      mask = mask & (indices != label)
    factor = nn.where(mask, factor, 1.)
    floor_prob = nn.where(mask, floor_prob, 0.)
  # The gradient is expected to be the negative target prob, thus negative floor_prob.
  # The gradient is expected to be 0. for masked frames, thus the clipping logic.
  return nn.scaled_gradient(log_prob, scale=factor, shift=-floor_prob, scale_shift_by_sum_over_axis=axis)
