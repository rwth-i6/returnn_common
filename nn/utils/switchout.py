"""
Switchout, see :func:`switchout`.
"""

from __future__ import annotations
from ... import nn


def switchout(
    source: nn.Tensor,
    switchout_prob: float,
    *,
    on_forward: bool = False,
) -> nn.Tensor:
    """
    Switchout - similar as dropout (:func:`dropout`) but on class labels, randomly replace by some other class label.
    """
    if not source.data.sparse_dim:
        raise ValueError(f"switchout only works on sparse data, got {source}")
    with nn.Cond(nn.train_flag() | on_forward) as cond:
        random_label = nn.random_label(source.dims, source.data.sparse_dim, dtype=source.dtype)
        cond.true = nn.where(nn.random_uniform(source.dims, maxval=1.0) < switchout_prob, random_label, source)
        cond.false = source
    return cond.result
