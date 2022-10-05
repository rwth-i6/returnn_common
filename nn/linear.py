"""
Provides the :class:`Linear` module.
"""

from .. import nn


class Linear(nn.Module):
  """
  Linear transformation.
  """

  def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim, *, with_bias=True):
    super().__init__()
    assert isinstance(in_dim, nn.Dim) and isinstance(out_dim, nn.Dim)
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.weight = nn.Parameter((nn.dim_match_priority_when_needed(self.in_dim, self.out_dim), self.out_dim))
    self.weight.initial = nn.init.Glorot()
    self.with_bias = with_bias
    self.bias = None
    if with_bias:
      self.bias = nn.Parameter((self.out_dim,))
      self.bias.initial = 0.

  def __call__(self, source: nn.Tensor) -> nn.Tensor:
    out = nn.dot(source, self.weight, reduce=self.in_dim)
    if self.with_bias:
      out += self.bias
    return out
