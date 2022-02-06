"""
Provides the :class:`Linear` module.
"""

from typing import Optional
from .. import nn


class Linear(nn.Module):
  """
  Linear transformation.
  """

  def __init__(self, out_dim: nn.Dim, *, in_dim: Optional[nn.Dim] = None, with_bias=True):
    super().__init__()
    self.out_dim = out_dim
    self.out_dim_inner = out_dim
    self.in_dim = in_dim
    self.weight = None  # type: Optional[nn.Parameter]
    self.with_bias = with_bias
    self.bias = None  # type: Optional[nn.Parameter]
    if in_dim:
      self._lazy_init(in_dim)

  def _lazy_init(self, in_dim: nn.Dim):
    self.in_dim = in_dim
    if in_dim == self.out_dim:
      self.out_dim_inner = self.out_dim.copy(same_as_self=False, description=f"{self}:out-dim-inner")
    self.weight = nn.Parameter((self.in_dim, self.out_dim_inner))
    self.weight.initial = nn.init.Glorot()
    if self.with_bias:
      self.bias = nn.Parameter((self.out_dim_inner,))
      self.bias.initial = 0.

  @nn.scoped
  def __call__(self, source: nn.Tensor) -> nn.Tensor:
    source = nn.check_in_feature_dim_lazy_init(source, self.in_dim, self._lazy_init)
    out = nn.dot(source, self.weight, reduce=self.in_dim)
    if self.with_bias:
      out += self.bias
    if self.out_dim_inner != self.out_dim:
      out, _ = nn.reinterpret_new_dim(out, in_dim=self.out_dim_inner, out_dim=self.out_dim)
    return out
