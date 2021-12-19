"""
Provides the :class:`Linear` module.
"""

from typing import Optional
from .. import nn


class Linear(nn.Module):
  """
  Linear transformation.
  """

  def __init__(self, out_dim: nn.Dim, *, in_dim: Optional[nn.Dim] = None):
    super().__init__()
    self.out_dim = out_dim
    self.out_dim_inner = out_dim
    self.in_dim = in_dim
    self.weight = None  # type: Optional[nn.Parameter]
    self.bias = None  # type: Optional[nn.Parameter]
    if in_dim:
      self._lazy_init(in_dim)

  def _lazy_init(self, in_dim: nn.Dim):
    if self.in_dim:
      assert self.in_dim == in_dim
    else:
      self.in_dim = in_dim
      if in_dim == self.out_dim:
        self.out_dim_inner = self.out_dim.copy(same_as_self=False, description=f"{self}:out-dim-inner")
      self.weight = nn.Parameter((self.in_dim, self.out_dim_inner))
      self.bias = nn.Parameter((self.out_dim_inner,))

  @nn.scoped
  def __call__(self, source: nn.LayerRef) -> nn.Layer:
    self._lazy_init(source.dim)
    out = nn.dot(source, self.weight, reduce=self.in_dim) + self.bias
    if self.out_dim_inner != self.out_dim:
      out = nn.reinterpret_data(out, set_dim_tags={self.out_dim_inner: self.out_dim})
    return out
