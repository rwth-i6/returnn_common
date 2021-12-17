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
    self.in_dim = in_dim
    self.weight = nn.Parameter((in_dim, out_dim))  # TODO
    self.bias = nn.Parameter((out_dim,))  # TODO

  def _lazy_init(self, source: nn.LayerRef):
    pass  # TODO

  @nn.scoped
  def __call__(self, source: nn.LayerRef) -> nn.Layer:
    self._lazy_init(source)
    return nn.dot(source, self.weight) + self.bias
