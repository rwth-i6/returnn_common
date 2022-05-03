"""
Const helpers
"""

from typing import Optional, Union, Sequence
from .. import nn


def zeros(shape: Sequence[nn.Dim], dtype: Optional[str] = nn.NotSpecified,
          *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
  """
  zeros
  """
  value = 0
  if dtype is None or dtype is nn.NotSpecified:
    dtype = "float32"
  if dtype == "bool":
    value = False
  return nn.constant(value=value, shape=shape, dtype=dtype, name=name or "zeros")


def zeros_like(value: nn.Tensor, *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
  """
  zeros with shape and dtype from value. But there is no dependency on value in the computation graph.
  """
  return zeros(shape=value.shape_ordered, dtype=value.dtype, name=name)


def ones(shape: Sequence[nn.Dim], dtype: Optional[str] = nn.NotSpecified,
         *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
  """
  ones
  """
  value = 1
  if dtype is None or dtype is nn.NotSpecified:
    dtype = "float32"
  if dtype == "bool":
    value = True
  return nn.constant(value=value, shape=shape, dtype=dtype, name=name or "ones")


def ones_like(value: nn.Tensor, *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
  """
  ones with shape and dtype from value. But there is no dependency on value in the computation graph.
  """
  return ones(shape=value.shape_ordered, dtype=value.dtype, name=name)
