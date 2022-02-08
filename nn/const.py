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
  return nn.constant(value=0., shape=shape, dtype=dtype, name=name or "zeros")


def zeros_like(value: nn.Tensor, *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
  """
  zeros with shape and dtype from value. But there is no dependency on value in the computation graph.
  """
  return zeros(shape=value.data.dim_tags, dtype=value.dtype, name=name)


def ones(shape: Sequence[nn.Dim], dtype: Optional[str] = nn.NotSpecified,
         *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
  """
  ones
  """
  return nn.constant(value=1., shape=shape, dtype=dtype, name=name or "ones")


def ones_like(value: nn.Tensor, *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
  """
  ones with shape and dtype from value. But there is no dependency on value in the computation graph.
  """
  return ones(shape=value.data.dim_tags, dtype=value.dtype, name=name)
