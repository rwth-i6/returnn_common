"""
Const helpers
"""

from typing import Optional, Union, Sequence
from .. import nn


def zeros(shape: Sequence[nn.Dim], dtype: Optional[str] = nn.NotSpecified,
          *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Layer:
  """
  zeros
  """
  return nn.constant(value=0, shape=shape, dtype=dtype, name=name)


def zeros_like(value: nn.LayerRef, *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Layer:
  """
  zeros with shape and dtype from value. But there is no dependency on value in the computation graph.
  """
  return zeros(shape=value.data.dim_tags, dtype=value.dtype, name=name)
