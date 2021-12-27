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
