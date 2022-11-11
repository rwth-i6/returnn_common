"""
Utilities which affect the gradient
"""

from __future__ import annotations
from typing import Optional
from ... import nn


def stop_gradient(source: nn.Tensor, name: Optional[str] = None) -> nn.Tensor:
  """wraps tf.stop_gradient"""
  return nn.scaled_gradient(source, scale=0, name=name)
