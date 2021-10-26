"""
Some basic math functions
(potential activation functions).
"""

from .base import LayerRef, Layer
from ._generated_layers import activation


def tanh(x: LayerRef) -> Layer:
  """tanh"""
  return activation(x, activation="tanh")
