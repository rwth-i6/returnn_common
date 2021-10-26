"""
Some basic math functions
(potential activation functions).
"""

from .base import LayerRef, Layer
from ._generated_layers import activation


def relu(x: LayerRef) -> Layer:
  """ReLU"""
  return activation(x, activation="relu")


def elu(x: LayerRef) -> Layer:
  """ELU https://arxiv.org/abs/1511.07289"""
  return activation(x, activation="elu")


def selu(x: LayerRef) -> Layer:
  """SELU https://arxiv.org/abs/1706.02515"""
  return activation(x, activation="selu")


def gelu(x: LayerRef) -> Layer:
  """GELU https://arxiv.org/abs/1606.08415"""
  return activation(x, activation="gelu")


def exp(x: LayerRef) -> Layer:
  """exp"""
  return activation(x, activation="exp")


def log(x: LayerRef) -> Layer:
  """log"""
  return activation(x, activation="log")


def tanh(x: LayerRef) -> Layer:
  """tanh"""
  return activation(x, activation="tanh")


def sigmoid(x: LayerRef) -> Layer:
  """sigmoid"""
  return activation(x, activation="sigmoid")


def swish(x: LayerRef) -> Layer:
  """swish"""
  return activation(x, activation="swish")
