"""
Some basic math functions
(potential activation functions).
"""

from .. import nn
from .base import LayerRef, Layer


def relu(x: LayerRef) -> Layer:
  """ReLU"""
  return _activation(x, activation="relu")


def elu(x: LayerRef) -> Layer:
  """ELU https://arxiv.org/abs/1511.07289"""
  return _activation(x, activation="elu")


def selu(x: LayerRef) -> Layer:
  """SELU https://arxiv.org/abs/1706.02515"""
  return _activation(x, activation="selu")


def gelu(x: LayerRef) -> Layer:
  """GELU https://arxiv.org/abs/1606.08415"""
  return _activation(x, activation="gelu")


def exp(x: LayerRef) -> Layer:
  """exp"""
  return _activation(x, activation="exp")


def log(x: LayerRef) -> Layer:
  """log"""
  return _activation(x, activation="log")


def tanh(x: LayerRef) -> Layer:
  """tanh"""
  return _activation(x, activation="tanh")


def sigmoid(x: LayerRef) -> Layer:
  """sigmoid"""
  return _activation(x, activation="sigmoid")


def log_sigmoid(x: LayerRef) -> Layer:
  """log sigmoid"""
  return _activation(x, activation="log_sigmoid")


def swish(x: LayerRef) -> Layer:
  """swish"""
  return _activation(x, activation="swish")


# softmax already provided via generated layers
softmax = nn.softmax


def log_softmax(x: LayerRef, **kwargs) -> Layer:
  """
  Wraps :func:`nn.softmax` with log_space=True.
  """
  return nn.softmax(x, log_space=True, **kwargs)


def _activation(x: LayerRef, activation: str) -> Layer:
  """
  RETURNN ActivationLayer.
  Only for internal use.
  If anything is missing here in this module, please just add it.
  """
  return nn.make_layer({"class": "activation", "from": x, "activation": activation}, name=activation)
