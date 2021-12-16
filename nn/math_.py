"""
Some basic math functions
(potential activation functions).
"""

from typing import Optional, Union
from .. import nn


def identity(x: nn.LayerRef) -> nn.LayerRef:
  """
  Identity function. Just to have one canonical.
  Also see :func:`nn.copy`, which creates a new layer (which itself does nothing though).
  """
  return x


def relu(x: nn.LayerRef) -> nn.Layer:
  """ReLU"""
  return _activation(x, activation="relu")


def elu(x: nn.LayerRef) -> nn.Layer:
  """ELU https://arxiv.org/abs/1511.07289"""
  return _activation(x, activation="elu")


def selu(x: nn.LayerRef) -> nn.Layer:
  """SELU https://arxiv.org/abs/1706.02515"""
  return _activation(x, activation="selu")


def gelu(x: nn.LayerRef) -> nn.Layer:
  """GELU https://arxiv.org/abs/1606.08415"""
  return _activation(x, activation="gelu")


def glu(x: nn.LayerRef, axis: nn.Dim) -> nn.Layer:
  """GLU https://arxiv.org/abs/1612.08083"""
  from . import split
  a, b = split(x, axis=axis, out_dims=[axis // 2, axis // 2])
  return a * sigmoid(b)


def exp(x: nn.LayerRef) -> nn.Layer:
  """exp"""
  return _activation(x, activation="exp")


def log(x: nn.LayerRef) -> nn.Layer:
  """log"""
  return _activation(x, activation="log")


def tanh(x: nn.LayerRef) -> nn.Layer:
  """tanh"""
  return _activation(x, activation="tanh")


def sigmoid(x: nn.LayerRef) -> nn.Layer:
  """sigmoid"""
  return _activation(x, activation="sigmoid")


def log_sigmoid(x: nn.LayerRef) -> nn.Layer:
  """log sigmoid"""
  return _activation(x, activation="log_sigmoid")


def swish(x: nn.LayerRef) -> nn.Layer:
  """swish"""
  return _activation(x, activation="swish")


# softmax already provided via generated layers


def log_softmax(x: nn.LayerRef, *, axis: nn.Dim, **kwargs) -> nn.Layer:
  """
  Wraps :func:`nn.softmax` with log_space=True.
  """
  return nn.softmax(x, axis=axis, log_space=True, **kwargs)


def _activation(x: nn.LayerRef, activation: str) -> nn.Layer:
  """
  RETURNN ActivationLayer.
  Only for internal use.
  If anything is missing here in this module, please just add it.
  """
  return nn.make_layer({"class": "activation", "from": x, "activation": activation}, name=activation)


def cumsum(
      x: nn.LayerRef, *,
      axis: nn.Dim,
      additional_left_summand_per_element: Optional[Union[str, int, float]] = nn.NotSpecified,
      reverse: bool = nn.NotSpecified,
      name: Optional[str] = None) -> nn.Layer:
  """
  Applies cumsum.
  See :func:`._generated_layers._cumsum`.
  """
  from ._generated_layers import _cumsum
  layer, state = _cumsum(
    x, axis=axis,
    additional_left_summand_per_element=additional_left_summand_per_element,
    reverse=reverse,
    name=name)
  del state
  return layer
