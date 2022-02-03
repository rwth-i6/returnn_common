"""
Some basic math functions
(potential activation functions).
"""

from typing import Optional, Union
from .. import nn
from ._generated_layers import _eval


def identity(x: nn.Tensor) -> nn.Tensor:
  """
  Identity function. Just to have one canonical.
  Also see :func:`nn.copy`, which creates a new layer (which itself does nothing though).
  """
  return x


def relu(x: nn.Tensor) -> nn.Tensor:
  """ReLU"""
  return _activation(x, activation="relu")


def elu(x: nn.Tensor) -> nn.Tensor:
  """ELU https://arxiv.org/abs/1511.07289"""
  return _activation(x, activation="elu")


def selu(x: nn.Tensor) -> nn.Tensor:
  """SELU https://arxiv.org/abs/1706.02515"""
  return _activation(x, activation="selu")


def gelu(x: nn.Tensor) -> nn.Tensor:
  """GELU https://arxiv.org/abs/1606.08415"""
  return _activation(x, activation="gelu")


@nn.scoped
def glu(x: nn.Tensor, axis: nn.Dim) -> nn.Tensor:
  """GLU https://arxiv.org/abs/1612.08083"""
  from . import split
  a, b = split(x, axis=axis, out_dims=[axis // 2, axis // 2])
  return a * sigmoid(b)


def exp(x: nn.Tensor) -> nn.Tensor:
  """exp. see also :func:`safe_exp`"""
  return _activation(x, activation="exp")


def safe_exp(x: nn.Tensor, *, eps: float = 1e-7) -> nn.Tensor:
  """
  exp (:func:`exp`) with extra logic
    replacing earlier log_softmax by softmax, log_sigmoid by sigmoid, log by identity, etc.
  Also, for the fallback exp, clips the min and max value.

  Note that the default eps is higher than the default in RETURNN.
  """
  return _eval(x, eval=f"safe_exp(source(0), eps={eps!r})", name="safe_exp")


def log(x: nn.Tensor) -> nn.Tensor:
  """log. see also :func:`safe_log`"""
  return _activation(x, activation="log")


def safe_log(x: nn.Tensor, *, eps: float = 1e-7, use_fake_grad: bool = True) -> nn.Tensor:
  """
  log (:func:`log`) with extra logic
    replacing earlier softmax by log_softmax, sigmoid by log_sigmoid, exp by identity, etc.
  Also, for the fallback log, adds some eps in the backprop (only in backprop) to avoid nan/inf.

  Note that the default eps is higher than the default in RETURNN.
  """
  return _eval(x, eval=f"safe_log(source(0), eps={eps!r}, use_fake_grad={use_fake_grad!r})", name="safe_log")


def tanh(x: nn.Tensor) -> nn.Tensor:
  """tanh"""
  return _activation(x, activation="tanh")


def sigmoid(x: nn.Tensor) -> nn.Tensor:
  """sigmoid"""
  return _activation(x, activation="sigmoid")


def log_sigmoid(x: nn.Tensor) -> nn.Tensor:
  """log sigmoid"""
  return _activation(x, activation="log_sigmoid")


def sqrt(x: nn.Tensor) -> nn.Tensor:
  """sqrt"""
  return _activation(x, activation="sqrt")


def rsqrt(x: nn.Tensor) -> nn.Tensor:
  """rsqrt"""
  return _activation(x, activation="rsqrt")


def swish(x: nn.Tensor) -> nn.Tensor:
  """swish"""
  return _activation(x, activation="swish")


def squared_difference(a: nn.Tensor, b: nn.Tensor, *, name: Optional[str] = None) -> nn.Tensor:
  """wraps tf.math.squared_difference"""
  return _eval([a, b], eval="tf.math.squared_difference(source(0), source(1))", name=name or "squared_difference")


# softmax already provided via generated layers


def log_softmax(x: nn.Tensor, *, axis: nn.Dim, **kwargs) -> nn.Tensor:
  """
  Wraps :func:`nn.softmax` with log_space=True.
  """
  return nn.softmax(x, axis=axis, log_space=True, **kwargs)


def _activation(x: nn.Tensor, activation: str) -> nn.Tensor:
  """
  RETURNN ActivationLayer.
  Only for internal use.
  If anything is missing here in this module, please just add it.
  """
  return nn.make_layer({"class": "activation", "from": x, "activation": activation}, name=activation)


def cumsum(
      x: nn.Tensor, *,
      axis: nn.Dim,
      additional_left_summand_per_element: Optional[Union[str, int, float]] = nn.NotSpecified,
      reverse: bool = nn.NotSpecified,
      name: Optional[str] = None) -> nn.Tensor:
  """
  Applies cumsum.
  See :func:`._generated_layers._cumsum`.
  """
  from ._generated_layers import rec_cum_sum
  layer, state = rec_cum_sum(
    x, axis=axis,
    additional_left_summand_per_element=additional_left_summand_per_element,
    reverse=reverse,
    name=name)
  del state
  return layer
