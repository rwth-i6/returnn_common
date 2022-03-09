"""
Some basic math functions
(potential activation functions).
"""

from typing import Optional, Union, Dict, Any
from .. import nn


def identity(x: nn.Tensor) -> nn.Tensor:
  """
  Identity function. Just to have one canonical.
  Also see :func:`nn.copy`, which creates a new layer (which itself does nothing though).
  """
  return x


# noinspection PyShadowingBuiltins
def abs(x: nn.Tensor) -> nn.Tensor:
  """abs"""
  return _activation(x, activation="abs")


def neg(x: nn.Tensor) -> nn.Tensor:
  """negative"""
  return _activation(x, activation="negative")


def logical_not(x: nn.Tensor) -> nn.Tensor:
  """logical not"""
  return _activation(x, activation="logical_not")


def ceil(x: nn.Tensor) -> nn.Tensor:
  """ceil"""
  return _activation(x, activation="ceil")


def floor(x: nn.Tensor) -> nn.Tensor:
  """floor"""
  return _activation(x, activation="floor")


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
  return _activation(x, "safe_exp", opts=dict(eps=eps))


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
  return _activation(x, "safe_log", opts=dict(eps=eps, use_fake_grad=use_fake_grad))


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
  from ._generated_layers import _combine
  return _combine([a, b], kind="squared_difference", name=name or "squared_difference")


# softmax already provided via generated layers


def log_softmax(x: nn.Tensor, *, axis: nn.Dim, **kwargs) -> nn.Tensor:
  """
  Wraps :func:`nn.softmax` with log_space=True.
  """
  return nn.softmax(x, axis=axis, log_space=True, **kwargs)


def _activation(x: nn.Tensor, activation: str, *, opts: Optional[Dict[str, Any]] = None) -> nn.Tensor:
  """
  RETURNN ActivationLayer.
  Only for internal use.
  If anything is missing here in this module, please just add it.
  """
  d = {"class": "activation", "from": x, "activation": activation}
  if opts:
    d["opts"] = opts
  return nn.make_layer(d, name=activation)


@nn.scoped
def gating(x: nn.Tensor, *, axis: Optional[nn.Dim] = None,
           gate_func=sigmoid, act_func=identity) -> nn.Tensor:
  """
  Like in gated linear unit (GLU): https://arxiv.org/abs/1612.08083
  GLU refers also to the linear transformation before the gating -- this is why this function is not called GLU.
  GLU uses gate_func=sigmoid and act_func=identity (the defaults here).

  There are other potential gating variants you might be interested at.
  See for example: https://arxiv.org/abs/2002.05202, e.g. gate_func=gelu.
  """
  if axis is None:
    axis = x.feature_dim
  from . import split
  a, b = split(x, axis=axis, out_dims=[axis // 2, axis // 2])
  return act_func(a) * gate_func(b)


def compare(a: nn.Tensor, b: nn.Tensor, *, kind: str) -> nn.Tensor:
  """
  compare a and b
  """
  a_const = nn.constant_value(a)
  b_const = nn.constant_value(b)
  if a_const is not None and b_const is not None:
    import operator
    res_const = {
      "equal": operator.eq, "not_equal": operator.ne,
      "less": operator.lt, "less_equal": operator.le,
      "greater": operator.gt, "greater_equal": operator.ge}[kind](a_const, b_const)
    return nn.constant(value=res_const, dtype="bool", name="const_" + kind)
  from ._generated_layers import _compare
  if b_const is not None:
    return _compare(a, kind=kind, value=b_const, name=kind)
  if a_const is not None:
    kind_rev = {
      "equal": "equal", "not_equal": "not_equal",
      "less": "greater", "less_equal": "greater_equal",
      "greater": "less", "greater_equal": "less_equal"}[kind]
    return _compare(b, kind=kind_rev, value=a_const, name=kind)
  return _compare([a, b], kind=kind, name=kind)


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
