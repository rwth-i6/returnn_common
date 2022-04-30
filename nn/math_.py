"""
Some basic math functions
(potential activation functions).
"""

from typing import Optional, Union, Sequence, Dict, Any
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


def log1p(x: nn.Tensor) -> nn.Tensor:
  """log1p"""
  return _activation(x, activation="log1p")


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
  """
  (a - b) ** 2. (conj(a-b) * (a-b) if a or b are complex.)

  Wraps tf.math.squared_difference.
  """
  return combine(a, b, kind="squared_difference", name=name or "squared_difference")


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


def maximum(a: Union[nn.Tensor, int, float], b: Union[nn.Tensor, int, float],
            *, name: Optional[str] = None) -> nn.Tensor:
  """
  Wraps tf.math.maximum.
  """
  return combine(a, b, kind="maximum", name=name or "maximum")


def minimum(a: Union[nn.Tensor, int, float], b: Union[nn.Tensor, int, float],
            *, name: Optional[str] = None) -> nn.Tensor:
  """
  Wraps tf.math.minimum.
  """
  return combine(a, b, kind="minimum", name=name or "minimum")


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


# noinspection PyShadowingBuiltins,PyShadowingNames
def combine(
             *sources: Union[nn.Tensor, nn.RawTensorTypes],
             kind: str,
             allow_broadcast_all_sources: Union[bool, nn.NotSpecified] = nn.NotSpecified,
             name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
  """
  Applies a binary operation, such as addition, to all sources while accumulating the partial results.
  In the first step, the binary operation is performed on the first two sources.
  After the first step, the previous results is always the left-hand operator.

  Its basic working is similar to the `reduce` function used in functional programming.
  Also see :class:`ActivationLayer`, or :class:`CompareLayer`.

  :param sources:
  :param str kind:
    currently accepted values are `average`, `add`, `sub`, `mul`, `truediv`, `floordiv`, `mod`, `pow`,
    `maximum`, `minimum`,
    `logical_and`, `logical_or`,
    `squared_difference`,
    or `eval`,
    or any function in the tf.math or tf namespace.
  :param bool|NotSpecified allow_broadcast_all_sources: allow broadcasting for all sources.
    e.g. shape [A] + [B] -> shape [A,B]. by default disabled, and there must be some source with all dims.
  :param str|None name:
  :return: layer
  """
  sources = [nn.convert_to_tensor(x) for x in sources]
  args = {
    'class': 'combine',
    'from': sources,
    'kind': kind,
  }
  args.update(_args_allow_broadcast_all_sources(sources, "combine", allow_broadcast_all_sources))
  return nn.make_layer(args, name=name or 'combine')


def compare(a: Union[nn.Tensor, nn.RawTensorTypes], b: Union[nn.Tensor, nn.RawTensorTypes], *,
            kind: str,
            allow_broadcast_all_sources: Union[bool, nn.NotSpecified] = nn.NotSpecified,
            name: Optional[str] = None) -> nn.Tensor:
  """
  compare a and b
  """
  a = nn.convert_to_tensor(a)
  b = nn.convert_to_tensor(b)
  a_const = nn.constant_value(a)
  b_const = nn.constant_value(b)
  if a_const is not None and b_const is not None:
    import operator
    res_const = {
      "equal": operator.eq, "not_equal": operator.ne,
      "less": operator.lt, "less_equal": operator.le,
      "greater": operator.gt, "greater_equal": operator.ge}[kind](a_const, b_const)
    return nn.constant(value=res_const, dtype="bool", name=name or "const_" + kind)
  from ._generated_layers import _compare
  if b_const is not None:
    return _compare(
      a, kind=kind, value=b_const, name=name or kind, allow_broadcast_all_sources=allow_broadcast_all_sources)
  if a_const is not None:
    kind_swapped = {
      "equal": "equal", "not_equal": "not_equal",
      "less": "greater", "less_equal": "greater_equal",
      "greater": "less", "greater_equal": "less_equal"}[kind]
    return _compare(
      b, kind=kind_swapped, value=a_const, name=name or kind, allow_broadcast_all_sources=allow_broadcast_all_sources)
  args = dict(kind=kind, name=name or kind)  # type: Dict[str, Any]
  args.update(_args_allow_broadcast_all_sources((a, b), "compare", allow_broadcast_all_sources))
  return _compare([a, b], **args)


def _args_allow_broadcast_all_sources(sources: Sequence[nn.Tensor],
                                      op_name: str,
                                      allow_broadcast_all_sources: Union[bool, nn.NotSpecified]
                                      ) -> Dict[str, Any]:
  args = {}
  common_dims = set(sum((x.data.dim_tags for x in sources), ()))
  if all(set(x.data.dim_tags) != common_dims for x in sources):
    if allow_broadcast_all_sources is False:
      raise ValueError(f"{op_name}: sources {sources!r} not allowed with allow_broadcast_all_sources=False")
    if allow_broadcast_all_sources is nn.NotSpecified:
      raise ValueError(f"{op_name}: sources {sources!r} require explicit allow_broadcast_all_sources=True")
    args['allow_broadcast_all_sources'] = True
  elif allow_broadcast_all_sources is not nn.NotSpecified:
    args['allow_broadcast_all_sources'] = allow_broadcast_all_sources
  return args


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
