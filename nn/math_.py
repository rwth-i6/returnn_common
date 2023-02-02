"""
Some basic math functions
(potential activation functions).
"""

from typing import Optional, Union, Tuple, Sequence, Dict, Any
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


def rint(x: nn.Tensor) -> nn.Tensor:
    """rint"""
    return _activation(x, activation="rint")


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


def expm1(x: nn.Tensor) -> nn.Tensor:
    """expm1(x) = exp(x) - 1"""
    return _activation(x, activation="expm1")


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
    """log1p(x) = log(1 + x)"""
    return _activation(x, activation="log1p")


def sin(x: nn.Tensor) -> nn.Tensor:
    """sin"""
    return _activation(x, activation="sin")


def cos(x: nn.Tensor) -> nn.Tensor:
    """cos"""
    return _activation(x, activation="cos")


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


def square(x: nn.Tensor) -> nn.Tensor:
    """
    return x^2
    """
    return _activation(x, activation="square")


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


def maximum(
    a: Union[nn.Tensor, int, float], b: Union[nn.Tensor, int, float], *, name: Optional[str] = None
) -> nn.Tensor:
    """
    Wraps tf.math.maximum.
    """
    return combine(a, b, kind="maximum", name=name or "maximum")


def minimum(
    a: Union[nn.Tensor, int, float], b: Union[nn.Tensor, int, float], *, name: Optional[str] = None
) -> nn.Tensor:
    """
    Wraps tf.math.minimum.
    """
    return combine(a, b, kind="minimum", name=name or "minimum")


def clip_by_value(
    x: nn.Tensor,
    clip_value_min: Union[nn.Tensor, int, float],
    clip_value_max: Union[nn.Tensor, int, float],
) -> nn.Tensor:
    """
    Wraps tf.clip_by_value.
    """
    return nn.make_layer(
        {
            "class": "eval",
            "from": [
                nn.convert_to_tensor(x),
                nn.convert_to_tensor(clip_value_min),
                nn.convert_to_tensor(clip_value_max),
            ],
            "eval": "tf.clip_by_value(source(0), source(1), source(2))",
        },
        name="clip_by_value",
        name_ctx_ignore_top_stack_frames=1,
    )


def gating(x: nn.Tensor, *, axis: Optional[nn.Dim] = None, gate_func=sigmoid, act_func=identity) -> nn.Tensor:
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
    name: Optional[Union[str, nn.NameCtx]] = None,
) -> nn.Tensor:
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
    if len(sources) == 2 and sum([isinstance(s, (int, float)) for s in sources]) == 1:
        # Special case for 2 sources, one of which is a constant.
        # We simplify this by using an EvalLayer.
        (tensor,) = [s for s in sources if isinstance(s, nn.Tensor)]
        a, b = ["source(0)" if s is tensor else str(s) for s in sources]
        bin_ops = {"add": "+", "sub": "-", "mul": "*", "truediv": "/", "floordiv": "//", "mod": "%", "pow": "**"}
        if kind in bin_ops:
            return nn.make_layer(
                {"class": "eval", "from": tensor, "eval": f"{a} {bin_ops[kind]} {b}"},
                name=name or kind,
                name_ctx_ignore_top_stack_frames=1,
            )
        funcs = {"maximum": "tf.maximum", "minimum": "tf.minimum"}
        if kind in funcs:
            return nn.make_layer(
                {"class": "eval", "from": tensor, "eval": f"{funcs[kind]}({a}, {b})"},
                name=name or kind,
                name_ctx_ignore_top_stack_frames=1,
            )
    sources = [nn.convert_to_tensor(x) for x in sources]
    args = {
        "class": "combine",
        "from": sources,
        "kind": kind,
    }
    args.update(_args_allow_broadcast_all_sources(sources, "combine", allow_broadcast_all_sources))
    return nn.make_layer(args, name=name or kind, name_ctx_ignore_top_stack_frames=1)


def combine_bc(a: Union[nn.Tensor, nn.RawTensorTypes], kind: str, b: Union[nn.Tensor, nn.RawTensorTypes]) -> nn.Tensor:
    """
    shorter version of :func:`combine`
    with allow_broadcast_all_sources=True.
    """
    return combine(a, b, kind=kind, allow_broadcast_all_sources=True)


def compare(
    a: Union[nn.Tensor, nn.RawTensorTypes],
    b: Union[nn.Tensor, nn.RawTensorTypes],
    *,
    kind: str,
    allow_broadcast_all_sources: Union[bool, nn.NotSpecified] = nn.NotSpecified,
    name: Optional[str] = None,
) -> nn.Tensor:
    """
    compare a and b.
    note that you can also just do `a <= b` or so.
    """
    kind = {
        "==": "equal",
        "!=": "not_equal",
        "<": "less",
        "<=": "less_equal",
        ">": "greater",
        ">=": "greater_equal",
    }.get(kind, kind)
    a = nn.convert_to_tensor(a)
    b = nn.convert_to_tensor(b)
    a_const = nn.constant_value(a)
    b_const = nn.constant_value(b)
    if a_const is not None and b_const is not None:
        import operator

        res_const = {
            "equal": operator.eq,
            "not_equal": operator.ne,
            "less": operator.lt,
            "less_equal": operator.le,
            "greater": operator.gt,
            "greater_equal": operator.ge,
        }[kind](a_const, b_const)
        return nn.constant(value=res_const, dtype="bool", name=name or "const_" + kind)
    from ._generated_layers import _compare

    if b_const is not None:
        return _compare(
            a, kind=kind, value=b_const, name=name or kind, allow_broadcast_all_sources=allow_broadcast_all_sources
        )
    if a_const is not None:
        kind_swapped = {
            "equal": "equal",
            "not_equal": "not_equal",
            "less": "greater",
            "less_equal": "greater_equal",
            "greater": "less",
            "greater_equal": "less_equal",
        }[kind]
        return _compare(
            b,
            kind=kind_swapped,
            value=a_const,
            name=name or kind,
            allow_broadcast_all_sources=allow_broadcast_all_sources,
        )
    args = dict(kind=kind, name=name or kind)  # type: Dict[str, Any]
    args.update(_args_allow_broadcast_all_sources((a, b), "compare", allow_broadcast_all_sources))
    return _compare([a, b], **args)


def compare_bc(a: Union[nn.Tensor, nn.RawTensorTypes], kind: str, b: Union[nn.Tensor, nn.RawTensorTypes]) -> nn.Tensor:
    """
    shorter version of :func:`compare`
    with allow_broadcast_all_sources=True.
    """
    return compare(a, b, kind=kind, allow_broadcast_all_sources=True)


def _args_allow_broadcast_all_sources(
    sources: Sequence[nn.Tensor], op_name: str, allow_broadcast_all_sources: Union[bool, nn.NotSpecified]
) -> Dict[str, Any]:
    args = {}
    common_dims = set(sum((x.data.dim_tags for x in sources), ()))
    if all(set(x.data.dim_tags) != common_dims for x in sources):
        if allow_broadcast_all_sources is False:
            raise ValueError(f"{op_name}: sources {sources!r} not allowed with allow_broadcast_all_sources=False")
        if allow_broadcast_all_sources is nn.NotSpecified:
            raise ValueError(f"{op_name}: sources {sources!r} require explicit allow_broadcast_all_sources=True")
        args["allow_broadcast_all_sources"] = True
    elif allow_broadcast_all_sources is not nn.NotSpecified:
        args["allow_broadcast_all_sources"] = allow_broadcast_all_sources
    return args


def cumsum(
    x: nn.Tensor,
    *,
    axis: nn.Dim,
    additional_left_summand_per_element: Optional[Union[str, int, float]] = nn.NotSpecified,
    reverse: bool = nn.NotSpecified,
    name: Optional[str] = None,
) -> nn.Tensor:
    """
    Applies cumsum.
    See :func:`._generated_layers._cumsum`.
    """
    from ._generated_layers import rec_cum_sum

    layer, state = rec_cum_sum(
        x,
        axis=axis,
        additional_left_summand_per_element=additional_left_summand_per_element,
        reverse=reverse,
        name=name,
    )
    del state
    return layer


# noinspection PyShadowingBuiltins
def top_k(
    source: nn.Tensor,
    *,
    axis: Union[nn.Dim, Sequence[nn.Dim]],
    k: Union[int, nn.Tensor],
    k_dim: Optional[nn.Dim] = None,
    sorted: bool = True,
    name: Optional[str] = None,
) -> Tuple[nn.Tensor, Union[nn.Tensor, Sequence[nn.Tensor]], nn.Dim]:
    """
    Basically wraps tf.nn.top_k.

    Directly returns the top_k values.
    The indices are accessible via the "indices" sub-layer.

    For an input [B,D] with axis=D, the output and indices values are shape [B,K].

    It's somewhat similar to :class:`ReduceLayer` with max and argmax.
    The axis dim is reduced and then a new dim for K is added.

    Axis can also cover multiple axes, such as [beam,classes].
    In that cases, there is not a single "indices" sub-layer,
    but sub-layers "indices0" .. "indices{N-1}"
    corresponding to each axis, in the same order.

    All other axes are treated as batch dims.

    :param source:
    :param axis: the axis to do the top_k on, which is reduced, or a sequence of axes
    :param k: the "K" in "TopK"
    :param k_dim: the new axis dim for K. if not provided, will be automatically created.
    :param sorted:
    :param name:
    :return: values, indices (multiple if axis is a sequence), k_dim
    """
    from ._generated_layers import _top_k
    from .base import _get_sub_layer

    values, k_dim = _top_k(source, axis=axis, k=k, k_dim=k_dim, sorted=sorted, name=name)
    if isinstance(axis, (tuple, list)):
        axes = axis
        single_axis = False
    else:
        assert isinstance(axis, nn.Dim)
        axes = [axis]
        single_axis = True
    indices = []
    for i, a in enumerate(axes):
        assert isinstance(a, nn.Dim)
        sub_name = "indices" if single_axis else f"indices{i}"
        indices_data = values.data.copy_template(name=f"{values.data.name}_{sub_name}_{a.description}")
        indices_data.dtype = "int32"
        indices_data.sparse_dim = a
        indices.append(_get_sub_layer(values, sub_name, data=indices_data))
    if single_axis:
        indices = indices[0]
    return values, indices, k_dim
