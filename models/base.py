"""
Base interfaces.

The core interfaces for the user are:

* :class:`ILayerMaker`, to directly create a layer dict.
  We recommend using this only for directly wrapping RETURNN layers
  and not for any higher-level logic,
  which should be done as a :class:`Module`.

* :class:`Module`, to write PyTorch-style code, which acts like a subnetwork.
  We recommend using this as the base interface
  for any higher-level interfaces
  (such as a generic decoder interface).

Instances of both objects can be called directly,
and return instances of type :class:`LayerRef`,
which can be thought of as analogue to :class:`torch.Tensor` or :class:`tf.Tensor`.

Use ``x.mark_as_loss()`` to mark some output (layer ref) as a loss.

The root network should be a :class:`Module`,
and then you can use ``mod.make_root_net_dict()``
to get the network dict.
Code example::

    class Network(Module):
      def __init__(self):
        super().__init__()
        self.lstm = Lstm(n_out=1024)

      def forward(self):
        x = get_extern_data("data")
        y = self.lstm(x)
        return y

    net = Network()
    net_dict = net.make_root_net_dict()


Alternatively, use ``with NameCtx.new_root() as name_ctx``
to setup an unnamed root name context
and then ``name_ctx.make_net_dict()``
to get the network dict.
Code example::

    with NameCtx.new_root() as root_name_ctx:
      lstm = Lstm(n_out=1024)
      x = get_extern_data("data")
      y = lstm(x)

    net_dict = root_name_ctx.make_net_dict()

---

Code conventions:

- Usual, as in RETURNN, PEP8, 2-space indents, 120 char line limit.
- Pure interface classes are prefixed with `I`.
  (`Module` is an exception because this is made analogue to PyTorch).

"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from returnn.util.basic import NotSpecified
from returnn.tf.util.data import DimensionTag
from tensorflow.python.util import nest


LayerDictRaw = Dict[str, Any]
LayerRefRaw = str
NetDictRaw = Dict[str, LayerDictRaw]
RawTensorTypes = Union[int, float, complex, bool, str]


class LayerRef:
  """
  Refers to a layer.
  """

  def __init__(self, *, name_ctx: NameCtx):
    self.name_ctx = name_ctx
    assert name_ctx.layer_ref is None
    name_ctx.layer_ref = self

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.name_ctx}>"

  def get_name(self) -> str:
    """
    Return layer name, valid in the current active name context.
    """
    return self.name_ctx.get_name_in_current_ctx()

  def get_abs_name(self) -> str:
    """
    Return absolute layer name starting from root context.
    """
    return self.name_ctx.get_abs_name()

  def mark_as_loss(self, loss_scale: Optional[float] = 1.0):
    """
    Mark this as a loss.
    """
    raise TypeError("mark_as_loss can only be called on a layer, not a layer-ref.")

  def __add__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="add", name="add")

  def __sub__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="sub", name="sub")

  def __mul__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="mul", name="mul")

  def __truediv__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="truediv", name="truediv")

  def __radd__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([convert_to_layer_ref(other), self], kind="add", name="add")

  def __rsub__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([convert_to_layer_ref(other), self], kind="sub", name="sub")

  def __rmul__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([convert_to_layer_ref(other), self], kind="mul", name="mul")

  def __rtruediv__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([convert_to_layer_ref(other), self], kind="truediv", name="truediv")

  def __neg__(self) -> LayerRef:
    from .layers import eval
    return eval(self, eval="-source(0)", name="neg")

  def __invert__(self) -> LayerRef:
    from .layers import eval
    return eval(self, eval="tf.logical_not(source(0))", name="invert")

  def __pow__(self, other: Union[RawTensorTypes, LayerRef], modulo=None) -> LayerRef:
    assert modulo is None
    from .layers import eval
    return eval([self, convert_to_layer_ref(other)], eval="tf.math.pow(source(0), source(1))", name="pow")

  def __rpow__(self, other: Union[RawTensorTypes, LayerRef], modulo=None) -> LayerRef:
    assert modulo is None
    from .layers import eval
    return eval([convert_to_layer_ref(other), self], eval="tf.math.pow(source(0), source(1))", name="pow")

  def __and__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="logical_and", name="logical_and")

  def __or__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="logical_or", name="logical_or")

  def __abs__(self) -> LayerRef:
    from .layers import eval
    return eval(self, eval="tf.abs(source(0))", name="abs")

  def __ceil__(self) -> LayerRef:
    from .layers import eval
    return eval(self, eval="tf.math.ceil(source(0))", name="ceil")

  def __floor__(self) -> LayerRef:
    from .layers import eval
    return eval(self, eval="tf.math.floor(source(0))", name="floor")

  def __floordiv__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from .layers import eval
    return eval([self, convert_to_layer_ref(other)], eval="tf.math.floordiv(source(0), source(1))", name="floordiv")

  def __eq__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from .layers import compare
    return compare([self, convert_to_layer_ref(other)], kind="equal", name="equal")

  def __ne__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from .layers import compare
    return compare([self, convert_to_layer_ref(other)], kind="not_equal", name="not_equal")

  def __lt__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from .layers import compare
    return compare([self, convert_to_layer_ref(other)], kind="less", name="less")

  def __le__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from .layers import compare
    return compare([self, convert_to_layer_ref(other)], kind="less_equal", name="less_equal")

  def __gt__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from .layers import compare
    return compare([self, convert_to_layer_ref(other)], kind="greater", name="greater")

  def __ge__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from .layers import compare
    return compare([self, convert_to_layer_ref(other)], kind="greater_equal", name="greater_equal")


class Layer(LayerRef):
  """
  Represents a layer and its output, created by :class:`ILayerMaker`.
  """

  def __init__(self, *, layer_dict: LayerDictRaw):
    super(Layer, self).__init__(name_ctx=NameCtx.top())
    assert self.name_ctx.layer is None
    self.name_ctx.layer = self
    self.layer_dict = layer_dict

  def mark_as_loss(self, loss_scale: Optional[float] = 1.0):
    """
    Mark this as a loss.
    """
    assert "loss" not in self.layer_dict
    self.layer_dict["loss"] = "as_is"
    if loss_scale is not None:
      assert "loss_scale" not in self.layer_dict
      self.layer_dict["loss_scale"] = loss_scale

  def _sis_hash(self):
    from sisyphus.hash import sis_hash_helper  # noqa
    return sis_hash_helper(self.layer_dict)


class ILayerMaker:
  """
  Makes a layer.
  """
  def __init__(self):
    self.calls = []  # type: List[Layer]

  def __repr__(self):
    return f"<{self.__class__.__name__}>"

  def make_layer_dict(self, *args, **kwargs) -> LayerDictRaw:
    """
    Return layer dict.

    The :class:`LayerDictRaw` can references other layers by using ``layer.get_name()``,
    or also by using :class:`LayerRef` instances directly,
    which will automatically be translated to ``layer.get_name()``.
    """
    raise NotImplementedError

  def get_canonical_name(self) -> str:
    """
    Get a canonical layer name if we do not have a Module attribute.
    """
    name = self.__class__.__name__
    if name.startswith("_"):
      name = name[1:]
    if name[:1].isupper():
      from returnn.util.basic import camel_case_to_snake_case
      name = camel_case_to_snake_case(name)
    return name

  def _make_layer(self, *args, **kwargs) -> Layer:
    name_ctx = NameCtx.top()
    assert name_ctx.maker is self
    if self.calls:
      name_ctx.is_repeated_call = True
    layer_dict = self.make_layer_dict(*args, **kwargs)
    return make_layer(layer_dict, name_ctx=name_ctx)

  def __call__(self, *args, name: Optional[Union[str, NameCtx]] = None, **kwargs) -> Layer:
    with NameCtx.get_from_call(maker=self, name=name):
      return self._make_layer(*args, **kwargs)


# noinspection PyAbstractClass
class _ReturnnWrappedLayerBase(ILayerMaker):
  """
  Base class for all automatically wrapped layers.
  """
  returnn_layer_class: Optional[str] = None
  has_recurrent_state: bool = False

  def _get_recurrent_state(self, layer: Layer) -> Union[LayerRef, Tuple[LayerRef, ...]]:
    """
    :returns: the recurrent state

    You might override this in case the state is more complex,
    and return some named tuple or any other hierarchical structure.
    """
    assert self.has_recurrent_state
    from ._generated_layers import _get_last_hidden_state
    return _get_last_hidden_state(layer)

  def __call__(self, *args, name: Optional[Union[str, NameCtx]] = None, **kwargs) -> (
        Union[Layer, Tuple[Layer, Union[Layer, Tuple[LayerRef, ...]]]]):
    with NameCtx.get_from_call(maker=self, name=name):
      layer = self._make_layer(*args, **kwargs)
      if not self.has_recurrent_state:
        return layer
      state = self._get_recurrent_state(layer)
      return layer, state


def make_layer(layer_dict: LayerDictRaw, *,
               name: Optional[str] = None, name_ctx: Optional[NameCtx] = None) -> Layer:
  """
  Creates the layer. This also registers the layer instance in the top name ctx.
  This assumes that the top name ctx corresponds to this layer maker.
  This is usually only used internally via :class:`ILayerMaker`.

  :param LayerDictRaw layer_dict:
  :param str|None name: layer name. if given, will create a new :class:`NameCtx`
  :param NameCtx|None name_ctx: if given, will use this name ctx.
    You can either pass ``name_ctx`` or ``name`` but not both.
  """
  if name:
    assert not name_ctx
    assert isinstance(name, str)
    name_ctx = NameCtx(suggested_name=name)
  elif name_ctx:
    assert isinstance(name_ctx, NameCtx)
  else:
    name_ctx = NameCtx.top()
  assert not name_ctx.layer_ref and not name_ctx.layer  # not yet assigned
  layer_dict = nest.map_structure(
    lambda x: x.get_name() if isinstance(x, LayerRef) else x,
    layer_dict)
  name_ctx.is_subnet_ctx = False
  if name_ctx.maker and name_ctx.maker.calls:
    name_ctx.is_repeated_call = True
    if name_ctx.parent and name_ctx.parent.is_repeated_call:
      pass  # do nothing, parent will already set reuse_params
    else:
      layer_dict = layer_dict.copy()
      assert "reuse_params" not in layer_dict
      layer_dict["reuse_params"] = name_ctx.maker.calls[0].get_name()
  layer = Layer(layer_dict=layer_dict)
  if name_ctx.maker:
    name_ctx.maker.calls.append(layer)
  return layer


def convert_to_layer_ref(x: Union[LayerRef, int, float, complex, bool, str]) -> LayerRef:
  """
  In case it is not a layer ref yet, it will make some constant.
  """
  if isinstance(x, LayerRef):
    return x
  from .layers import constant
  return constant(value=x)


class Module(ILayerMaker):
  """
  This represents a subnetwork in RETURNN, or the root network.

  You can write PyTorch-like code here, like::

      def __init__(self, dim: int, activation=tanh):
        self.layer_norm = LayerNorm()
        self.linear = Linear(dim)
        self.activation = activation

      def forward(self, x: LayerRef) -> LayerRef:
        x_ = x
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.activation(x)
        return x_ + x

  """

  def forward(self, *args, **kwargs) -> LayerRef:
    """
    Constructs the output.
    You can write PyTorch-style code here.
    """
    raise NotImplementedError

  def __call__(self, *args, name: Optional[Union[str, NameCtx]] = None, **kwargs) -> Union[Layer, Any]:
    from .layers import copy
    with NameCtx.get_from_call(maker=self, name=name) as name_ctx:
      name_ctx.is_subnet_ctx = True
      res = self.forward(*args, **kwargs)
      if isinstance(res, LayerRef):
        copy(res, name=name_ctx.get_child("output"))
      else:
        # we return more than one layer (thus also working on other layers of the subnet, that are not output)
        # by convention: first layer is the output layer
        res_flat = nest.flatten(res)
        copy(res_flat[0], name=name_ctx.get_child("output"))
      subnet_layer = self._make_layer()
    if isinstance(res, LayerRef):
      return subnet_layer  # maybe nicer to return subnet layer
    return res

  def make_layer_dict(self) -> LayerDictRaw:
    """
    Make subnet layer dict.
    """
    name_ctx = NameCtx.top()
    assert name_ctx.maker is self
    return {"class": "subnetwork", "from": [], "subnetwork": name_ctx.make_net_dict()}

  def make_root_net_dict(self) -> NetDictRaw:
    """
    Make net dict, to be used as the main RETURNN network, not within a subnetwork.
    Extern data can be accessed via :func:`get_root_extern_data`.
    """
    from .layers import copy
    with NameCtx(maker=self, parent=None) as name_ctx:
      name_ctx.is_subnet_ctx = True
      res = self.forward()
      if "output" not in name_ctx.childs:
        copy(res, name=name_ctx.get_child("output"))
      return name_ctx.make_net_dict()


class Loop:
  """
  This represents a RecLayer subnetwork in RETURNN,
  i.e. where the calculation per step is defined explicitly.

  (For RecLayer with a predefined unit, see :class:`Rec`.
   Or for example :class:`Lstm`.)

  To define a loop like this pseudo Python code::

    x  # given, shape (batch, time, dim)
    h = Zeros([batch,dim])()  # initial state, shape (batch,dim)
    out = []
    for t in range(x.max_seq_len):
      x_lin = Linear(dim)(x[t])
      h_prev = h
      h = Linear(dim)(x_lin + h_prev)
      out.append(h)

    h  # final state
    out  # shape (time, batch, h_dim)

  You would write::

    with Loop() as loop:
      x_t = loop.unstack(x)
      x_lin = Linear(dim)(x_t)
      loop.state.h = State(shape=[batch,dim], initial=0)  # optional
      loop.state.h = Linear(dim)(x_lin + loop.state.h)
      out = loop.stack(loop.state.h)

  ``state`` is :class:`Loop._StateHolder` and manages the recurrent state.

  This code must be run within a :func:`Module.forward`
  or with some active global name context (:class:`NameCtx`).

  This API is currently in development, and might change.
  See: https://github.com/rwth-i6/returnn_common/issues/16
  """

  def __init__(self, *,
               max_seq_len: Optional[Union[str, int]] = NotSpecified,
               optimize_move_layers_out: Optional[bool] = NotSpecified,
               cheating: bool = NotSpecified,
               unroll: bool = NotSpecified,
               back_prop: Optional[bool] = NotSpecified,
               use_global_rec_step_offset: bool = NotSpecified,
               include_eos: bool = NotSpecified,
               debug: Optional[bool] = NotSpecified,
               name: str = "loop"
               ):
    super(Loop, self).__init__()
    self.extra_opts = {
      key: value for (key, value) in locals().items()
      if value is not NotSpecified and key not in {"self", "__class__", "name"}}
    self.layer_maker = _LoopLayerMaker(loop=self)
    self.name_ctx = NameCtx(maker=self.layer_maker, suggested_name=name, parent=NameCtx.current_ctx())
    self.name_ctx.is_subnet_ctx = True
    self.name_ctx.extend_reserved_names({"output", "end"})
    self.state = _StateHolder(loop=self)
    self.unstacked_refs = []  # type: List[LayerRef]
    self.outputs = []  # type: List[LayerRef]
    self.end_ref = None  # type: Optional[LayerRef]

  def __enter__(self) -> Loop:
    self.name_ctx.__enter__()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    try:
      if not exc_type:
        if not self.outputs:  # stack or last was called at least once, so we have some output
          raise Exception(f"{self}: call `stack` or `last` at least once to define some output")
        if not self.end_ref and not self.unstacked_refs:
          raise Exception(f"{self}: call `unstack` or `end` at least once to define the loop length")
        # Make sure there is an "output" layer. (Similar as for Module with subnetwork.)
        if "output" not in self.name_ctx.childs:
          from .layers import copy
          copy(self.outputs[0], name=self.name_ctx.get_child("output"))
    finally:
      self.name_ctx.__exit__(exc_type, exc_val, exc_tb)
    if not exc_type:
      self.layer_maker(name=self.name_ctx)

  def unstack(self, source: LayerRef, *, axis: Union[str, DimensionTag], name: Optional[str] = None) -> LayerRef:
    """
    Unrolls over the specified axis, and provides each frame in each loop iteration.
    """
    from .layers import rec_unstack
    res = rec_unstack(source, axis=axis, name=name)
    self.unstacked_refs.append(res)
    return res

  def stack(self, source: LayerRef, *, name: Optional[str] = None) -> LayerRef:
    """
    Accumulates the frames of source within the loop,
    to make it accessible outside the loop.
    """
    from .layers import copy
    if not name and "output" not in self.name_ctx.childs:
      name = self.name_ctx.get_child("output")
    res = copy(source, name=name)
    assert isinstance(res, Layer)
    if res.name_ctx.name != "output":
      res.layer_dict["is_output_layer"] = True
    self.outputs.append(res)
    return res

  def last(self, source: LayerRef, *, name: Optional[str] = None) -> LayerRef:
    """
    Gets the last value from source.
    """
    # TODO ...
    raise NotImplementedError("Loop.last not implemented yet...")

  def end(self, source: LayerRef) -> LayerRef:
    """
    For loops with dynamic ending condition (which might not use unstack),
    this defines the ending condition.
    """
    assert not self.end_ref  # do not call this multiple times
    from .layers import copy
    self.end_ref = copy(source, name=self.name_ctx.get_child("end"))
    return self.end_ref


class _LoopLayerMaker(ILayerMaker):
  def __init__(self, loop: Loop):
    super(_LoopLayerMaker, self).__init__()
    self.loop = loop

  def make_layer_dict(self) -> LayerDictRaw:
    """
    Makes layer dict for this loop, i.e. a RecLayer.
    """
    name_ctx = NameCtx.top()
    assert name_ctx.maker is self
    return {"class": "rec", "from": [], "unit": name_ctx.make_net_dict(), **self.loop.extra_opts}


class _StateHolder:
  def __init__(self, loop: Loop):
    self._loop = loop
    self._state = {}  # type: Dict[str, State]

  def _get_state(self, name: str) -> State:
    if name in self._state:
      return self._state[name]
    state = State()
    state.set_name_and_loop(name=name, loop=self._loop)
    self._state[name] = state
    return state

  def __getattr__(self, item):
    return self._get_state(item).get()

  def __setattr__(self, key, value):
    if key in {"_state", "_loop"}:
      return super().__setattr__(key, value)
    if isinstance(value, State):
      value.set_name_and_loop(name=key, loop=self._loop)
      self._state[key] = value
      return
    self._get_state(key).assign(value)


class State:
  """
  Represents some recurrent state, to be used with :class:`Loop`.
  It can also represent some nested hierarchy of states.
  """

  def __init__(self, *, shape=None, initial=None):
    super(State, self).__init__()
    self.shape = shape
    self.initial = initial
    self.loop = None  # type: Optional[Loop]
    self.name = None  # type: Optional[str]
    self.name_ctx = None  # type: Optional[NameCtx]
    self.assigned_value = None

  def set_name_and_loop(self, *, name: str, loop: Loop):
    """
    Assigns the name (internally on first assignment).
    """
    if self.name == name and self.loop is loop:
      return
    assert not self.loop and not self.name and not self.name_ctx  # not yet assigned
    self.loop = loop
    self.name = name
    self.name_ctx = NameCtx(suggested_name=name)

  def assign(self, value):
    """
    Assign the new value for the current iteration.
    """
    assert value is not None
    assert self.assigned_value is None, (
      f"Cannot assign the rec state {self.loop}/{self.name} multiple times, "
      f"assigned previously to {self.assigned_value}, now to {value}")
    self.assigned_value = value
    from .layers import copy
    copy(value, name=self.name_ctx)

  def get(self):
    """
    Return prev or current value
    """
    if self.assigned_value is None:  # not yet assigned
      # Return prev value
      return NameCtx.top().get_child_layer_ref(f"prev:{self.name_ctx.name}")
    return self.assigned_value


def get_root_extern_data(data_key: str) -> LayerRef:
  """
  Get extern data from root.
  """
  scope = NameCtx.top()  # must exist
  scope_abs = scope.get_abs_name_ctx_list()
  root_scope = scope_abs[0]
  root_layer_name = f"data:{data_key}"
  return get_special_layer(root_layer_name, scope=root_scope)


def get_extern_data(data_key: str) -> LayerRef:
  """
  Get extern data from current scope.
  """
  return get_special_layer(f"data:{data_key}")


def get_special_layer(name: str, *, scope: Optional[NameCtx] = None) -> LayerRef:
  """
  Special layer can be "data:..." or whatever.
  """
  if not scope:
    scope = NameCtx.current_ctx()  # must exist
  return scope.get_child_layer_ref(name)


def get_sub_layer(layer: LayerRef, name: str) -> LayerRef:
  """
  Like the "{layer}/{name}" syntax in RETURNN.
  Normally this should only be needed for internal usage.
  """
  return layer.name_ctx.get_child_layer_ref(name)


class NameCtx:
  """
  This is a helper class to keep track of the current name context when creating layers.
  Usually you do not need to access this directly.
  """

  stack = []  # type: List[NameCtx]
  _ReservedNames = {"data", "output"}

  @classmethod
  def top(cls) -> NameCtx:
    """
    Return the top of the stack.
    Assumes that it exists.
    """
    assert cls.stack
    return cls.stack[-1]

  @classmethod
  def current_ctx(cls) -> NameCtx:
    """
    Return the current context.
    This is the top from the stack with is_subnet_ctx.
    """
    top = cls.top()
    if not top.is_subnet_ctx:
      assert top.parent and top.parent.is_subnet_ctx
      return top.parent
    assert top.is_subnet_ctx
    return top

  @classmethod
  def new_root(cls) -> NameCtx:
    """
    Create new root name context
    """
    ctx = NameCtx(parent=None)
    ctx.is_subnet_ctx = True
    return ctx

  def __init__(self, *,
               maker: Optional[ILayerMaker] = None,
               suggested_name: Optional[str] = None,
               name: Optional[str] = None,
               parent: Optional[NameCtx] = NotSpecified):
    self.maker = maker
    self.layer_ref = None  # type: Optional[LayerRef]
    self.layer = None  # type: Optional[Layer]
    self.is_subnet_ctx = False
    self.is_repeated_call = False
    self.childs = {}  # type: Dict[str, NameCtx]
    self.parent = parent if parent is not NotSpecified else (self.current_ctx() if self.stack else None)
    if not name:
      if suggested_name:
        name = self._get_unique_name(suggested_name)
      elif self.parent:
        name = self._get_name()
    self.name = name
    if self.parent:
      assert self.name
      assert self.parent.is_subnet_ctx
      assert self.name not in self.parent.childs
      self.parent.childs[self.name] = self

  @classmethod
  def get_from_call(cls, *, name: Optional[Union[str, NameCtx]], maker: ILayerMaker) -> NameCtx:
    """
    This is used e.g. for user module or maker calls.
    The name argument can either be a predefined name ctx, or a suggested name.
    """
    if isinstance(name, NameCtx):
      if name.maker is None:
        name.maker = maker
      else:
        assert name.maker is maker
      return name
    assert not name or isinstance(name, str)
    return NameCtx(maker=maker, suggested_name=name)

  def __repr__(self):
    ls = self.get_abs_name_ctx_list()
    debug_name = "/".join(repr(ctx.name) for ctx in ls)
    return f"<{self.__class__.__name__} maker:{self.maker} name:{debug_name} root:{id(ls[0]):x}>"

  def extend_reserved_names(self, names: Set[str]):
    """
    Extend reserved child names.
    """
    # Do not update inplace because we want an own instance on self.
    self._ReservedNames = self._ReservedNames | names

  def make_net_dict(self) -> NetDictRaw:
    """
    Create net dict.
    """
    net_dict = {}
    for key, value in self.childs.items():
      if value.layer:
        net_dict[key] = value.layer.layer_dict
    return net_dict

  def make_default_output(self, ref: LayerRef) -> LayerRef:
    """
    Assume this is a subnet, and make a default output.
    """
    from .layers import copy
    assert self.is_subnet_ctx
    assert "output" not in self.childs
    return copy(ref, name=self.get_child("output"))

  def get_abs_name_ctx_list(self) -> List[NameCtx]:
    """
    Return list [root name ctx, ..., self].
    """
    ls = []
    cur = self
    while cur:
      ls.append(cur)
      cur = cur.parent
    return list(reversed(ls))

  def get_abs_name(self) -> str:
    """
    Return absolute layer name starting from root context.
    """
    ls = self.get_abs_name_ctx_list()
    assert len(ls) >= 2 and not ls[0].name and ls[-1] is self and ls[-1].name
    return "/".join(ctx.name for ctx in ls[1:])

  def get_name_in_current_ctx(self) -> str:
    """
    Get layer name valid for current scope.
    """
    cur_scope = NameCtx.current_ctx()
    if self.parent is cur_scope:  # fast path
      return self.name
    cur_scope_abs = cur_scope.get_abs_name_ctx_list()
    self_name_abs = self.get_abs_name_ctx_list()
    assert cur_scope_abs[0] is self_name_abs[0]  # same root
    common_len = 0
    max_common_len = min(len(cur_scope_abs), len(self_name_abs))
    while common_len < max_common_len and cur_scope_abs[common_len] is self_name_abs[common_len]:
      common_len += 1
    prefix = "base:" * (len(cur_scope_abs) - common_len)
    postfix = "/".join([ctx.name for ctx in self_name_abs[common_len:]])
    return prefix + postfix

  def get_child(self, name: str) -> NameCtx:
    """
    Makes sure the child exists.
    """
    if name in self.childs:
      return self.childs[name]
    else:
      return NameCtx(name=name, parent=self)  # also registers in self.childs

  def get_child_with_layer_ref(self, name: str) -> NameCtx:
    """
    Makes sure the child exists, including a corresponding layer ref.
    Creates the child together with a layer ref if it does not exist yet.
    """
    child = self.get_child(name)
    if not child.layer_ref:
      layer_ref = LayerRef(name_ctx=child)
      assert child.layer_ref is layer_ref
    return child

  def get_child_layer_ref(self, name: str) -> LayerRef:
    """
    Get child layer ref. Makes sure it exists.
    """
    return self.get_child_with_layer_ref(name).layer_ref

  def __enter__(self):
    if self.parent:
      assert self.stack[-1] is self.parent, f"{self}.__enter__: stack {self.stack} top is not parent {self.parent}"
    else:
      assert not self.stack, f"{self}.__enter__ without parent, unexpected stack {self.stack}"
    self.stack.append(self)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert self.stack[-1] is self, f"{self}.__exit__: stack {self.stack} top is not self"
    self.stack.pop(-1)

  def _get_name(self) -> str:
    assert self.parent and self.maker
    if self.parent.maker:
      reserved_names = set(self.parent.childs.keys()) | self._ReservedNames
      for key, value in vars(self.parent.maker).items():
        if key in reserved_names:
          continue
        if value is self.maker:
          return key
    return self._get_unique_name()

  def _get_suggested_name(self) -> str:
    assert self.parent and self.maker
    # Check parent maker (or module) whether it has our maker (or module) as attrib,
    # and use this attrib name.
    # Check all parents for cases like Loop.
    parent = self.parent
    while parent:
      if parent.maker:
        for key, value in vars(parent.maker).items():
          if value is self.maker:
            return key
      parent = parent.parent
    # Check potential previous calls, and reuse their name.
    for call in self.maker.calls:
      if call is self:
        continue  # ignore this
      if call.name_ctx.parent is self.parent:
        return call.name_ctx.name
    # Fallback to the canonical name.
    return self.maker.get_canonical_name()

  def _get_unique_name(self, suggested_name: Optional[str] = None) -> str:
    name = suggested_name or self._get_suggested_name()
    reserved_names = set(self.parent.childs.keys()) | self._ReservedNames
    if self.parent.maker:
      reserved_names |= set(vars(self.parent.maker).keys())
    if name not in reserved_names:
      return name
    i = 0
    while True:
      name_ = f"{name}_{i}"
      if name_ not in reserved_names:
        return name_
      i += 1
