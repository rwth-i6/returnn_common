"""
Base interfaces.

The core interfaces for the user are:

* :class:`Module` and using :func:`make_layer` to directly create a RETURNN layer via dict.
  We recommend using this only for directly wrapping RETURNN layers
  and not for any higher-level logic,
  which should be done as a :class:`Module`.

* :class:`Module`, to write PyTorch-style code, which acts like a subnetwork.
  We recommend using this as the base interface
  for any higher-level interfaces
  (such as a generic decoder interface).
  Use :func:`scoped` as a decorator for the ``__call__`` method.

Instances of both objects can be called directly,
and return instances of type :class:`LayerRef`,
which can be thought of as analogue to :class:`torch.Tensor` or :class:`tf.Tensor`.

Use ``x.mark_as_loss()`` to mark some output (layer ref) as a loss.

The root network should be a :class:`Module`,
and then you can use ``make_root_net_dict()``
to get the network dict.
Code example::

    class Network(nn.Module):
      def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(nn.FeatureDim("lstm-out", 1024))

      @nn.scoped
      def __call__(self, x: nn.LayerRef) -> nn.Layer:
        y = self.lstm(x)
        return y

    net = Network()
    net_dict = make_root_net_dict(net, "data")

---

Code conventions:

- Usual, as in RETURNN, PEP8, 2-space indents, 120 char line limit.
- Pure interface classes are prefixed with `I`.
  (`Module` is an exception because this is made analogue to PyTorch).

"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union, Set, Sequence, Iterator, Iterable
from returnn.tf.util.data import *  # Dim, Data, and others
from returnn.util.basic import NotSpecified, OptionalNotImplementedError
from tensorflow.python.util import nest


LayerDictRaw = Dict[str, Any]
LayerRefRaw = str
NetDictRaw = Dict[str, LayerDictRaw]
RawTensorTypes = Union[int, float, complex, bool, str]

_min_returnn_behavior_version = 11


class LayerRef:
  """
  Refers to a layer.

  An instance of this class can be treated very much like a tensor.
  It supports all the common unary and binary math operations such as addition.
  This is the intended view point for the user,
  to treat instances of this class like a tensor.

  For most layers, instead of just having an instance of :class:`LayerRef`,
  you would instead directly have an instance of :class:`Layer`.

  You do not create instances of this object explicitly
  but they are created via :func:`get_special_layer` or :class:`NameCtx.get_child_layer_ref`,
  or layers (:class:`Layer`) via :func:`make_layer`.
  """

  def __init__(self, *, name_ctx: NameCtx, data: Data):
    self.parent_modules = []  # type: List[Tuple[Module, str]]  # with attr
    self.name_ctx = name_ctx
    assert name_ctx.layer_ref is None
    name_ctx.layer_ref = self
    self.data = data

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.name_ctx}>"

  @property
  def shape(self) -> Set[Dim]:
    """
    :return: shape (set of dims)
    """
    return self.data.dim_tags_set_implicit

  @property
  def dtype(self) -> str:
    """
    :return: data type (e.g. "float32")
    """
    return self.data.dtype

  @property
  def dim(self) -> Optional[Dim]:
    """
    :return: feature dim
    """
    return self.data.feature_dim_or_sparse_dim

  def get_name_in_current_ctx(self) -> str:
    """
    :return: RETURNN layer name, valid in the current active name context.
    """
    return self.get_name_in_ctx(ctx=NameCtx.current_ctx())

  def get_name_in_ctx(self, ctx: NameCtx) -> str:
    """
    :return: RETURNN layer name in the given name context.
    """
    if not self.name_ctx.parent and ctx != self.name_ctx:
      # We allow creating name ctx early without having a known parent,
      # such as for Parameter, which might be created outside a name context,
      # or in an unrelated name context.
      # We caught this case here, and now assign some parent.
      assert self.parent_modules  # cannot assign parent without parent modules
      for parent_module, attr in self.parent_modules:
        if getattr(parent_module, attr, None) is not self:
          continue  # might have been reset later...
        if parent_module.calls:
          self.name_ctx.assign_parent(parent_module.calls[0], attr)
          break
      assert self.name_ctx.parent, f"{self.parent_modules}"  # could not find parent
    return self.name_ctx.get_name_in_ctx(ctx=ctx)

  def get_abs_name(self) -> str:
    """
    :return: absolute RETURNN layer name starting from root context.
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
    from ._generated_layers import _eval
    return _eval(self, eval="-source(0)", name="neg")

  def __invert__(self) -> LayerRef:
    from ._generated_layers import _eval
    return _eval(self, eval="tf.logical_not(source(0))", name="invert")

  def __pow__(self, other: Union[RawTensorTypes, LayerRef], modulo=None) -> LayerRef:
    assert modulo is None
    from ._generated_layers import _eval
    return _eval([self, convert_to_layer_ref(other)], eval="tf.math.pow(source(0), source(1))", name="pow")

  def __rpow__(self, other: Union[RawTensorTypes, LayerRef], modulo=None) -> LayerRef:
    assert modulo is None
    from ._generated_layers import _eval
    return _eval([convert_to_layer_ref(other), self], eval="tf.math.pow(source(0), source(1))", name="pow")

  def __and__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="logical_and", name="logical_and")

  def __or__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _combine
    return _combine([self, convert_to_layer_ref(other)], kind="logical_or", name="logical_or")

  def __abs__(self) -> LayerRef:
    from ._generated_layers import _eval
    return _eval(self, eval="tf.abs(source(0))", name="abs")

  def __ceil__(self) -> LayerRef:
    from ._generated_layers import _eval
    return _eval(self, eval="tf.math.ceil(source(0))", name="ceil")

  def __floor__(self) -> LayerRef:
    from ._generated_layers import _eval
    return _eval(self, eval="tf.math.floor(source(0))", name="floor")

  def __floordiv__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _eval
    return _eval([self, convert_to_layer_ref(other)], eval="tf.math.floordiv(source(0), source(1))", name="floordiv")

  def __eq__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _compare
    return _compare([self, convert_to_layer_ref(other)], kind="equal", name="equal")

  def __ne__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _compare
    return _compare([self, convert_to_layer_ref(other)], kind="not_equal", name="not_equal")

  def __lt__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _compare
    return _compare([self, convert_to_layer_ref(other)], kind="less", name="less")

  def __le__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _compare
    return _compare([self, convert_to_layer_ref(other)], kind="less_equal", name="less_equal")

  def __gt__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _compare
    return _compare([self, convert_to_layer_ref(other)], kind="greater", name="greater")

  def __ge__(self, other: Union[RawTensorTypes, LayerRef]) -> LayerRef:
    from ._generated_layers import _compare
    return _compare([self, convert_to_layer_ref(other)], kind="greater_equal", name="greater_equal")


class Layer(LayerRef):
  """
  Represents a layer and its output, created by :func:`make_layer`.
  You would not create an instance of this explicitly.
  """

  def __init__(self, *, layer_dict: LayerDictRaw, name_ctx: NameCtx, predefined_out_data: Optional[Data] = None):
    if predefined_out_data:
      data = predefined_out_data
    else:
      data = _data_from_layer_dict(layer_dict)
    super(Layer, self).__init__(name_ctx=name_ctx, data=data)
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


class Parameter(Layer):
  """
  This represents a (potential trainable) parameter,
  aka ``tf.Variable`` in TensorFlow,
  wrapping to ``VariableLayer`` in RETURNN.
  """
  def __init__(self, shape: Sequence[Dim], dtype: str = "float32"):
    if not all(isinstance(dim, Dim) for dim in shape):
      raise TypeError(f"shape {shape} must be a sequence of Dim")
    if not all(isinstance(dim.dimension, int) for dim in shape):
      raise ValueError(f"shape {shape} must be static")
    # Note: At creation time, we don't know the name yet.
    # The name will be inferred by the parent modules and the attribute chain.
    name_ctx = NameCtx(name="parameter", parent=None)  # this is incomplete and will be configured later
    data = Data("parameter", dim_tags=list(shape), dtype=dtype)
    super(Parameter, self).__init__(
      layer_dict={"class": "variable", "shape": list(shape), "dtype": dtype},
      predefined_out_data=data,
      name_ctx=name_ctx)


def scoped(func):
  """
  Decorator to create a new scope (subnetwork) for the function.
  This would be used for modules.
  """
  assert callable(func)

  def _wrapper(*args, name: Optional[Union[str, NameCtx]] = None, **kwargs):
    if args and isinstance(args[0], Module):
      self = args[0]
    else:
      self = _Functional(func)
    from . import copy
    with NameCtx.get_from_call(module=self, name=name) as name_ctx:
      name_ctx.is_subnet_ctx = True
      res = func(*args, **kwargs)
      if name_ctx.parent is None:  # root
        # special logic, no output layers, no subnetwork layer needed
        self.calls.append(name_ctx)
        return res
      if isinstance(res, LayerRef):
        out = copy(res, name=name_ctx.get_child("output"))
      else:
        # we return more than one layer (thus also working on other layers of the subnet, that are not output)
        # by convention: first layer is the output layer
        res_flat = nest.flatten(res)
        out = copy(res_flat[0], name=name_ctx.get_child("output"))
      assert out.data
      # Now create the subnetwork layer itself.
      subnet_layer = make_layer(
        {"class": "subnetwork", "from": [], "subnetwork": name_ctx.make_net()},
        name=name_ctx, predefined_out_data=out.data)
    if isinstance(res, LayerRef):
      return subnet_layer  # maybe nicer to return subnet layer
    return res

  _wrapper.__name__ = func.__name__
  _wrapper.__qualname__ = func.__qualname__
  return _wrapper


class Module:
  """
  This can represent a subnetwork in RETURNN.

  You can write PyTorch-like code here, like::

      class MyModule(nn.Module):

        def __init__(self, dim: int, activation=tanh):
          super().__init__()
          self.linear = Linear(dim)
          self.activation = activation

        @nn.scoped_method
        def __call__(self, x: LayerRef) -> LayerRef:
          x_ = x
          x = layer_norm(x)
          x = self.linear(x)
          x = self.activation(x)
          return x_ + x

  It is also used to wrap existing RETURNN layers
  by using :func:`make_layer`.

  A RETURNN layer also has some specific input and output,
  and usually its own parameters.

  This is in contrast to PyTorch or Keras, where a module or layer
  has params, but getting some output for some input
  requires an additional `forward` or `__call__` call,
  which can be called multiple times.
  Every such call would then share the same module parameters.

  :class:`Module` is similar to PyTorch/Keras
  in that it can be called multiple times.
  Every call would create a RETURNN layer,
  where every call after the first would share the params
  with the first layer,
  via the RETURNN ``reuse_params`` layer option.

  A user would create an instance and then call it,
  and get :class:`Layer` instances.
  The naming logic of created layers
  is handled via :class:`NameCtx`.

  A developer which wants to derive its own module
  would overwrite :func:`__call__` and use :func:`make_layer`.
  Usually :func:`make_layer` is never needed though,
  as all standard RETURNN layers are already wrapped,
  and any potential operation should be possible to be defined
  using the standard RETURNN layers.
  For one-time usages to wrap RETURNN layers, you might not need an own :class:`Module`
  and you could use :func:`make_layer` directly instead.
  For defining own modules (subnetworks)
  based on existing modules or layers,
  see :class:`Module`.
  """
  layer_name_scope = NotSpecified  # type: Union[NotSpecified, str]
  default_name: Optional[str] = None

  def __init__(self):
    """
    By convention, any options to the module or module are passed to the constructor,
    and potential changing inputs (other layers)
    are passed to :func:`__call__` (:func:`make_layer_dict`).
    """
    # Actually we would want an ordered set for parents, but Python does not provide this.
    # We abuse a dict as a set. This is ordered since Python 3.6, see #43.
    # Also note that the current code does not clean this up when you do delattr later or so.
    self._parents = {}  # type: Dict[Tuple[Module, str], None]  # (parent,attrib) -> None
    self.calls = []  # type: List[NameCtx]

  def __repr__(self):
    return f"<{self.__class__.__name__}>"

  def default_initial_state(self) -> LayerState:
    """
    :return: default initial state, to be used if the module (layer) has recurrent (hidden) state.
      When a module has recurrent state,
      the convention is to return a tuple with instance :class:`LayerState` as the last item,
      and to accept the ``state`` argument with a :class:`LayerState` with the same nested structure.
      This can be a nested structure and should match the structure of the ``state`` argument and returned value.
    """
    raise OptionalNotImplementedError

  def get_default_name(self) -> str:
    """
    Get a default layer name (used when we do not have a Module attribute pointing to this).
    """
    if self.default_name:
      return self.default_name
    name = self.__class__.__name__
    if name.startswith("_"):
      name = name[1:]
    if name[:1].isupper():
      from returnn.util.basic import camel_case_to_snake_case
      name = camel_case_to_snake_case(name)
    return name

  @scoped
  def __call__(self, *args, **kwargs) -> Union[Layer, Tuple[Layer, LayerState], Any]:
    raise NotImplementedError

  def __setattr__(self, key: str, value):
    super().__setattr__(key, value)
    if isinstance(value, Module):
      value._parents[(self, key)] = None
    if isinstance(value, LayerRef):
      if (self, key) not in value.parent_modules:
        value.parent_modules.append((self, key))

  def parents_with_attr(self) -> Iterator[Tuple[Module, str]]:
    """
    Get all (immediate) parent modules, and the attrib name which points to us
    """
    # We rely on deterministic order of dict.
    for parent, attr in self._parents.keys():
      # We currently don't do proper cleanup of _parents via delattr etc,
      # so explicitly check.
      if getattr(parent, attr, None) is self:
        yield parent, attr

  def children(self, *, recurse: bool = True) -> Iterator[Module]:
    """
    Get all (immediate) children modules
    """
    for name, child in self.named_children(recurse=recurse):
      yield child

  def named_children(self,
                     *, recurse: bool = True, memo: Optional[Set[Module]] = None, prefix: str = ''
                     ) -> Iterator[Tuple[str, Module]]:
    """
    Get all children modules
    """
    if memo is None:
      memo = set()
    if self not in memo:
      for name, module in vars(self).items():
        if not isinstance(module, Module):
          continue
        sub_prefix = prefix + ('.' if prefix else '') + name
        memo.add(module)
        yield sub_prefix, module
        if recurse:
          for name_, mod_ in module.named_children(recurse=True, memo=memo, prefix=sub_prefix):
            yield name_, mod_

  def named_parameters(self, *, recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
    """
    Get all children parameters
    """
    memo = set()  # over name contexts because we cannot hash layer refs

    def _iter_params(module: Module, prefix: str) -> Iterator[Tuple[str, Parameter]]:
      for key, value in vars(module).items():
        if isinstance(value, Parameter) and value.name_ctx not in memo:
          sub_prefix = prefix + ('.' if prefix else '') + key
          memo.add(value.name_ctx)
          yield sub_prefix, value

    for name, param in _iter_params(module=self, prefix=''):
      yield name, param
    if recurse:
      for child_prefix, child_mod in self.named_children(recurse=True):
        for name, param in _iter_params(module=child_mod, prefix=child_prefix):
          yield name, param

  @property
  def has_parameters(self):
    """
    Whether this module has variables
    """
    for _, _ in self.named_parameters(recurse=True):
      return True
    return False


class _Functional(Module):
  """
  Used via :func:`scoped`.
  """

  def __init__(self, func):
    super().__init__()
    self.func = func

  def get_default_name(self) -> str:
    """default name"""
    return self.func.__qualname__

  @scoped
  def __call__(self, *args, **kwargs):  # not really needed but nicer to define it
    return self.func(*args, **kwargs)


class LayerState(dict):
  """
  Covers all the state of a layer,
  i.e. exactly what needs to be stored and passed into the module or module
  next time you call it as initial state.

  This behaves somewhat like a namedtuple, although we derive from dict.
  """
  def __init__(self, *args, **kwargs):
    if kwargs:
      assert not args
      super().__init__(**kwargs)
    elif args:
      assert len(args) == 1
      if isinstance(args[0], dict):
        super().__init__(**args[0])
      else:
        super().__init__(state=args[0])
    else:
      super().__init__()

  def __repr__(self):
    return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for (k, v) in self.items())})"

  def __getattr__(self, item):
    if item in self:
      return self[item]
    raise AttributeError(f"{self}.{item}")

  def __setattr__(self, key, value):
    self[key] = value


# noinspection PyAbstractClass
class ReturnnWrappedLayerBase(Module):
  """
  Base class for all automatically wrapped layers.
  """
  returnn_layer_class: Optional[str] = None
  has_recurrent_state: bool = False
  has_variables: bool = False

  @staticmethod
  def returnn_layer_get_recurrent_state(layer: Layer) -> LayerState:
    """
    :returns: the recurrent state

    You might override this in case the state is more complex,
    and return some named tuple or any other hierarchical structure.
    """
    from ._generated_layers import _get_last_hidden_state
    # Note that this is actually layer specific.
    # We try to use a number of heuristics to get it right for the common cases.
    name = f"{layer.name_ctx.name}_state"
    out_dim = layer.layer_dict["out_dim"]
    if layer.layer_dict["class"] == "rec" and isinstance(layer.layer_dict["unit"], str):
      if "lstm" in layer.layer_dict["unit"].lower():
        h = _get_last_hidden_state(layer, out_dim=out_dim, key="h", name=f"{name}_h")
        c = _get_last_hidden_state(layer, out_dim=out_dim, key="c", name=f"{name}_c")
        return LayerState(h=h, c=c)
    return LayerState(_get_last_hidden_state(layer, out_dim=out_dim, name=name))

  def default_initial_state(self) -> LayerState:
    """
    :return: default initial state
    """
    assert self.has_recurrent_state
    # Match the logic of _get_recurrent_state above.
    if self.returnn_layer_class == "rec":
      unit = getattr(self, "unit")
      if isinstance(unit, str):
        if "lstm" in unit.lower():
          return LayerState(h=0, c=0)  # TODO get real shape... how to get batch dim?
    raise NotImplementedError(f"{self}.default_initial_state")

  @staticmethod
  def handle_recurrent_state(args: Dict[str, Any], *,
                             axis: Dim,
                             state: Optional[Union[LayerRef, Dict[str, LayerRef], NotSpecified]] = NotSpecified,
                             initial_state: Optional[Union[LayerRef, Dict[str, LayerRef], NotSpecified]] = NotSpecified,
                             ):
    """
    Update the args to include either state or initial_state,
    depending on whether we operate per step or on an axis.

    :param args: layer arguments
    :param axis: single_step_dim specifies to operate for a single step
    :param state: prev state when operating a single step
    :param initial_state: initial state when operating on an axis
    """
    if axis == single_step_dim:
      assert state is not NotSpecified
      assert initial_state is NotSpecified
      args['state'] = state
    else:
      assert state is NotSpecified
      if initial_state is not NotSpecified:
        args['initial_state'] = initial_state


def make_layer(layer_dict: LayerDictRaw, *,
               name: Optional[Union[str, NameCtx]] = None,
               module: Optional[Module] = None,
               predefined_out_data: Optional[Data] = None) -> Layer:
  """
  Creates the layer. This also registers the layer instance in the top name ctx.
  When no name is given, this assumes that the top name ctx corresponds to this module.

  If a layer has params, and you want the param sharing logic,
  you should instead derive a new class from :class:`Module`.
  Usually, you do not need either of these,
  as all standard layers should already be wrapped,
  and it should be possible to define any possible logic
  using that.
  (If this is not the case, please report an issue.)

  :param LayerDictRaw layer_dict: can contain :class:`LayerRef` instances
  :param str|NameCtx|None name:
    if str: (suggested) layer name. if given, will create a new :class:`NameCtx`
    if NameCtx, will use this.
  :param Module|None module: if given, will create new name scope with this module
  :param Data|None predefined_out_data: normally we can derive the out data automatically.
    If this should be skipped, you can pass this explicitly.
  """
  if isinstance(name, str) or module:
    assert not name or isinstance(name, str)
    name_ctx = NameCtx.get_from_call(module=module, name=name)
    return make_layer(layer_dict=layer_dict, name=name_ctx, predefined_out_data=predefined_out_data)
  elif isinstance(name, NameCtx):
    name_ctx = name
    if NameCtx.top() is name:
      pass  # go on
    else:
      with name_ctx:
        return make_layer(layer_dict=layer_dict, predefined_out_data=predefined_out_data)
  else:
    name_ctx = NameCtx.top()
  assert not name_ctx.layer_ref and not name_ctx.layer  # not yet assigned
  layer_dict = layer_dict.copy()

  if name_ctx.module and name_ctx.module.has_parameters:
    # We must check whether the RETURNN abs layer name is consistent with our module naming hierarchy,
    # and make it consistent if not (https://github.com/rwth-i6/returnn_common/issues/25).
    if name_ctx.is_root:
      pass  # nothing to do
    else:
      # The parent name ctx RETURNN layer will also have the right name_scope set,
      # so this layers name scope default is simply based on that.
      layer_abs_name_scope_parent = name_ctx.parent.layer_abs_name_scope
      if layer_abs_name_scope_parent:
        layer_abs_name_scope_parent += "/"
      layer_abs_name_scope_default = layer_abs_name_scope_parent + name_ctx.name
      if layer_abs_name_scope_default != name_ctx.layer_abs_name_scope:  # default does not match what we require
        assert "name_scope" not in layer_dict
        if name_ctx.layer_abs_name_scope == name_ctx.parent.layer_abs_name_scope:
          layer_dict["name_scope"] = ""
        elif name_ctx.layer_abs_name_scope.startswith(layer_abs_name_scope_parent):  # can use relative
          layer_dict["name_scope"] = name_ctx.layer_abs_name_scope[len(layer_abs_name_scope_parent):]
        else:  # must use absolute
          layer_dict["name_scope"] = "/" + name_ctx.layer_abs_name_scope

  name_ctx.is_subnet_ctx = False
  layer = Layer(layer_dict=layer_dict, name_ctx=name_ctx, predefined_out_data=predefined_out_data)
  if name_ctx.module:
    name_ctx.module.calls.append(name_ctx)
  return layer


def convert_to_layer_ref(x: Union[LayerRef, int, float, complex, bool, str]) -> LayerRef:
  """
  In case it is not a layer ref yet, it will make some constant.
  """
  if isinstance(x, LayerRef):
    return x
  from . import constant
  return constant(value=x)


def make_root_net_dict(model: Module, *args: Data, **kwargs: Data) -> Dict[str, Any]:
  """
  Make net dict, to be used as the main RETURNN network, not within a subnetwork.
  Any passed arguments are keys of extern data,
  and are forwarded to the module.
  """
  assert isinstance(model, Module)
  from . import copy
  with NameCtx(module=model, parent=None) as name_ctx:
    name_ctx.is_subnet_ctx = True
    args = tuple(get_extern_data(arg) for arg in args)
    kwargs = {key: get_extern_data(value) for (key, value) in kwargs.items()}
    res = model(*args, **kwargs, name=name_ctx)
    if "output" not in name_ctx.children:
      if isinstance(res, LayerRef):
        copy(res, name=name_ctx.get_child("output"))
      else:
        res_list = nest.flatten(res)
        assert res_list and isinstance(res_list[0], LayerRef)
        copy(res_list[0], name=name_ctx.get_child("output"))
    net = name_ctx.make_net()
  config = {
    "network": net.make_net_dict_raw(),
    "extern_data": {
      data_key: {key: value for (key, value) in data.get_kwargs().items() if key not in {"name"}}
      for (data_key, data) in name_ctx.extern_data.items()},
    "behavior_version": _min_returnn_behavior_version,
  }
  return config


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
               max_seq_len: Optional[Union[str, int, callable]] = NotSpecified,
               optimize_move_layers_out: Optional[bool] = NotSpecified,
               unroll: bool = NotSpecified,
               axis: Optional[Dim] = NotSpecified,
               debug: Optional[bool] = NotSpecified,
               name: str = "loop"
               ):
    super(Loop, self).__init__()
    self.extra_opts = {
      key: value for (key, value) in locals().items()
      if value is not NotSpecified and key not in {"self", "__class__", "name"}}
    self.layer_module = _LoopLayerModule(loop=self)
    self.name_ctx = NameCtx(module=self.layer_module, suggested_name=name, parent=NameCtx.current_ctx())
    self.name_ctx.is_subnet_ctx = True
    self.name_ctx.extend_reserved_names({"output", "end"})
    self._entered_scope = False
    self._exited_scope = False
    self._state = _StateHolder(loop=self)
    self.unstacked_refs = []  # type: List[LayerRef]
    self.outputs = []  # type: List[LayerRef]
    self._has_given_axis = bool(axis)
    if not axis:
      axis = SpatialDim(f"{name}-dim")
    self.axis = axis
    self.end_ref = None  # type: Optional[LayerRef]

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.name_ctx.get_abs_name_repr()}>"

  def __enter__(self) -> Loop:
    assert not self._entered_scope, f"{self}: cannot enter twice"
    self._entered_scope = True
    self.name_ctx.__enter__()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert not self._exited_scope, f"{self}: cannot exit twice"
    self._exited_scope = True
    try:
      if not exc_type:
        if not self.outputs:  # stack or last was called at least once, so we have some output
          raise Exception(f"{self}: call `stack` or `last` at least once to define some output")
        if not self.end_ref and not self.unstacked_refs:
          raise Exception(f"{self}: call `unstack` or `end` at least once to define the loop length")
        # Make sure there is an "output" layer. (Similar as for Module with subnetwork.)
        if "output" not in self.name_ctx.children:
          from . import copy
          copy(self.outputs[0], name=self.name_ctx.get_child("output"))
    finally:
      self.name_ctx.__exit__(exc_type, exc_val, exc_tb)
    if not exc_type:
      self.layer_module()  # create the rec layer itself

  @property
  def state(self) -> Union[_StateHolder, LayerState]:
    """state holder inside the loop"""
    if not self._exited_scope:
      return self._state
    # noinspection PyProtectedMember
    return self._state._get_last()

  @state.setter
  def state(self, initial_state: LayerState):
    assert len(self._state) == 0, f"can only assign {self}.state once for the initial state"
    for key, value in initial_state.items():
      self._state[key] = value

  def unstack(self, source: LayerRef, *,
              name: Optional[str] = None
              ) -> LayerRef:
    """
    Unrolls over the specified axis, and provides each frame in each loop iteration.
    The axis can be specified globally for the :class:`Loop` instance (recommended)
    or locally here (not recommended).
    """
    from . import rec_unstack
    assert self._has_given_axis, "%s: unstack() requires a given axis" % self
    opts = {"axis": self.axis}
    res = rec_unstack(source, name=name, **opts)
    self.unstacked_refs.append(res)
    return res

  def stack(self, source: LayerRef, *, name: Optional[str] = None) -> LayerRef:
    """
    Accumulates the frames of source within the loop,
    to make it accessible outside the loop.
    """
    from . import copy
    if not name and "output" not in self.name_ctx.children:
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
    assert isinstance(source, Layer)
    source.layer_dict["need_last"] = True
    sub_layer_name = source.name_ctx.get_name_in_ctx(self.name_ctx)
    with self.name_ctx.parent:  # need to be outside the loop
      return make_layer(
        {"class": "rec_last_output", "rec_layer": self.name_ctx.layer_ref, "sub_layer_name": sub_layer_name},
        name=name or sub_layer_name.replace("/", "_"))

  def end(self, source: LayerRef, *, include_eos: bool) -> LayerRef:
    """
    For loops with dynamic ending condition (which might not use unstack),
    this defines the ending condition.

    :param source: the ending condition
    :param include_eos: if True, the last() and stack() function include the current ending frame, otherwise not
    """
    assert not self.end_ref, f"{self}.end() can only be called once"
    self.extra_opts["include_eos"] = include_eos
    from . import copy
    self.end_ref = copy(source, name=self.name_ctx.get_child("end"))
    return self.end_ref


class _LoopLayerModule(Module):
  layer_name_scope = ""

  def __init__(self, loop: Loop):
    super(_LoopLayerModule, self).__init__()
    self.loop = loop

  def __call__(self) -> Layer:
    """
    Makes layer dict for this loop, i.e. a RecLayer.
    """
    name_ctx = self.loop.name_ctx
    out = name_ctx.children["output"].layer_ref
    return make_layer(
      {"class": "rec", "from": [], "unit": name_ctx.make_net(), **self.loop.extra_opts},
      name=name_ctx,
      predefined_out_data=out.data.copy_add_dim_by_tag(self.loop.axis, unbroadcast=True, axis=0))

  def named_children(self, *,
                     recurse: bool = True, memo: Optional[Set[Module]] = None, prefix: str = ''
                     ) -> Iterator[Tuple[str, Module]]:
    """
    Children
    """
    # We rely on deterministic order of dict.
    for name, sub_name_ctx in self.loop.name_ctx.children.items():
      if sub_name_ctx.module:
        yield name, sub_name_ctx.module


class PrevLayerRef(LayerRef):
  """
  Refers to a layer from the previous loop iteration.
  """
  @classmethod
  def get_prev_ref(cls, *, cur_layer_name_ctx: NameCtx, initial: LayerRef) -> PrevLayerRef:
    """
    Create prev ref.
    """
    parent_name_ctx = cur_layer_name_ctx.parent
    prev_layer_name_ctx = parent_name_ctx.get_child(f"prev:{cur_layer_name_ctx.name}")
    if prev_layer_name_ctx.layer_ref:
      prev_layer_ref = prev_layer_name_ctx.layer_ref
      assert isinstance(prev_layer_ref, PrevLayerRef)
      assert prev_layer_ref.cur_layer_name_ctx is cur_layer_name_ctx
    else:
      prev_layer_ref = PrevLayerRef(
        name_ctx=prev_layer_name_ctx, cur_layer_name_ctx=cur_layer_name_ctx, data=initial.data)
      assert prev_layer_name_ctx.layer_ref is prev_layer_ref
    return prev_layer_ref

  def __init__(self, *, name_ctx: NameCtx, cur_layer_name_ctx: NameCtx, data: Data):
    # At the time we instantiate this, cur_layer_name_ctx.layer probably does not exist yet.
    super().__init__(name_ctx=name_ctx, data=data)
    self.cur_layer_name_ctx = cur_layer_name_ctx


class _StateHolder:
  def __init__(self, loop: Loop):
    self._loop = loop
    self._state = {}  # type: Dict[str, State]

  def __repr__(self):
    return f"{self._loop}.state"

  def _get_state(self, name: str) -> State:
    if name in self._state:
      return self._state[name]
    raise AttributeError(f"{self}: Unknown state attrib {name!r}. Assign the initial state first.")

  def _get_last(self) -> LayerState:
    return LayerState({key: value.get_last() for (key, value) in self._state.items()})

  def __getitem__(self, item):
    return self._get_state(item).get()

  def __setitem__(self, key, value):
    if isinstance(value, State):
      # noinspection PyProtectedMember
      value._set_name_and_loop(name=key, loop=self._loop)
      self._state[key] = value
      return
    self._get_state(key).assign(value)

  def __getattr__(self, item):
    return self[item]

  def __setattr__(self, key, value):
    if key in {"_state", "_loop"}:
      return super().__setattr__(key, value)
    self[key] = value

  def keys(self) -> Iterable[str]:
    """keys"""
    return self._state.keys()

  def __len__(self):
    return len(self._state)


class State:
  """
  Represents some recurrent state, to be used with :class:`Loop`.
  It can also represent some nested hierarchy of states.
  """

  def __init__(self, *, initial: Union[LayerRef, Any]):
    """
    :param initial: some layer-ref, or any kind of nested structure of layers.
    """
    super(State, self).__init__()
    assert initial is not None
    initial = nest.map_structure(convert_to_layer_ref, initial)
    self.initial = initial
    self.loop = None  # type: Optional[Loop]
    self.name = None  # type: Optional[str]
    self.name_ctx = None  # type: Optional[Union[NameCtx, Any]]  # same nested structure as initial
    self.assigned_value = None

  def _set_name_and_loop(self, *, name: str, loop: Loop):
    """
    Assigns the name (internally on first assignment).
    """
    if self.name == name and self.loop is loop:
      return
    assert not self.loop and not self.name and not self.name_ctx  # not yet assigned
    self.loop = loop
    self.name = name
    self.name_ctx = nest.map_structure_with_tuple_paths(
      lambda path, ref: NameCtx(suggested_name='.'.join(str(key) for key in ('state', name) + path)),
      self.initial)

  def assign(self, value):
    """
    Assign the new value for the current iteration.
    """
    assert self.name_ctx is not None
    assert value is not None
    assert self.assigned_value is None, (
      f"Cannot assign the rec state {self.loop}/{self.name} multiple times, "
      f"assigned previously to {self.assigned_value}, now to {value}")
    nest.assert_same_structure(self.initial, value)
    nest.assert_same_structure(self.name_ctx, value)
    self.assigned_value = value

    def _map_ref_to_name_ctx(layer_ref: LayerRef, name_ctx: NameCtx, initial: LayerRef):
      assert isinstance(layer_ref, LayerRef)
      assert isinstance(name_ctx, NameCtx)

      # Potential optimization for RETURNN layers.
      # See ReturnnWrappedLayerBase._get_recurrent_state.
      if isinstance(layer_ref, Layer):
        if layer_ref.layer_dict["class"] == "get_last_hidden_state":
          used_state_eliminate_optimization = False
          key = layer_ref.layer_dict.get("key", "state")
          src = layer_ref.layer_dict["from"]
          assert isinstance(src, Layer)
          src_state_opt = src.layer_dict.get("state")
          if isinstance(src_state_opt, LayerState):
            src_state_for_key = src_state_opt.get(key)
            if isinstance(src_state_for_key, PrevLayerRef):
              if src_state_for_key.cur_layer_name_ctx is name_ctx:
                # The 'state' argument of the rec layer refers to "prev:..." of the state.
                # So we don't need to pass it now.
                used_state_eliminate_optimization = True
                src_state_opt[key] = None
                if all(opt is None for opt in nest.flatten(src_state_opt)):
                  del src.layer_dict["state"]
                # We need to pass the initial_state instead though.
                src_initial_state_opt = src.layer_dict.setdefault("initial_state", LayerState())
                src_initial_state_opt[key] = initial
                # If there is any other code which refers to this state, it can access the passed layer.
                # So anyway pass through.

          if not used_state_eliminate_optimization:
            raise NotImplementedError(
              f"{self}.assign to {layer_ref} on {src}:"
              f" We need https://github.com/rwth-i6/returnn_common/issues/31"
              f" and https://github.com/rwth-i6/returnn/issues/732.")

      _move_layer_ref_to_new_name_ctx(layer_ref=layer_ref, name_ctx=name_ctx)

    nest.map_structure(_map_ref_to_name_ctx, value, self.name_ctx, self.initial)

  @staticmethod
  def _map_name_ctx_to_prev_layer_ref(name_ctx: NameCtx, initial: LayerRef) -> PrevLayerRef:
    assert isinstance(name_ctx, NameCtx)
    return PrevLayerRef.get_prev_ref(cur_layer_name_ctx=name_ctx, initial=initial)

  def get(self):
    """
    Return prev or current value
    """
    assert self.name_ctx is not None
    if self.assigned_value is None:  # not yet assigned
      # Return prev value
      return nest.map_structure(self._map_name_ctx_to_prev_layer_ref, self.name_ctx, self.initial)
    return self.assigned_value

  def _map_name_ctx_to_last_layer_ref(self, name_ctx: NameCtx) -> LayerRef:
    assert isinstance(name_ctx, NameCtx)
    assert name_ctx.layer_ref, f"{self.loop} state {name_ctx} not assigned?"
    assert self.loop.name_ctx.layer_ref, f"{self.loop} not yet exited?"
    return self.loop.last(name_ctx.layer_ref)

  def get_last(self):
    """
    Outside the loop, get the last instance.
    """
    assert self.name_ctx is not None
    assert self.assigned_value is not None
    return nest.map_structure(self._map_name_ctx_to_last_layer_ref, self.name_ctx)


def get_extern_data(data: Data) -> LayerRef:
  """
  Get extern data from root ctx.
  """
  assert isinstance(data, Data)  # the usage was different before. make sure we get this correct
  root_scope = NameCtx.top().root  # must exist
  if data.name not in root_scope.extern_data:
    root_scope.extern_data[data.name] = data
  else:
    assert root_scope.extern_data[data.name] is data
  root_layer_name = f"data:{data.name}"
  return _get_special_layer(root_layer_name, scope=root_scope, data=data)


def _get_special_layer(name: str, *, scope: Optional[NameCtx] = None, data: Data) -> LayerRef:
  """
  Special layer can be "data:..." or whatever.
  """
  if not scope:
    scope = NameCtx.current_ctx()  # must exist
  return scope.get_child_layer_ref(name, data=data)


def _get_sub_layer(layer: LayerRef, name: str, *, data: Data) -> LayerRef:
  """
  Like the "{layer}/{name}" syntax in RETURNN.
  Normally this should only be needed for internal usage.
  """
  return layer.name_ctx.get_child_layer_ref(name, data=data)


class Net:
  """
  Represents a RETURNN (sub) network.
  It can create a net dict when needed.
  """
  def __init__(self, *, name_ctx: NameCtx):
    self.name_ctx = name_ctx

  def _map_elem_resolve(self, obj: Any) -> Any:
    if isinstance(obj, LayerRef):
      return obj.get_name_in_ctx(ctx=self.name_ctx)
    if isinstance(obj, Net):
      return obj.make_net_dict_raw()
    return obj

  def make_net_dict_raw(self) -> NetDictRaw:
    """
    Create raw net dict, not containing any :class:`LayerRef` or :class:`Net` instances anymore.
    """
    net_dict = {}
    # Due to late assignments of name context parents (e.g. for Parameter),
    # the name_ctx.children dict might change while we iterate over it.
    # To avoid that, we iterate over a copy.
    # We must then check if no new children were added.
    while True:
      children = list(self.name_ctx.children.values())
      for sub_name_ctx in children:
        if sub_name_ctx.name in net_dict:
          continue
        if sub_name_ctx.layer:
          layer_dict = sub_name_ctx.layer.layer_dict
          layer_dict = nest.map_structure(self._map_elem_resolve, layer_dict)
          net_dict[sub_name_ctx.name] = layer_dict
      if len(self.name_ctx.children) == len(children):  # we never would delete entries, so this should be safe
        break
    return net_dict


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
               module: Optional[Module] = None,
               suggested_name: Optional[str] = None,
               name: Optional[str] = None,
               parent: Optional[NameCtx] = NotSpecified):
    self.module = module
    self.layer_ref = None  # type: Optional[LayerRef]
    self.layer = None  # type: Optional[Layer]
    self._layer_abs_name_scope = None  # type: Optional[str]
    self.is_subnet_ctx = False
    self.children = {}  # type: Dict[str, NameCtx]
    self.extern_data = {}  # type: Dict[str, Data]
    self.parent = parent if parent is not NotSpecified else (self.current_ctx() if self.stack else None)
    self.name = name  # early assign such that debug repr works later
    if not name:
      if suggested_name:
        name = self._get_unique_name(suggested_name)
      elif self.parent:
        name = self._get_unique_name()
    self.name = name
    if self.parent:
      self.parent._add_child(self)

  @classmethod
  def get_from_call(cls, *, name: Optional[Union[str, NameCtx]], module: Module) -> NameCtx:
    """
    This is used e.g. for user module or module calls.
    The name argument can either be a predefined name ctx, or a suggested name.
    """
    if isinstance(name, NameCtx):
      if name.module is None:
        name.module = module
      else:
        assert name.module is module
      return name
    assert not name or isinstance(name, str)
    return NameCtx(module=module, suggested_name=name)

  def __repr__(self):
    return f"<{self.__class__.__name__} module:{self.module} name:{self.get_abs_name_repr()}>"

  def __hash__(self):
    return hash(id(self))

  def assign_parent(self, parent: NameCtx, suggested_name: str):
    """
    Assign parent to this name context, when it is not set yet.
    """
    assert not self.parent
    self.parent = parent
    self.name = self._get_unique_name(suggested_name)
    self.parent._add_child(self)

  @property
  def root(self) -> NameCtx:
    """
    :return: root name ctx
    """
    root = self
    while root.parent:
      root = root.parent
    return root

  @property
  def is_root(self) -> bool:
    """
    :return: whether this is a root ctx
    """
    return not self.parent

  def extend_reserved_names(self, names: Set[str]):
    """
    Extend reserved child names.
    """
    # Do not update inplace because we want an own instance on self.
    self._ReservedNames = self._ReservedNames | names

  def make_net(self) -> Net:
    """
    Create new (sub) net, an instance of :class:`Net`.
    """
    return Net(name_ctx=self)

  def make_default_output(self, ref: LayerRef) -> LayerRef:
    """
    Assume this is a subnet, and make a default output.
    """
    from . import copy
    assert self.is_subnet_ctx
    assert "output" not in self.children
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
    :return: absolute RETURNN layer name starting from root context.
    """
    ls = self.get_abs_name_ctx_list()
    assert len(ls) >= 2 and not ls[0].name and ls[-1] is self and ls[-1].name
    return "/".join(ctx.name for ctx in ls[1:])

  def get_abs_name_repr(self) -> str:
    """
    :return: Some repr for our absolute name.
    """
    ls = self.get_abs_name_ctx_list()
    if len(ls) == 0:
      debug_name = "???"
    elif len(ls) == 1 and ls[0].name is None:
      debug_name = "/"
    else:
      debug_name = "/".join(repr(ctx.name) if i > 0 or ctx.name is not None else '' for i, ctx in enumerate(ls))
    return debug_name

  @property
  def layer_abs_name_scope(self) -> str:
    """
    :return: layer abs name scope, i.e. the TF name scope of variables
    """
    if self._layer_abs_name_scope is not None:
      return self._layer_abs_name_scope
    assert self.module
    if self.module.layer_name_scope is not NotSpecified:
      assert isinstance(self.module.layer_name_scope, str)
      if self.module.layer_name_scope == "":
        self._layer_abs_name_scope = self.parent.layer_abs_name_scope
      else:
        parent_prefix = self.parent.layer_abs_name_scope
        if parent_prefix:
          parent_prefix += "/"
        self._layer_abs_name_scope = parent_prefix + self.module.layer_name_scope
    else:
      self._layer_abs_name_scope = self._get_abs_canonical_name()
    return self._layer_abs_name_scope

  def _get_abs_canonical_name(self, join_str="/") -> str:
    """
    :param str join_str: maybe "." is more common for attrib chains.
      however, we use "/" as default, to make this consistent to :func:`get_abs_name`.
    :return: unique absolute layer name for the module (module) hierarchy.
      https://github.com/rwth-i6/returnn_common/issues/25
    """
    assert self.module, f"{self} is not assigned to a module (module)"
    root = self.root
    root_module = root.module  # might be None
    assert root_module, f"root name ctx {self.root} is not assigned to a module (module)"
    if root_module is self.module:
      return ""  # special case
    # Do a depth-first search through the parents, starting from self.module, until we find root_module.
    # Use depth-first instead of breadth-first to prefer the first parent when there are multiple.
    queue = [self.module]
    cache = {}  # module -> full name
    while queue:
      module = queue.pop(-1)  # depth-first
      postfix = (join_str + cache[module]) if module in cache else ""
      queue_ext = []
      for parent, attr in module.parents_with_attr():
        if parent in cache:
          continue
        for call in parent.calls:
          if call.root is root:  # same name ctx hierarchy
            assert call.is_root or call.layer_abs_name_scope is not None
            if call.is_root or call.layer_abs_name_scope == "":
              return attr + postfix
            assert call.layer_abs_name_scope
            return call.layer_abs_name_scope + join_str + attr + postfix
        cache[parent] = attr + postfix
        queue_ext.append(parent)
      queue.extend(reversed(queue_ext))
      if root_module in cache:
        break
    if root_module not in cache:
      err_msgs = []
      for module, name in cache.items():
        err_msgs.append(f"  {module}: {name}\n")
      if not err_msgs:
        err_msgs.append(f"  (None, {self.module} has no parent modules)\n")
      raise Exception(
        f"{self}: no abs canonical name found."
        f" Found partial names:\n{''.join(err_msgs)}"
        f"There must be a path of attribs from the root {root_module} to {self.module}.")
    return cache[root_module]

  def get_name_in_ctx(self, ctx: NameCtx) -> str:
    """
    Get layer name valid in given scope.
    """
    if self.parent is ctx:  # fast path
      return self.name
    ctx_scope_abs = ctx.get_abs_name_ctx_list()
    self_name_abs = self.get_abs_name_ctx_list()
    assert ctx_scope_abs[0] is self_name_abs[0]  # same root
    common_len = 0
    max_common_len = min(len(ctx_scope_abs), len(self_name_abs))
    while common_len < max_common_len and ctx_scope_abs[common_len] is self_name_abs[common_len]:
      common_len += 1
    prefix = "base:" * (len(ctx_scope_abs) - common_len)
    postfix = "/".join([ctx.name for ctx in self_name_abs[common_len:]])
    return prefix + postfix

  def get_name_in_current_ctx(self) -> str:
    """
    Get layer name valid for current scope.
    """
    return self.get_name_in_ctx(ctx=NameCtx.current_ctx())

  def _add_child(self, child: NameCtx):
    assert child.name
    assert child.parent is self
    assert child.name not in self.children
    self.children[child.name] = child

  def get_child(self, name: str) -> NameCtx:
    """
    Makes sure the child exists.
    """
    if name in self.children:
      return self.children[name]
    else:
      return NameCtx(name=name, parent=self)  # also registers in self.children

  def get_child_with_layer_ref(self, name: str, *, data: Data) -> NameCtx:
    """
    Makes sure the child exists, including a corresponding layer ref.
    Creates the child together with a layer ref if it does not exist yet.
    """
    child = self.get_child(name)
    if not child.layer_ref:
      layer_ref = LayerRef(name_ctx=child, data=data)
      assert child.layer_ref is layer_ref
    return child

  def get_child_layer_ref(self, name: str, *, data: Data) -> LayerRef:
    """
    Get child layer ref. Makes sure it exists.
    """
    return self.get_child_with_layer_ref(name, data=data).layer_ref

  def __enter__(self):
    self.stack.append(self)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert self.stack[-1] is self, f"{self}.__exit__: stack {self.stack} top is not self"
    self.stack.pop(-1)

  def _get_suggested_name(self) -> str:
    assert self.module
    reserved_names = set(self.parent.children.keys()) | self._ReservedNames
    if self.parent.module:
      # Check parent name scope module, any attrib from there to self.module.
      # Do a depth-first search through the parents, starting from self.module,
      # until we find self.parent.module.
      # Somewhat consistent to _get_abs_canonical_name.
      queue = [self.module]
      cache = {}  # parent -> full attrib
      while queue:
        module = queue.pop(-1)  # depth-first
        postfix = f".{cache[module]}" if module in cache else ""
        queue_ext = []
        for parent, attr in module.parents_with_attr():
          if parent in cache:
            if cache[parent] in reserved_names:
              cache[parent] = attr + postfix  # anyway overwrite
            continue
          cache[parent] = attr + postfix
          queue_ext.append(parent)
        queue.extend(reversed(queue_ext))
        if self.parent.module in cache:
          break
      if self.parent.module in cache:
        return cache[self.parent.module]
    # Check parent module (or module), and use this attrib name.
    # First check if we can find any attr which is not yet reserved.
    for parent, attr in self.module.parents_with_attr():
      if attr not in reserved_names:
        return attr
    # Now again, to just use any.
    for parent, attr in self.module.parents_with_attr():
      return attr
    # Check potential previous calls, and reuse their name.
    for call in self.module.calls:
      if call is self:
        continue  # ignore this
      if call.parent is self.parent:
        return call.name
    # Fallback to the canonical name.
    return self.module.get_default_name()

  def _get_unique_name(self, suggested_name: Optional[str] = None) -> str:
    name = suggested_name or self._get_suggested_name()
    reserved_names = set(self.parent.children.keys()) | self._ReservedNames
    if self.parent.module:
      # Also reserve all attrib names of the parent module.
      # However, we allow to use the name if it is the attrib itself.
      if self.module and name not in reserved_names and getattr(self.parent.module, name, None) is self.module:
        return name
      if self.layer_ref and name not in reserved_names and getattr(self.parent.module, name, None) is self.layer_ref:
        return name
      reserved_names |= set(vars(self.parent.module).keys())
    if name not in reserved_names:
      return name
    i = 0
    while True:
      name_ = f"{name}_{i}"
      if name_ not in reserved_names:
        return name_
      i += 1


def _move_layer_ref_to_new_name_ctx(*, layer_ref: LayerRef, name_ctx: NameCtx):
  """
  Moves an existing layer ref (with assigned name ctx)
  to another name ctx (without assigned layer or layer ref).

  This assumes that there are no other references to layer_ref.name_ctx
  because those would become invalid.
  References to layer_ref itself should be fine.
  """
  assert not name_ctx.layer and not name_ctx.layer_ref  # none yet assigned

  # Remove layer_ref.name_ctx from its parent name ctx.
  _remove_name_ctx_from_parent(layer_ref.name_ctx)

  # Now reassign.
  layer_ref.name_ctx = name_ctx
  name_ctx.layer_ref = layer_ref
  if isinstance(layer_ref, Layer):
    name_ctx.layer = layer_ref


def _remove_name_ctx_from_parent(name_ctx: NameCtx):
  old_name_ctx = name_ctx.parent.children.pop(name_ctx.name)
  assert old_name_ctx is name_ctx


def _data_from_layer_dict(layer_dict: LayerDictRaw) -> Data:
  from returnn.tf.network import TFNetwork, ExternData, get_layer_class
  from returnn.tf.layers.base import InternalLayer, LayerBase
  from returnn.util import BehaviorVersion
  from returnn.config import Config
  config = Config({
    "behavior_version": _min_returnn_behavior_version,
  })
  BehaviorVersion.set(_min_returnn_behavior_version)
  net = TFNetwork(config=config, extern_data=ExternData(), name="dummy_net")

  ref_to_layer_name = {}  # type: Dict[NameCtx, str]

  def _get_unique_name(name) -> str:
    reserved_names = set(net.layers.keys()) | {"data"}
    if name not in reserved_names:
      return name
    i = 0
    while True:
      name_ = f"{name}_{i}"
      if name_ not in reserved_names:
        return name_
      i += 1

  def _get_layer_name(ref: LayerRef) -> str:
    if ref.name_ctx in ref_to_layer_name:
      return ref_to_layer_name[ref.name_ctx]
    name = _get_unique_name(ref.name_ctx.name)
    ref_to_layer_name[ref.name_ctx] = name
    assert name not in net.layers
    net.layers[name] = InternalLayer(name=name, network=net, output=ref.data)
    return name

  def _get_layer(name: str) -> LayerBase:
    assert name in net.layers
    return net.layers[name]

  def _map_layer_dict_elem(value):
    if isinstance(value, LayerRef):
      return _get_layer_name(value)
    return value

  layer_dict = nest.map_structure(_map_layer_dict_elem, layer_dict)
  out_name = _get_unique_name("output")

  layer_desc = layer_dict.copy()
  layer_class = None
  try:
    layer_class = get_layer_class(layer_desc.pop("class"))
    # Note about name:
    # The name can be to the root network (full name) or to the owning/direct network (`net`) (base_name).
    # The name can optionally have a prefix (here we only care about extra net prefix "extra...:").
    # The prefix is implied by the owning network.
    layer_desc["_network"] = net
    layer_desc["_name"] = out_name

    layer_class.transform_config_dict(layer_desc, network=net, get_layer=_get_layer)

    # noinspection PyProtectedMember
    layer_desc = net._create_layer_layer_desc(name=out_name, layer_desc=layer_desc, template=True)
    out_data = layer_class.get_out_data_from_opts(**layer_desc)

  except Exception as exc:
    raise Exception(
      f"Failed to infer output Data from {layer_class.layer_class if layer_class else None!r} layer"
      f" {layer_desc!r}") from exc

  return out_data
