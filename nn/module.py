"""
Base module class, :class:`Module`.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Sequence, List, Tuple, Union, Set, Iterator, Callable, TypeVar
from returnn.util.basic import NotSpecified, OptionalNotImplementedError
from .. import nn


T = TypeVar("T", bound="Module")


class Module:
  """
  This can represent a subnetwork in RETURNN.

  You can write PyTorch-like code here, like::

      class MyModule(nn.Module):

        def __init__(self, dim: nn.Dim, activation=tanh):
          super().__init__()
          self.layer_norm = nn.LayerNorm()
          self.linear = nn.Linear(dim)
          self.activation = activation

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
          x_ = x
          x = self.layer_norm(x)
          x = self.linear(x)
          x = self.activation(x)
          return x_ + x

  A module (here, just like in PyTorch or Keras)
  has params, but getting some output for some input
  requires an additional `forward` or `__call__` call,
  which can be called multiple times.
  Every such call would then share the same module parameters.

  In contrast, a normal RETURNN layer already has some assigned input and output,
  and usually its own parameters.
  Thus, a module here is different in that aspect,
  that it decouples the module call from the module definition including parameters.

  Every module call creates a RETURNN layer,
  where every call after the first would share the params
  with the first layer.

  A user would create an instance and then call it,
  and get :class:`Layer` instances.

  The RETURNN naming logic of created layers
  is handled via :class:`NameCtx`
  but usually the user does not need to care about that.

  A module developer which wants to derive its own module
  would usually overwrite :func:`__call__` and :func:`__init__`.

  To actually make it a subnetwork in RETURNN,
  any function (e.g. :func:`__call__`) would be decorated with :func:`scoped`.

  The :func:`__init__` would usually get module-level arguments
  which describe the parameters.
  As a module might be called multiple times,
  any input-specific arguments such as spatial dims
  are usually arguments of :func:`__call__`.
  Other arguments which might vary between calls
  would also be arguments of :func:`__call__`
  such as epsilon
  although there are no strict rules.
  """

  def __init__(self):
    """
    By convention, any options to the module or module are passed to the constructor,
    and potential changing inputs (other layers)
    are passed to :func:`__call__`.
    """
    # Actually we would want an ordered set for parents, but Python does not provide this.
    # We abuse a dict as a set. This is ordered since Python 3.6, see #43.
    # Also note that the current code does not clean this up when you do delattr later or so.
    self._parents = _ModuleParents()  # type: Dict[Tuple[Module, str], None]  # (parent,attrib) -> None
    self.calls = _ModuleCalls()  # type: List[nn.NameCtx]

  def __repr__(self):
    return f"<{self.__class__.__name__}>"

  def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
    """
    :return: default initial state, to be used if the module (layer) has recurrent (hidden) state.
      When a module has recurrent state,
      the convention is to return a tuple with instance :class:`LayerState` as the last item,
      and to accept the ``state`` argument with a :class:`LayerState` with the same nested structure.
      This can be a nested structure and should match the structure of the ``state`` argument and returned value.
    """
    state = nn.LayerState()
    for key, mod in self.named_children(recurse=False):
      sub_state = mod.default_initial_state(batch_dims=batch_dims)
      if sub_state:
        state[key] = sub_state
    if state:
      return state
    return None

  def get_default_name(self) -> str:
    """
    Get a default layer name (used when we do not have a Module attribute pointing to this).
    This is used by :class:`NameCtx` for the RETURNN layer naming
    (but only when the RETURNN layer name is not implied by other the module attribute hierarchy).
    """
    name = self.__class__.__name__
    if name.startswith("_"):
      name = name[1:]
    if name[:1].isupper():
      from returnn.util.basic import camel_case_to_snake_case
      name = camel_case_to_snake_case(name)
    return name

  def __call__(self, *args, **kwargs) -> Union[nn.Tensor, Tuple[nn.Tensor, nn.LayerState], Any]:
    """
    Main module call.

    Note that there is nothing really specific about this method.
    Your module can have other methods as well,
    and you don't necessarily need to define this.
    Only certain other functions or modules like Sequential make use of it.
    """
    raise OptionalNotImplementedError

  def __setattr__(self, key: str, value):
    super().__setattr__(key, value)
    sub_calls = []  # type: List[nn.NameCtx]
    if isinstance(value, Module):
      value._parents[(self, key)] = None
      sub_calls = value.calls
    elif isinstance(value, nn.Tensor):
      if (self, key) not in value.parent_modules:
        value.parent_modules.append((self, key))
      sub_calls = [value.name_ctx]
    if sub_calls:
      if not self.calls:
        nn.NameCtx.current_ctx()  # make sure self module gets some NameCtx
      for sub_call in sub_calls:
        for self_call in self.calls:
          if (
                (sub_call.parent is None and self_call.control_flow_ctx() is None) or  # e.g. nn.Parameter
                (self_call.root is sub_call.root and self_call.control_flow_ctx() is sub_call.control_flow_ctx())):
            sub_call.assign_parent(self_call, key)
            break

  def parents_with_attr(self) -> Iterator[Tuple[nn.Module, str]]:
    """
    Get all (immediate) parent modules, and the attrib name which points to us.
    The order is deterministic by insertion order.
    """
    # We rely on deterministic order of dict.
    for parent, attr in self._parents.keys():
      # We currently don't do proper cleanup of _parents via delattr etc,
      # so explicitly check.
      if getattr(parent, attr, None) is self:
        yield parent, attr

  def children(self, *, recurse: bool = True) -> Iterator[nn.Module]:
    """
    Get all (immediate) children modules
    """
    for name, child in self.named_children(recurse=recurse):
      yield child

  def named_children(self,
                     *, recurse: bool = True, memo: Optional[Set[nn.Module]] = None, prefix: str = ''
                     ) -> Iterator[Tuple[str, nn.Module]]:
    """
    Get all children modules (excluding self)
    """
    if memo is None:
      memo = set()
    for name, module in vars(self).items():
      if not isinstance(module, Module):
        continue
      if module in memo:
        continue
      sub_prefix = prefix + ('.' if (prefix and not prefix.endswith(".")) else '') + name
      memo.add(module)
      yield sub_prefix, module
      if recurse:
        for name_, mod_ in module.named_children(recurse=True, memo=memo, prefix=sub_prefix):
          yield name_, mod_

  def named_parameters(self, *, recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
    """
    Get all children parameters, together with their names.

    With recurse=True (default), this iterates over all children modules
    and iterates through their parameters as well.

    Note that some modules (e.g. :class:`nn.Linear`) can behave lazy,
    i.e. they only create the parameters on the first call,
    e.g. when the input dimension is unknown and thus the parameter shape is not defined
    before the first call.
    This means you need to first call the module once to get all the parameters.
    https://github.com/rwth-i6/returnn_common/issues/149
    """
    memo = set()  # over name contexts because we cannot hash layer refs

    def _iter_params(module: Module, prefix: str) -> Iterator[Tuple[str, nn.Parameter]]:
      for key, value in vars(module).items():
        if isinstance(value, nn.Parameter) and value.name_ctx not in memo:
          sub_prefix = prefix + ('.' if prefix else '') + key
          memo.add(value.name_ctx)
          yield sub_prefix, value

    for name, param in _iter_params(module=self, prefix=''):
      yield name, param
    if recurse:
      for child_prefix, child_mod in self.named_children(recurse=True):
        for name, param in _iter_params(module=child_mod, prefix=child_prefix):
          yield name, param

  def parameters(self, *, recurse: bool = True) -> Iterator[nn.Parameter]:
    """
    Get all children parameters. Also see :func:`named_parameters` for some more documentation.
    """
    for name, param in self.named_parameters(recurse=recurse):
      yield param

  @property
  def has_parameters(self):
    """
    Whether this module has variables
    """
    for _, _ in self.named_parameters(recurse=True):
      return True
    return False

  def apply(self: T, fn: Callable[[nn.Module], None]) -> T:
    """
    Applies the function ``fn`` to all children modules and self.

    :return: self
    """
    for child in self.children():
      fn(child)
    fn(self)
    return self


class Functional(Module):
  """
  Used via :func:`nn.scoped` for functions (pure functional, i.e. not methods of another module)
  and via :class:`nn.ModuleList` to wrap up any functions or lambdas as modules.
  """

  def __init__(self, func):
    super().__init__()
    assert callable(func)
    self.func = func

  def __repr__(self):
    return f"{self.__class__.__name__}({self.func.__qualname__})"

  def get_default_name(self) -> str:
    """default name"""
    import re
    name = self.func.__qualname__
    assert isinstance(name, str)
    if name.startswith("Tensor.__"):
      m = re.match(r"^Tensor\.__(.*)__$", name)
      if m:
        return m.group(1)
    if ".<locals>." in name:
      name = name.replace(".<locals>.", ".")
    return name

  def __call__(self, *args, **kwargs):
    return self.func(*args, **kwargs)


# noinspection PyAbstractClass
class ReturnnWrappedLayerBase(Module):
  """
  Base class for all automatically wrapped layers.
  """
  returnn_layer_class: Optional[str] = None
  has_recurrent_state: bool = False
  has_variables: bool = False

  @staticmethod
  def returnn_layer_get_recurrent_state(layer: nn.Tensor) -> nn.LayerState:
    """
    :returns: the recurrent state

    You might override this in case the state is more complex,
    and return some named tuple or any other hierarchical structure.
    """
    from ._generated_layers import _get_last_hidden_state
    # Note that this is actually layer specific.
    # We try to use a number of heuristics to get it right for the common cases.
    name = f"{layer.name_ctx.name}_state"
    layer_class = layer.layer_dict["class"]
    if layer_class in {"cum_concat", "cumsum"}:
      return nn.LayerState(layer)  # the layer output itself is its state
    if layer_class == "window":
      return nn.LayerState(_get_last_hidden_state(layer, out_dim=layer.feature_dim, name=name))
    # This is some very generic fallback code, which probably does not work correctly in some cases.
    out_dim = layer.layer_dict["out_dim"]
    if layer_class == "rec" and isinstance(layer.layer_dict["unit"], str):
      if "lstm" in layer.layer_dict["unit"].lower():
        h = _get_last_hidden_state(layer, out_dim=out_dim, key="h", name=f"{name}_h")
        c = _get_last_hidden_state(layer, out_dim=out_dim, key="c", name=f"{name}_c")
        return nn.LayerState(h=h, c=c)
    return nn.LayerState(_get_last_hidden_state(layer, out_dim=out_dim, name=name))

  def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> nn.LayerState:
    """
    :return: default initial state
    """
    from .const import zeros
    assert self.has_recurrent_state
    # Match the logic of _get_recurrent_state above.
    if self.returnn_layer_class == "rec":
      unit = getattr(self, "unit")
      if isinstance(unit, str):
        if "lstm" in unit.lower():
          out_dim = getattr(self, "out_dim")
          return nn.LayerState(h=zeros(list(batch_dims) + [out_dim]), c=zeros(list(batch_dims) + [out_dim]))
      raise NotImplementedError(f"{self}.default_initial_state for RecLayer with unit {unit!r}")
    raise NotImplementedError(f"{self}.default_initial_state")

  @staticmethod
  def handle_recurrent_state(args: Dict[str, Any], *,
                             axis: nn.Dim,
                             state: Optional[Union[nn.Tensor, Dict[str, nn.Tensor], NotSpecified]] = NotSpecified,
                             ):
    """
    Update the args to include either state or initial_state,
    depending on whether we operate per step or on an axis.

    :param args: layer arguments
    :param axis: single_step_dim specifies to operate for a single step
    :param state: prev state when operating a single step or initial state when operating on an axis
    """
    if axis == nn.single_step_dim:
      args['state'] = state
    else:
      if state is not NotSpecified:
        args['initial_state'] = state


class _CollectionEmptyDeepCopyMixin:
  def __deepcopy__(self, memo):
    return type(self)()


class _ModuleParents(dict, _CollectionEmptyDeepCopyMixin):
  """list of parents should not be copied when copying a module"""


class _ModuleCalls(list, _CollectionEmptyDeepCopyMixin):
  """list of calls should not be copied when copying a module"""
