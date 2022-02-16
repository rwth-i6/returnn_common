"""
Base module class, :class:`Module`.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple, Union, Set, Iterator
from returnn.util.basic import NotSpecified
from .. import nn


class Module:
  """
  This can represent a subnetwork in RETURNN.

  You can write PyTorch-like code here, like::

      class MyModule(nn.Module):

        def __init__(self, dim: nn.Dim, activation=tanh):
          super().__init__()
          self.linear = nn.Linear(dim)
          self.activation = activation

        @nn.scoped
        def __call__(self, x: nn.Tensor) -> nn.Tensor:
          x_ = x
          x = layer_norm(x)
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
    self._parents = {}  # type: Dict[Tuple[Module, str], None]  # (parent,attrib) -> None
    self.calls = []  # type: List[nn.NameCtx]

  def __repr__(self):
    return f"<{self.__class__.__name__}>"

  def default_initial_state(self) -> Optional[nn.LayerState]:
    """
    :return: default initial state, to be used if the module (layer) has recurrent (hidden) state.
      When a module has recurrent state,
      the convention is to return a tuple with instance :class:`LayerState` as the last item,
      and to accept the ``state`` argument with a :class:`LayerState` with the same nested structure.
      This can be a nested structure and should match the structure of the ``state`` argument and returned value.
    """
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

  @nn.scoped
  def __call__(self, *args, **kwargs) -> Union[nn.Tensor, Tuple[nn.Tensor, nn.LayerState], Any]:
    raise NotImplementedError

  def __setattr__(self, key: str, value):
    super().__setattr__(key, value)
    if isinstance(value, Module):
      value._parents[(self, key)] = None
    if isinstance(value, nn.Tensor):
      if (self, key) not in value.parent_modules:
        value.parent_modules.append((self, key))

  def parents_with_attr(self) -> Iterator[Tuple[nn.Module, str]]:
    """
    Get all (immediate) parent modules, and the attrib name which points to us
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

  def named_parameters(self, *, recurse: bool = True) -> Iterator[Tuple[str, nn.Parameter]]:
    """
    Get all children parameters
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

  @property
  def has_parameters(self):
    """
    Whether this module has variables
    """
    for _, _ in self.named_parameters(recurse=True):
      return True
    return False


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
    return f"{self.__class__.__name__}({self.func.__name__})"

  def get_default_name(self) -> str:
    """default name"""
    return self.func.__qualname__

  @nn.scoped
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
    if layer_class == "cum_concat":
      return nn.LayerState(layer)  # the layer output itself is its state
    out_dim = layer.layer_dict["out_dim"]
    if layer_class == "rec" and isinstance(layer.layer_dict["unit"], str):
      if "lstm" in layer.layer_dict["unit"].lower():
        h = _get_last_hidden_state(layer, out_dim=out_dim, key="h", name=f"{name}_h")
        c = _get_last_hidden_state(layer, out_dim=out_dim, key="c", name=f"{name}_c")
        return nn.LayerState(h=h, c=c)
    return nn.LayerState(_get_last_hidden_state(layer, out_dim=out_dim, name=name))

  def default_initial_state(self) -> nn.LayerState:
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
          return nn.LayerState(h=zeros([nn.batch_dim, out_dim]), c=zeros([nn.batch_dim, out_dim]))
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
