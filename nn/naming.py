"""
All the naming logic,
which is responsible for mapping :class:`Module` hierarchy
and the call hierarchy (via :class:`Layer` and :class:`Tensor`)
and the parameters of the model
to a RETURNN net dict.

The main class is :class:`NameCtx`.

Note on the different type of name hierarchies in RETURNN and here in RETURNN-common:

- RETURNN layer name hierarchy.
  The root scope corresponds to the root :class:`TFNetwork`.
  "base:" is used to go one level up in the hierarchy relative to the current scope,
  similar as ".." for OS directories.
  Sub layers can exist depending on the type of parent layer.
  Most naturally this is the case for the :class:`SubnetworkLayer` but also other layers can have sub layers
  such as :class:`RecLayer` or more custom like :class:`SplitLayer`.
  Sub layers can usually always access all parent or base layers but not necessarily the other way around.
  E.g. sub layers of :class:`CondLayer` can not be accessed from outside (currently).
  Some layer names correspond to special layers which are not defined by the user, such as "data:...".

- TensorFlow name scope hierarchy (and variable scope hierarchy, which is mostly the same).
  RETURNN mostly keeps the TF name scope consistent with the RETURNN layer name hierarchy, with some exceptions:
  - Layer names with special symbols (including "/") or which are not valid TF name scope names
    are escaped.
  - The ``name_scope`` option in each layer can overwrite the TF name scope.
    It can either overwrite another absolute TF name scope or a relative TF name scope.

- :class:`NameCtx` hierarchy, which mostly corresponds to the RETURNN layer name hierarchy.
  It also covers special layers (such as "data:...") and virtual subnetworks
  (which do not consume a new level, such as the true/false branching subnetworks of :class:`CondLayer`).

- The :class:`Module` hierarchy.
  Submodules are via attributes of the parent module and the attribute names define the hierarchy.
  :class:`Parameter`s must be attributes of modules and their TF name scope is defined
  by the module hierarchy such that different ways to access the parameter will not lead to different TF name scopes
  (see https://github.com/rwth-i6/returnn_common/issues/25).
  See :func:`Module.layer_abs_name_scope`.
  This is applied to all modules which have parameters, via :func:`make_layer`,
  and to :class:`Parameter`s itself, via :func:`Tensor._assign_parent`.
  This is applied by setting ``name_scope`` in the corresponding layer.

"""

from __future__ import annotations
from typing import Optional, Union, Any, Sequence, List, Tuple, Set, Dict
from tensorflow.python.util import nest
from returnn.util.basic import NotSpecified
from .. import nn


def scoped(func):
  """
  Decorator to create a new scope (subnetwork) for the function
  or module method.

  This is usually used for the ``__call__`` method of a module
  or for pure functions.
  """
  assert callable(func)

  def _wrapper(*args, name: Optional[Union[str, nn.NameCtx]] = None, **kwargs):
    if name == "":
      return func(*args, **kwargs)
    if args and isinstance(args[0], nn.Module):
      self = args[0]
    else:
      self = nn.Functional(func)
    from . import copy
    with nn.NameCtx.get_from_call(module=self, name=name) as name_ctx:
      name_ctx.is_subnet_ctx = True
      res = func(*args, **kwargs)
      if name_ctx.parent is None:  # root
        # special logic, no output layers, no subnetwork layer needed
        self.calls.append(name_ctx)
        return res
      if isinstance(res, nn.Tensor):
        out = copy(res, name=name_ctx.get_child("output"))
      else:
        # we return more than one layer (thus also working on other layers of the subnet, that are not output)
        # by convention: first layer is the output layer
        res_flat = nest.flatten(res)
        out = copy(res_flat[0], name=name_ctx.get_child("output"))
      assert out.data
      # Now create the subnetwork layer itself.
      subnet_layer = nn.make_layer(
        {"class": "subnetwork", "from": [], "subnetwork": name_ctx.make_net()},
        name=name_ctx, predefined_out_data=out.data)
    if isinstance(res, nn.Tensor):
      return subnet_layer  # maybe nicer to return subnet layer
    return res

  _wrapper.__name__ = func.__name__
  _wrapper.__qualname__ = func.__qualname__
  return _wrapper


class NameCtx:
  """
  This is a helper class to keep track of the current name context when creating layers.
  Usually you do not need to access this directly
  except for creating the root name ctx
  and getting out the final RETURNN config or net dict.

  A name ctx represents one absolute layer name in the RETURNN layer hierarchy,
  except for the root name ctx.

  A name ctx thus can have a parent name ctx (if it is not the root),
  and potentially child name contexts.

  See the documentation on name hierarchies for RETURNN and RETURNN-common in the module docstring at the top.
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
               module: Optional[nn.Module] = None,
               suggested_name: Optional[str] = None,
               name: Optional[str] = None,
               virtual: bool = False,
               can_access_children: bool = True,
               parent: Optional[NameCtx] = NotSpecified):
    self.module = module
    self.layer_ref = None  # type: Optional[nn.Tensor]
    self.layer = None  # type: Optional[nn.Tensor]
    self._layer_abs_name_scope = None  # type: Optional[str]
    self.is_subnet_ctx = False
    self.virtual = virtual  # does not consume a layer name in RETURNN. see get_name_in_ctx
    self.can_access_children = can_access_children  # from outside
    self.children = {}  # type: Dict[str, NameCtx]
    self.extern_data = {}  # type: Dict[str, nn.Data]  # only for the root name ctx
    self.global_batch = None  # type: Optional[nn.BatchInfo]
    self.marked_outputs = []  # type: List[nn.Tensor]
    self.marked_losses = []  # type: List[nn.Tensor]
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
    self.custom_layer_name_scope = None  # type: Optional[str]
    self._assign_layer_name_scope()

  @classmethod
  def get_from_call(cls, *, name: Optional[Union[str, NameCtx]], module: nn.Module) -> NameCtx:
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
    parts = [self.get_abs_name_repr()]
    if self.layer_ref:
      parts.append("[%s]" % ",".join(self.layer_ref.data.get_batch_axes_short_description()))
    if self.module:
      parts.append(f"module:{self.module}")
    return f"<{self.__class__.__name__} {' '.join(parts)}>"

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
    self._assign_layer_name_scope()

  def move_layer_ref_here(self: NameCtx, layer_ref: nn.Tensor):
    """
    Moves an existing layer ref (with assigned name ctx)
    to another name ctx (without assigned layer or layer ref).

    This assumes that there are no other references to layer_ref.name_ctx
    because those would become invalid.
    References to layer_ref itself should be fine.
    """
    assert not self.layer and not self.layer_ref  # none yet assigned

    # Remove layer_ref.name_ctx from its parent name ctx.
    if layer_ref.name_ctx.parent:
      old_name_ctx = layer_ref.name_ctx.parent.children.pop(layer_ref.name_ctx.name)
      assert old_name_ctx is layer_ref.name_ctx
    old_name_ctx = layer_ref.name_ctx

    # Now reassign.
    layer_ref.name_ctx = self
    self.layer_ref = layer_ref
    if layer_ref.layer_dict:
      self.layer = layer_ref
    self.module = old_name_ctx.module
    if self.module:
      for i, call in enumerate(self.module.calls):
        if call is old_name_ctx:
          self.module.calls[i] = self
    self.children = old_name_ctx.children
    for child in self.children.values():
      child.parent = self

    if layer_ref.layer_dict:
      def _check_layer_opt_value(v):
        if isinstance(v, nn.Net):
          assert v.name_ctx is old_name_ctx
          v.name_ctx = self
      nest.map_structure(_check_layer_opt_value, layer_ref.layer_dict)

    # Make sure the name scope is correct.
    self.custom_layer_name_scope = None
    self._layer_abs_name_scope = None
    self._assign_layer_name_scope()

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

  @property
  def can_access_children_from_root(self):
    """
    :return: whether can_access_children for self and all parents
    """
    name = self
    while name:
      if not name.can_access_children:
        return False
      name = name.parent
    return True

  def extend_reserved_names(self, names: Set[str]):
    """
    Extend reserved child names.
    """
    # Do not update inplace because we want an own instance on self.
    self._ReservedNames = self._ReservedNames | names

  def _assign_layer_name_scope(self):
    # We must check whether the RETURNN abs layer name is consistent with our module naming hierarchy,
    # and make it consistent if not (https://github.com/rwth-i6/returnn_common/issues/25).
    # The parent name ctx RETURNN layer will also have the right name_scope set,
    # so this layers name scope default is simply based on that.
    # Note that parameters could be assigned lazily at some later point.
    if not self.parent or not self.module:
      return
    if self._layer_abs_name_scope is None:
      if self._get_abs_canonical_name(try_run=True) is None:  # not yet well defined, so try later
        return
    assert self.custom_layer_name_scope is None
    layer_abs_name_scope_parent = self.parent.layer_abs_name_scope_effective
    if layer_abs_name_scope_parent:
      layer_abs_name_scope_parent += "/"
    layer_abs_name_scope_default = layer_abs_name_scope_parent + self.name
    if layer_abs_name_scope_default != self.layer_abs_name_scope:  # default does not match what we require
      if self.layer_abs_name_scope == self.parent.layer_abs_name_scope_effective:
        self.custom_layer_name_scope = ""
      elif self.layer_abs_name_scope.startswith(layer_abs_name_scope_parent):  # can use relative
        self.custom_layer_name_scope = self.layer_abs_name_scope[len(layer_abs_name_scope_parent):]
      else:  # must use absolute
        self.custom_layer_name_scope = "/" + self.layer_abs_name_scope
    if self.layer:
      assert self.layer.layer_dict.get("name_scope", None) is None
      if self.custom_layer_name_scope is not None:
        self.layer.layer_dict["name_scope"] = self.custom_layer_name_scope

  def _remove_unused(self):
    # Collect all used tensor names.
    used_names = {self}  # type: Set[nn.NameCtx]
    queue = list(self.marked_outputs + self.marked_losses)  # type: List[nn.Tensor]
    while queue:
      tensor = queue.pop(0)
      if tensor.name_ctx is used_names:
        continue
      used_names.add(tensor.name_ctx)
      for dep in tensor.get_dependencies():
        if dep.name_ctx not in used_names:
          queue.append(dep)

    # Go through all names in the hierarchy and remove unused.
    visited = set()  # type: Set[nn.NameCtx]
    queue = [self]  # type: List[nn.NameCtx]
    while queue:
      name_ctx = queue.pop(0)
      if name_ctx in visited:
        continue
      visited.add(name_ctx)
      if name_ctx not in used_names:
        assert name_ctx.parent
        name_ctx.parent.children.pop(name_ctx.name)
        if name_ctx.layer_ref:
          for hook in name_ctx.layer_ref.remove_unused_cleanup_hooks:
            hook(name_ctx.layer_ref)
      else:
        for child in name_ctx.children.values():
          queue.append(child)

  def get_returnn_config(self) -> Dict[str, Any]:
    """
    :return: config dict updates for returnn
    """
    self._remove_unused()
    assert not self.parent, f"{self} get_returnn_config only makes sense in the root name ctx"
    net_dict = self.make_net().make_net_dict_raw()
    return {
      "behavior_version": nn.min_returnn_behavior_version,
      "extern_data": {
        data_key: {
          key: getattr(data, key)
          for key in [*data.get_kwargs(include_special_axes=False).keys(), "available_for_inference"]
          if key not in {"name", "batch"}}
        for (data_key, data) in self.extern_data.items()},
      "network": net_dict,
    }

  def get_returnn_config_serialized(self) -> str:
    """
    :return: serialized config, i.e. Python code
    """
    from ..utils.pprint import pformat
    config = self.get_returnn_config()
    dim_tags_proxy = ReturnnDimTagsProxy()
    config = dim_tags_proxy.collect_dim_tags_and_transform_config(config)

    code_lines = [
      "from returnn.tf.util.data import Dim, batch_dim, single_step_dim, SpatialDim, FeatureDim\n\n",
      "use_tensorflow = True\n",
      f"behavior_version = {config.pop('behavior_version')}\n\n",
      f"{dim_tags_proxy.py_code_str()}\n",
      f"extern_data = {pformat(config.pop('extern_data'))}\n",
      f"network = {pformat(config.pop('network'))}\n",
    ]
    if config:
      for key, value in config.items():
        code_lines.append(f"{key} = {pformat(value)}\n")
      code_lines.append("\n")
    return "".join(code_lines)

  def make_net(self) -> Net:
    """
    Create new (sub) net, an instance of :class:`Net`.
    """
    return Net(name_ctx=self)

  def make_default_output(self, ref: nn.Tensor) -> nn.Tensor:
    """
    Assume this is a subnet, and make a default output.
    """
    from . import copy
    assert self.is_subnet_ctx
    if ref.name_ctx is self.children.get("output", None):  # if this is the output layer already, allow and just return
      return ref
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
      debug_name = "/".join(
        (repr(ctx.name) if not ctx.virtual else f"({ctx.name!r})")
        if i > 0 or ctx.name is not None else ''
        for i, ctx in enumerate(ls))
    return debug_name

  @property
  def layer_abs_name_scope_effective(self) -> str:
    """
    :return: abs TF name scope which will be used by RETURNN for this layer
    """
    if not self.parent:  # root
      return ""
    prefix = self.parent.layer_abs_name_scope_effective
    if prefix:
      prefix += "/"
    if self.layer:
      assert self.layer.layer_dict.get("name_scope", None) == self.custom_layer_name_scope
    if self.custom_layer_name_scope is not None:
      name_scope = self.custom_layer_name_scope
      assert isinstance(name_scope, str)
      if name_scope == "":
        return prefix[:-1]
      elif name_scope.startswith("/"):
        return name_scope[1:]
      else:
        return prefix + name_scope
    if self.virtual:
      return prefix[:-1]
    return prefix + self.name

  @property
  def layer_abs_name_scope(self) -> str:
    """
    :return: expected layer abs name scope, i.e. the TF name scope of variables.
      This uses the module hierarchy (:func:`_get_abs_canonical_name`).

    Also see :func:`layer_abs_name_scope_effective`.
    """
    if self._layer_abs_name_scope is not None:
      return self._layer_abs_name_scope
    assert self.module
    self._layer_abs_name_scope = self._get_abs_canonical_name()
    return self._layer_abs_name_scope

  def _get_abs_canonical_name(self, *, join_str="/", try_run: bool = False) -> Optional[str]:
    """
    :param str join_str: maybe "." is more common for attrib chains.
      however, we use "/" as default, to make this consistent to :func:`get_abs_name`.
    :return: unique absolute layer name for the module hierarchy.
      https://github.com/rwth-i6/returnn_common/issues/25
    """
    assert self.module, f"{self} is not assigned to a module"
    root = self.root
    root_module = root.module  # might be None
    assert root_module, f"root name ctx {self.root} is not assigned to a module"
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
        assert isinstance(parent, nn.Module)
        if parent in cache:
          continue
        for call in parent.calls:
          assert isinstance(call, nn.NameCtx)
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
      if try_run:
        return None
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
    prefix = "".join(["base:" for ctx in reversed(ctx_scope_abs[common_len:]) if not ctx.virtual])
    postfix = "/".join([ctx.name for ctx in self_name_abs[common_len:] if not ctx.virtual])
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

  def get_child_with_layer_ref(self, name: str, *, data: nn.Data) -> NameCtx:
    """
    Makes sure the child exists, including a corresponding layer ref.
    Creates the child together with a layer ref if it does not exist yet.
    """
    child = self.get_child(name)
    if not child.layer_ref:
      layer_ref = nn.Tensor(name_ctx=child, data=data, is_ref=True)
      assert child.layer_ref is layer_ref
    return child

  def get_child_layer_ref(self, name: str, *, data: nn.Data) -> nn.Tensor:
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
    # Check parent module, and use this attrib name.
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


class Net:
  """
  Represents a RETURNN (sub) network.
  It can create a net dict when needed.
  """
  def __init__(self, *, name_ctx: NameCtx):
    self.name_ctx = name_ctx

  def __repr__(self):
    return f"Net{self.name_ctx!r}"

  def _map_elem_resolve(self, obj: Any) -> Any:
    if isinstance(obj, nn.Tensor):
      return obj.get_name_in_ctx(ctx=self.name_ctx)
    if isinstance(obj, Net):
      return obj.make_net_dict_raw()
    return obj

  def make_net_dict_raw(self) -> nn.NetDictRaw:
    """
    Create raw net dict, not containing any :class:`Tensor` or :class:`Net` instances anymore.
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


class ReturnnDimTagsProxy:
  """
  When serialized via __repr__, this represents a dict unique_name -> dim tag.
  All usages in the network and extern_data will also get proxies when serialized point to this dict.
  """

  def __init__(self):
    self.dim_refs_by_name = {}  # type: Dict[str, ReturnnDimTagsProxy.DimRefProxy]
    self.dim_refs_by_tag = {}  # type: Dict[nn.Dim, ReturnnDimTagsProxy.DimRefProxy]

  def __repr__(self):
    return "\n".join([
      "{",
      *(f"  {key!r}: {value.dim_repr()}," for key, value in self.dim_refs_by_name.items()),
      "}"])

  def copy(self) -> ReturnnDimTagsProxy:
    """
    :return: creates a shallow copy
    """
    new = ReturnnDimTagsProxy()
    new.dim_refs_by_name = self.dim_refs_by_name.copy()
    new.dim_refs_by_tag = self.dim_refs_by_tag.copy()
    return new

  def py_code_str(self):
    """
    :return: Python code
    """
    return "".join([
      *(f"{value.py_id_name()} = {value.dim_repr()}\n" for key, value in self.dim_refs_by_name.items()),
      ])

  def dim_ref_repr(self, dim: nn.Dim) -> str:
    """
    :return: for the given dim, Python code which refers to it, via ``dim_tags``
    """
    if dim == nn.batch_dim:
      return "batch_dim"
    if dim == nn.single_step_dim:
      return "single_step_dim"
    if dim.match_priority:
      return f"{self.dim_ref_repr(dim.copy(match_priority=0))}.copy(match_priority={dim.match_priority})"
    if dim.derived_from_op:
      if dim.derived_from_op.kind == "constant":
        return str(dim.derived_from_op.attribs["value"])
      if dim.derived_from_op.kind == "truediv_left":
        assert len(dim.derived_from_op.inputs) == 2
        a, b = dim.derived_from_op.inputs
        return f"{self.dim_ref_repr(a)}.div_left({self.dim_ref_repr(b)})"
      op_str = {"add": "+", "mul": "*", "truediv_right": "//"}[dim.derived_from_op.kind]
      return f" {op_str} ".join(self.dim_ref_repr(in_) for in_ in dim.derived_from_op.inputs)
    ref = self.dim_refs_by_tag[dim]
    return ref.py_id_name()

  class DimRefProxy:
    """
    This will be a reference to the global dim_tags __repr__.
    """
    def __init__(self, *, dim: nn.Dim, name: Optional[str], path: Tuple[Any, ...], parent: ReturnnDimTagsProxy):
      self.dim = dim
      self.name = name  # None, or valid Python identifier
      self.path = path
      self.parent = parent
      self.debug_idx = len(parent.dim_refs_by_name)

    def __repr__(self):
      return self.ref_repr()

    def ref_repr(self) -> str:
      """ref repr"""
      return self.parent.dim_ref_repr(self.dim)

    def py_id_name(self) -> str:
      """
      :return: valid Python identifier
      """
      assert self.name
      return self.name + "_dim"

    def dim_repr(self):
      """
      Dim repr, used for serialization of all registered dim tags.
      Any derived dims or special dims will not be registered and instead be represented
      with the same derivation referencing other registered dim tags.
      See :func:`ReturnnDimTagsProxy.dim_ref_repr`.
      """
      dim = self.dim
      assert not dim.is_batch_dim()
      assert dim.can_be_used_as_dim()
      assert not dim.derived_from_op
      assert not dim.match_priority
      # We assume FeatureDim, SpatialDim and Dim are imported.
      if dim.kind == nn.Dim.Types.Feature:
        return f"FeatureDim({dim.description!r}, {dim.dimension})"
      if dim.kind == nn.Dim.Types.Spatial:
        if dim.dimension is not None:
          return f"SpatialDim({dim.description!r}, {dim.dimension})"
        else:
          return f"SpatialDim({dim.description!r})"
      # generic fallback
      return f"Dim(kind={dim.kind}, description={dim.description!r}, dimension={dim.dimension})"

  class SetProxy:
    """
    This represents a set but with a predefined order.
    We want a deterministic order in the repr such that the generated code stays deterministic.
    """
    def __init__(self, values: Sequence[Any]):
      self.values = values

    def __repr__(self):
      return f"{{{', '.join(map(repr, self.values))}}}"

  def collect_dim_tags_and_transform_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Go through the config and collect all dim tags, replace them by proxies.

    :return: new config
    """
    import re

    def _sort_key(value):
      if isinstance(value, ReturnnDimTagsProxy.DimRefProxy):
        if value.dim.kind == nn.Dim.Types.Batch:
          return -1
        return value.debug_idx
      return value

    def _unique_name(dim: nn.Dim) -> str:
      assert dim not in self.dim_refs_by_tag
      name_ = dim.description
      name_ = re.sub(r"[^a-zA-Z0-9_]", "_", name_)
      if name_.endswith("_dim"):
        name_ = name_[:-len("_dim")]
      if name_ not in self.dim_refs_by_name:
        return name_
      i = 0
      while True:
        name__ = f"{name_}_{i}"
        if name__ not in self.dim_refs_by_name:
          return name__
        i += 1

    # Cannot use nest because nest does not support sets. Also nest requires them to be sorted.
    def _map(path, value):
      if isinstance(value, nn.Dim):
        if value in {nn.batch_dim, nn.single_step_dim}:
          # No need to register this.
          return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
        if value.derived_from_op:
          # Make sure all the inputs are registered.
          for i, child in enumerate(value.derived_from_op.inputs):
            _map(path + (value.derived_from_op.kind, i), child)
          # No need to register this.
          return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
        if value.match_priority != 0:
          _map(path, value.copy(match_priority=0))  # Register the dim tag without match_priority.
          # Now return the custom proxy for the dim tag with match_priority. No need to register this.
          return ReturnnDimTagsProxy.DimRefProxy(dim=value, name=None, path=path, parent=self)
        if value in self.dim_refs_by_tag:
          return self.dim_refs_by_tag[value]
        name = _unique_name(value)
        assert name not in self.dim_refs_by_name
        ref = ReturnnDimTagsProxy.DimRefProxy(dim=value, name=name, path=path, parent=self)
        self.dim_refs_by_name[name] = ref
        self.dim_refs_by_tag[value] = ref
        return ref
      if isinstance(value, dict):
        return {
          _map(path + (key, "key"), key): _map(path + (key, "value"), value_)
          for key, value_ in value.items()}
      if isinstance(value, list):
        return [_map(path + (i,), value_) for i, value_ in enumerate(value)]
      if isinstance(value, tuple) and type(value) is tuple:
        return tuple(_map(path + (i,), value_) for i, value_ in enumerate(value))
      if isinstance(value, tuple) and type(value) is not tuple:
        # noinspection PyProtectedMember,PyUnresolvedReferences,PyArgumentList
        return type(value)(*(_map(path + (key,), getattr(value, key)) for key in value._fields))
      if isinstance(value, set):
        values = [_map(path + (value,), value_) for value_ in value]
        values.sort(key=_sort_key)  # this should be possible now because it would be some sortable proxies
        return ReturnnDimTagsProxy.SetProxy(values)
      return value

    config = _map((), config)
    return config
