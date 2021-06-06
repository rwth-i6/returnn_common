"""
Base interfaces.

The core interfaces for the user are:

* :class:`ILayerMaker`, to directly create a layer dict
* :class:`Module`, to write PyTorch-style code

Instances of both objects can be called directly,
and return instances of type :class:`LayerRef`,
which can be thought of as analogue to :class:`torch.Tensor` or :class:`tf.Tensor`.

Use ``x.mark_as_loss()`` to mark some output (layer ref) as a loss.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from returnn.util.basic import NotSpecified
from tensorflow.python.util import nest


LayerDictRaw = Dict[str, Any]
LayerRefRaw = str
NetDictRaw = Dict[str, LayerDictRaw]


class LayerRef:
  """
  Refers to a layer.

  TODO:
    extend this by functions __add__, __sub__, etc.
  """

  def __init__(self, *, name_ctx: _NameCtx):
    self.name_ctx = name_ctx
    assert name_ctx.layer_ref is None
    name_ctx.layer_ref = self

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

  def mark_as_loss(self):
    """
    Mark this as a loss.
    """
    raise TypeError("mark_as_loss can only be called on a layer, not a layer-ref.")


class Layer(LayerRef):
  """
  Represents a layer and its output, created by :class:`ILayerMaker`.
  """

  def __init__(self, maker: ILayerMaker, layer_dict: LayerDictRaw):
    super(Layer, self).__init__(name_ctx=_NameCtx.top())
    assert self.name_ctx.maker is maker
    assert self.name_ctx.layer is None
    self.name_ctx.layer = self
    self.maker = maker
    self.layer_dict = layer_dict

  def mark_as_loss(self):
    """
    Mark this as a loss.
    """
    assert "loss" not in self.layer_dict
    self.layer_dict["loss"] = "as_is"

  def _sis_hash(self):
    from sisyphus.hash import sis_hash_helper  # noqa
    return sis_hash_helper(self.layer_dict)


class ILayerMaker:
  """
  Makes a layer.
  """
  def __init__(self):
    self.calls = []  # type: List[Layer]

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
    return self.__class__.__name__

  def __call__(self, *args, name: Optional[str] = None, **kwargs) -> Layer:
    with _NameCtx(maker=self, name=name):
      layer_dict = self.make_layer_dict(*args, **kwargs)
      if self.calls:
        layer_dict = layer_dict.copy()
        assert "reuse_params" not in layer_dict
        layer_dict["reuse_params"] = self.calls[0]
      layer_dict = nest.map_structure(
        lambda x: x.get_name() if isinstance(x, LayerRef) else x,
        layer_dict)
      layer = Layer(self, layer_dict)
      self.calls.append(layer)
      return layer


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

  def make_layer_dict(self, *args, **kwargs) -> LayerDictRaw:
    """
    Make subnet layer dict.
    """
    from .layers import Copy
    name_ctx = _NameCtx.top()
    assert name_ctx.maker is self
    name_ctx.is_subnet_ctx = True
    res = self.forward(*args, **kwargs)
    Copy()(res, name="output")
    return {"class": "subnetwork", "from": [], "subnetwork": name_ctx.make_net_dict()}

  def make_root_net_dict(self) -> NetDictRaw:
    """
    Make net dict, to be used as the main RETURNN network, not within a subnetwork.
    Extern data can be accessed via :func:`get_root_extern_data`.
    """
    with _NameCtx(maker=self, parent=None) as name_ctx:
      name_ctx.is_subnet_ctx = True
      self.forward()
      return name_ctx.make_net_dict()


class Rec(ILayerMaker):
  """
  This represents a RecLayer subnetwork in RETURNN.
  """

  def step(self) -> LayerRef:
    """
    Constructs the output for one step.
    You can write PyTorch-style code here.
    """
    raise NotImplementedError

  def make_layer_dict(self) -> LayerDictRaw:
    """
    Make subnet layer dict.
    """
    from .layers import Copy
    res = self.step()
    Copy()(res, name="output")
    name_ctx = _NameCtx.top()
    assert name_ctx.maker is self
    return {"class": "rec", "from": [], "unit": name_ctx.make_net_dict()}


def get_root_extern_data(data_key: str) -> LayerRef:
  """
  Get extern data from root.
  """
  scope = _NameCtx.top()  # must exist
  scope_abs = scope.get_abs_name_ctx_list()
  root_scope = scope_abs[0]
  root_layer_name = f"data:{data_key}"
  if root_layer_name in root_scope.childs:
    name_ = root_scope.childs[root_layer_name]
    assert name_.layer_ref
  else:
    name_ = _NameCtx(name=root_layer_name, parent=root_scope)
    LayerRef(name_ctx=name_)
    assert name_.layer_ref
  return name_.layer_ref


def get_extern_data(data_key: str) -> LayerRef:
  """
  Get extern data from current scope.
  """
  return get_special_layer(f"data:{data_key}")


def get_special_layer(name: str) -> LayerRef:
  """
  Special layer can be "data:..." or whatever.
  """
  scope = _NameCtx.current_ctx()  # must exist
  if name in scope.childs:
    name_ = scope.childs[name]
    assert name_.layer_ref
  else:
    name_ = _NameCtx(name=name, parent=scope)
    LayerRef(name_ctx=name_)
    assert name_.layer_ref
  return name_.layer_ref


class _NameCtx:
  """
  This is a helper class to keep track of the current name context when creating layers.
  Usually you do not need to access this directly.
  """

  stack = []  # type: List[_NameCtx]
  _ReservedNames = {"data", "output"}

  @classmethod
  def top(cls) -> _NameCtx:
    """
    Return the top of the stack.
    Assumes that it exists.
    """
    assert cls.stack
    return cls.stack[-1]

  @classmethod
  def current_ctx(cls) -> _NameCtx:
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

  def __init__(self, *,
               maker: Optional[ILayerMaker] = None,
               name: Optional[str] = None,
               parent: Optional[_NameCtx] = NotSpecified):
    self.maker = maker
    self.layer_ref = None  # type: Optional[LayerRef]
    self.layer = None  # type: Optional[Layer]
    self.is_subnet_ctx = False
    self.childs = {}  # type: Dict[str, _NameCtx]
    self.parent = parent if parent is not NotSpecified else (self.current_ctx() if self.stack else None)
    self.name = name if name else (self._get_name() if self.parent else None)
    if self.parent:
      assert self.parent.is_subnet_ctx
      assert self.name not in self.parent.childs
      self.parent.childs[self.name] = self

  def make_net_dict(self) -> NetDictRaw:
    """
    Create net dict.
    """
    net_dict = {}
    for key, value in self.childs.items():
      if value.layer:
        net_dict[key] = value.layer.layer_dict
    return net_dict

  def get_abs_name_ctx_list(self) -> List[_NameCtx]:
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
    cur_scope = _NameCtx.current_ctx()
    if self.parent is cur_scope:  # fast path
      return self.name
    cur_scope_abs = cur_scope.get_abs_name_ctx_list()
    self_name_abs = self.get_abs_name_ctx_list()
    assert cur_scope_abs[0] is self_name_abs[0]  # same root
    assert len(cur_scope_abs) > len(self_name_abs)  # not implemented otherwise
    common_len = 0
    while cur_scope_abs[common_len] is self_name_abs[common_len]:
      common_len += 1
    assert common_len == len(self_name_abs) - 2  # not implemented otherwise
    return "base:" * (len(cur_scope_abs) - len(self_name_abs) + 1) + self.name

  def __enter__(self):
    if self.parent:
      assert self.stack[-1] is self.parent
    else:
      assert not self.stack
    self.stack.append(self)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert self.stack[-1] is self
    self.stack.pop(-1)

  def _get_name(self) -> str:
    assert self.parent and self.parent.maker
    reserved_names = set(self.childs.keys()) | self._ReservedNames
    for key, value in vars(self.parent.maker):
      if key in reserved_names:
        continue
      if value is self.maker:
        return key
    return self._get_unique_name()

  def _get_suggested_name(self) -> str:
    for call in self.maker.calls:
      if call is self:
        continue  # ignore this
      if call.name_ctx.parent is self.parent:
        return call.name_ctx.name
    return self.maker.get_canonical_name()

  def _get_unique_name(self) -> str:
    name = self._get_suggested_name()
    reserved_names = set(vars(self.parent.maker).keys()) | set(self.childs.keys()) | self._ReservedNames
    if name not in reserved_names:
      return name
    i = 0
    while True:
      name_ = f"{name}_{i}"
      if name_ not in reserved_names:
        return name_
      i += 1
