"""
Base interfaces.

The core interfaces for the user are:

* :class:`ILayerMaker`, to directly create a layer dict
* :class:`Module`, to write PyTorch-style code

Instances of both objects can be called directly,
and return instances of type :class:`LayerRef`.

TODO: losses should be handled explicitly, more like PyTorch,
  and then mark_as_loss() or so (which would set "loss": "as_is").
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from returnn.util.basic import NotSpecified


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
    cur_scope = _NameCtx.top()
    if self.name_ctx.parent is cur_scope:  # fast path
      return self.name_ctx.name
    cur_scope_abs = cur_scope.get_abs_name_ctx_list()
    self_name_abs = self.name_ctx.get_abs_name_ctx_list()
    assert cur_scope_abs[0] is self_name_abs[0]  # same root
    assert len(cur_scope_abs) > len(self_name_abs)  # not implemented otherwise
    common_len = 0
    while cur_scope_abs[common_len] is self_name_abs[common_len]:
      common_len += 1
    assert common_len == len(self_name_abs) - 2  # not implemented otherwise
    return "base:" * (len(cur_scope_abs) - len(self_name_abs) + 1) + self.name_ctx.name


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

  def _sis_hash(self):
    from sisyphus.hash import sis_hash_helper
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

    The :class:`LayerDictRaw` can references other layers by using ``layer.get_name()``.
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
        layer_dict["reuse_params"] = self.calls[0].get_name()
      layer = Layer(self, layer_dict)
      self.calls.append(layer)
      return layer


class CopyLayer(ILayerMaker):
  """
  Copy the (single) input.
  """
  def make_layer_dict(self, source: LayerRef) -> LayerDictRaw:
    """
    Create CopyLayer.
    """
    return {"class": "copy", "from": source.get_name()}


class Module(ILayerMaker):
  """
  Like PyTorch.
  This represents a subnetwork in RETURNN, or the root network.
  Or some other layer which has a subnetwork, like RecLayer.

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
    res = self.forward(*args, **kwargs)
    CopyLayer()(res, name="output")
    name_ctx = _NameCtx.top()
    assert name_ctx.maker is self
    return {"class": "subnetwork", "from": [], "subnetwork": name_ctx.make_net_dict()}

  def make_root_net_dict(self) -> NetDictRaw:
    """
    Make net dict, to be used as the main RETURNN network, not within a subnetwork.
    Extern data can be accessed via :func:`get_root_extern_data`.
    """
    with _NameCtx(maker=self) as name_ctx:
      self.forward()
      return name_ctx.make_net_dict()


def get_root_extern_data(data_key: str) -> LayerRef:
  """
  Get extern data.
  """
  scope = _NameCtx.top()  # must exist
  scope_abs = scope.get_abs_name_ctx_list()
  root_scope = scope_abs[0]
  root_layer_name = f"data:{data_key}"
  if root_layer_name in root_scope.childs:
    name_ = root_scope.childs[root_layer_name]
    assert name_.layer_ref
  else:
    name_ = _NameCtx(name=root_layer_name, parent=root_scope, maker=None)
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

  def __init__(self, *,
               maker: Optional[ILayerMaker],
               name: Optional[str] = None,
               parent: Optional[_NameCtx] = NotSpecified):
    self.maker = maker
    self.layer_ref = None  # type: Optional[LayerRef]
    self.layer = None  # type: Optional[Layer]
    self.childs = {}  # type: Dict[str, _NameCtx]
    self.parent = parent if parent is not NotSpecified else (self.stack[-1] if self.stack else None)
    self.name = name if name else (self._get_name() if self.parent else None)
    if self.parent:
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
