"""
Base interfaces.

The core interfaces for the user are:

* :class:`ILayerMaker`, to directly create a layer dict
* :class:`Module`, to write PyTorch-style code

Instances of both objects can be called directly,
and return instances of type :class:`Layer`.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List


LayerDictRaw = Dict[str, Any]
LayerRefRaw = str
NetDictRaw = Dict[str, LayerDictRaw]


class Layer:
  """
  Represents a layer and its output, created by :class:`ILayerMaker`.

  TODO:
    extend this by functions __add__, __sub__, etc.
  """

  def __init__(self, *, maker: ILayerMaker, layer_dict: LayerDictRaw, name_ctx: _NameCtx):
    self.maker = maker
    self.layer_dict = layer_dict
    self.name_ctx = name_ctx
    name_ctx.layer = self

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

  def _sis_hash(self):
    from sisyphus.hash import sis_hash_helper
    return sis_hash_helper(self.layer_dict)


class ILayerMaker:
  """
  Makes a layer.
  """
  def make_layer_dict(self, *args, **kwargs) -> LayerDictRaw:
    """
    Return layer dict.

    The :class:`LayerDictRaw` can references other layers by using ``layer.get_name()``.
    """
    raise NotImplementedError

  def __call__(self, *args, name: Optional[str] = None, **kwargs) -> Layer:
    with _NameCtx(maker=self, name=name) as name_ctx:
      return Layer(maker=self, layer_dict=self.make_layer_dict(*args, **kwargs), name_ctx=name_ctx)


class CopyLayer(ILayerMaker):
  """
  Copy the (single) input.
  """
  def make_layer_dict(self, source: Layer) -> LayerDictRaw:
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

      def __init__(self, dim, activation=tanh):
        self.layer_norm = LayerNorm()
        self.linear = Linear(dim)
        self.activation = activation

      def forward(self, x: Layer) -> Layer:
        x_ = x
        x = self.layer_norm(x)
        x = self.linear(x)
        x = self.activation(x)
        return x_ + x

  """

  def forward(self, *args, **kwargs) -> Layer:
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


class _NameCtx:
  """
  This is a helper class to keep track of the current name context when creating layers.
  Usually you do not need to access this directly.
  """

  stack = []  # type: List[_NameCtx]

  @classmethod
  def top(cls) -> _NameCtx:
    """
    Return the top of the stack.
    Assumes that it exists.
    """
    assert cls.stack
    return cls.stack[-1]

  def __init__(self, *, maker: Optional[ILayerMaker], name: Optional[str] = None):
    self.maker = maker
    self.layer = None  # type: Optional[Layer]
    self.childs = {}  # type: Dict[str, _NameCtx]
    self.parent = self.stack[-1] if self.stack else None
    if self.parent:
      assert self.maker
    else:
      assert not self.maker
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
    reserved_names = set(self.childs.keys()) | {"output"}
    for key, value in vars(self.parent.maker):
      if key in reserved_names:
        continue
      if value is self.maker:
        return key
    return self._get_unique_name()

  def _get_unique_name(self) -> str:
    name = self.maker.__class__.__name__
    reserved_names = set(vars(self.parent.maker).keys()) | set(self.childs.keys()) | {"output"}
    if name not in reserved_names:
      return name
    i = 0
    while True:
      name_ = f"{name}_{i}"
      if name_ not in reserved_names:
        return name_
      i += 1
