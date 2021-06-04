"""
Base interfaces
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List


LayerDictRaw = Dict[str, Any]
LayerRefRaw = str
NetDictRaw = Dict[str, LayerDictRaw]


class Layer:
  """
  Represents a layer.
  """
  def __init__(self, *, maker: ILayerMaker, layer_dict: LayerDictRaw):
    self.maker = maker
    self.layer_dict = layer_dict
    self.assigned_name = None  # type: Optional[str]  # TODO ...

  def _sis_hash(self):
    from sisyphus.hash import sis_hash_helper
    return sis_hash_helper(self.layer_dict)


class ILayerMaker:
  """
  Makes a layer.
  """
  def call(self, *args, **kwargs) -> LayerDictRaw:
    """
    Return layer dict.
    """
    raise NotImplementedError

  def __call__(self, *args, **kwargs) -> Layer:
    # TODO...
    return Layer(maker=self, layer_dict=self.call(*args, **kwargs))


class Module(ILayerMaker):
  """
  Like PyTorch.
  This represents a subnetwork in RETURNN, or the root network.
  Or some other layer which has a subnetwork, like RecLayer.
  """

  def call(self, *args, **kwargs) -> Layer:
    """
    Constructs the output.
    """
    raise NotImplementedError

  def __call__(self, *args, **kwargs) -> Layer:
    # TODO...
    return self.call(*args, **kwargs)

  def make_root_net(self) -> _NameCtx:
    """
    Make net.
    """
    assert not _NameCtx.stack
    ctx = _NameCtx(module=self, layer=None)
    # TODO
    layer = self()
    return ctx


class _NameCtx:
  stack = []  # type: List[_NameCtx]

  def __init__(self, *, module: Optional[Module], layer: Optional[Layer]):
    self.layer = layer
    self.module = module
    self.childs = {}  # type: Dict[str, _NameCtx]
    self.stack.append(self)
