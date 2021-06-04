"""
Base interfaces
"""

from __future__ import annotations
from typing import Dict, Any, Optional


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
  This represents a subnetwork in RETURNN.
  """

  def call(self, *args, **kwargs) -> Layer:
    """
    Constructs the output.
    """
    raise NotImplementedError

  def __call__(self, *args, **kwargs) -> Layer:
    # TODO...
    return self.call(*args, **kwargs)
