"""
Wrap RETURNN layers
"""

from ._generated_layers import *  # noqa
from .base import Module  # noqa
from typing import Iterator, Iterable


def split(source: LayerRef, *,
          axis: Optional[str] = NotSpecified,
          num_splits: Optional[int] = NotSpecified,
          size_splits: Optional[List[int]] = NotSpecified,
          ) -> Tuple[LayerRef, ...]:
  """
  Split the input on the specified axis (by default feature).
  Basically a wrapper around tf.split.
  """
  from ._generated_layers import _split
  from .base import get_sub_layer
  res = _split(source, axis=axis, num_splits=num_splits, size_splits=size_splits)
  if num_splits is None:
    assert isinstance(size_splits, (tuple, list))
    num_splits = len(size_splits)
  return tuple(get_sub_layer(res, str(i)) for i in range(num_splits))


class Lstm(Rec):
  """
  LSTM
  """
  def __init__(self, *, rec_weight_dropout=0, rec_weight_dropout_shape=None, **kwargs):
    assert "unit_opts" not in kwargs, "we handle that here"
    unit_opts = {}
    if rec_weight_dropout:
      unit_opts["rec_weight_dropout"] = rec_weight_dropout
    if rec_weight_dropout_shape:
      unit_opts["rec_weight_dropout_shape"] = rec_weight_dropout_shape
    super(Lstm, self).__init__(
      unit="nativelstm2", unit_opts=unit_opts, **kwargs)


class ModuleList(Module):
  """
  Module list, getting passed an Interable of Modules and creates a list of Modules in that order
  """
  def __init__(self, modules: Optional[Iterable[Module]] = None):
    super().__init__()
    self._modules = []
    if modules is not None:
      for module in modules:
        self._modules.append(module)

  def append(self, module: Module) -> "ModuleList":
    """
    appends one module to the list"
    """
    self._modules.append(module)
    return self

  def extend(self, modules: Iterable[Module]) -> "ModuleList":
    """
    appends multiple modules to the list
    """
    for module in modules:
      self._modules.append(module)
    return self

  def insert(self, index: int, module: Module) -> "ModuleList":
    """
    inserts one module at a certain position into the list
    """
    self._modules.insert(index, module)
    return self

  def __len__(self) -> int:
    return len(self._modules)

  def __iter__(self) -> Iterator[Module]:
    return iter(self._modules)

  def __getitem__(self, idx: int) -> Module:
    return self._modules[idx]

  def forward(self) -> LayerRef:
    """
    Constructs the output.
    You can write PyTorch-style code here.
    """
    raise NotImplementedError
