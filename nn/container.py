"""
container functions
"""

from __future__ import annotations
from .base import Module, ILayerMaker, LayerRef
from typing import Iterable, Iterator, Optional, Union, Dict, Callable


_UnaryFuncT = Callable[[LayerRef], LayerRef]
_ModT = Union[ILayerMaker, _UnaryFuncT]


class ModuleList(Module):
  """
  Module list, getting passed an Iterable of Modules and creates a list of Modules in that order
  """
  def __init__(self, modules: Optional[Iterable[_ModT]] = None):
    super().__init__()
    if modules is not None:
      for idx, module in enumerate(modules):
        setattr(self, str(idx), _convert_to_maker(module))

  def _get_makers(self):
    return {key: value for (key, value) in vars(self).items() if isinstance(value, ILayerMaker)}

  def append(self, module: _ModT) -> ModuleList:
    """
    appends one module to the list
    """
    setattr(self, str(len(self)), _convert_to_maker(module))
    return self

  def extend(self, modules: Iterable[_ModT]) -> ModuleList:
    """
    appends multiple modules to the list
    """
    for module in modules:
      self.append(module)
    return self

  def __len__(self) -> int:
    return len(self._get_makers())

  def __iter__(self) -> Iterator[_ModT]:
    return iter(self._get_makers().values())

  def __getitem__(self, idx) -> ModuleList:
    from builtins import slice
    if isinstance(idx, slice):
      return self.__class__(dict(list(self._get_makers().items())[idx]))
    else:
      return list(self._get_makers().values())[idx]

  def __setitem__(self, idx: int, module: _ModT) -> None:
    key = list(self._get_makers().keys())[idx]
    return setattr(self, key, _convert_to_maker(module))

  forward = Module.forward  # stays abstract


class Sequential(ModuleList):
  """
  Sequential Module, takes callable of Modules which are then executed in sequence
  """

  def __init__(self, *modules: Union[_ModT, Dict[str, _ModT]]):
    super().__init__()
    if len(modules) == 1 and isinstance(modules[0], dict):
      for key, module in modules[0].items():
        setattr(self, key, _convert_to_maker(module))
    else:
      for idx, module in enumerate(modules):
        setattr(self, str(idx), _convert_to_maker(module))

  def forward(self, inp) -> LayerRef:
    """
    Forward
    """
    for module in self:
      inp = module(inp)
    return inp


def _convert_to_maker(obj: _ModT) -> ILayerMaker:
  if isinstance(obj, ILayerMaker):
    return obj
  elif callable(obj):
    return WrappedFunction(obj)
  else:
    raise TypeError(f"did not expect {obj!r}")


class WrappedFunction(Module):
  """
  Wrap any function as a module.
  """
  def __init__(self, func):
    super().__init__()
    assert callable(func)
    self.func = func

  def forward(self, *args, **kwargs) -> LayerRef:
    """
    Forward
    """
    return self.func(*args, **kwargs)
