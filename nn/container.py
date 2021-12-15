"""
container functions
"""

from __future__ import annotations
from . import nn
from .base import Module, ILayerMaker, LayerRef, Layer
from typing import Iterable, Iterator, Union, Dict, Callable


_UnaryFuncT = Callable[[LayerRef], LayerRef]
_ModT = Union[ILayerMaker, _UnaryFuncT]


class ModuleList(Module):
  """
  Module list, getting passed an Iterable of Modules and creates a list of Modules in that order
  """
  def __init__(self, *modules: Union[_ModT, Iterable[_ModT], Dict[str, _ModT], ModuleList]):
    super().__init__()
    if len(modules) == 1 and isinstance(modules[0], dict):
      for key, module in modules[0].items():
        setattr(self, key, _convert_to_maker(module))
    elif len(modules) == 1 and isinstance(modules[0], ModuleList):
      for key, module in modules[0]._get_makers().items():
        setattr(self, key, _convert_to_maker(module))
    elif len(modules) == 1 and _is_iterable(modules[0]):
      for idx, module in enumerate(modules[0]):
        setattr(self, str(idx), _convert_to_maker(module))
    else:
      for idx, module in enumerate(modules):
        setattr(self, str(idx), _convert_to_maker(module))

  def _get_makers(self) -> Dict[str, ILayerMaker]:
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

  __call__ = Module.__call__  # stays abstract


class Sequential(ModuleList):
  """
  Sequential Module, takes callable of Modules which are then executed in sequence
  """
  @nn.scoped_method
  def __call__(self, inp) -> LayerRef:
    """
    Forward
    """
    for module in self:
      inp = module(inp)
    return inp


def sequential(source: LayerRef, *modules) -> Layer:
  """
  Wraps ``Sequential(*modules)(source)``
  """
  return Sequential(*modules)(source)


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

  @nn.scoped_method
  def __call__(self, *args, **kwargs) -> LayerRef:
    """
    Forward
    """
    return self.func(*args, **kwargs)


def _is_iterable(obj) -> bool:
  # No good generic way, so do this ugly hack.
  # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
  try:
    iter(obj)
    return True
  except TypeError:
    return False
