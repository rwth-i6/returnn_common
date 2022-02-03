"""
container functions
"""

from __future__ import annotations
from .. import nn
from typing import Iterable, Iterator, Union, Dict, Callable


_UnaryFuncT = Callable[[nn.TensorRef], nn.TensorRef]
_ModT = Union[nn.Module, _UnaryFuncT]


class ModuleList(nn.Module):
  """
  Module list, getting passed an Iterable of Modules and creates a list of Modules in that order
  """
  def __init__(self, *modules: Union[_ModT, Iterable[_ModT], Dict[str, _ModT], ModuleList]):
    super().__init__()
    if len(modules) == 1 and isinstance(modules[0], dict):
      for key, module in modules[0].items():
        setattr(self, key, _convert_to_module(module))
    elif len(modules) == 1 and isinstance(modules[0], ModuleList):
      for key, module in modules[0]._get_modules().items():
        setattr(self, key, _convert_to_module(module))
    elif len(modules) == 1 and _is_iterable(modules[0]):
      for idx, module in enumerate(modules[0]):
        setattr(self, str(idx), _convert_to_module(module))
    else:
      for idx, module in enumerate(modules):
        setattr(self, str(idx), _convert_to_module(module))

  def _get_modules(self) -> Dict[str, nn.Module]:
    return {key: value for (key, value) in vars(self).items() if isinstance(value, nn.Module)}

  def append(self, module: _ModT) -> ModuleList:
    """
    appends one module to the list
    """
    setattr(self, str(len(self)), _convert_to_module(module))
    return self

  def extend(self, modules: Iterable[_ModT]) -> ModuleList:
    """
    appends multiple modules to the list
    """
    for module in modules:
      self.append(module)
    return self

  def __len__(self) -> int:
    return len(self._get_modules())

  def __iter__(self) -> Iterator[_ModT]:
    return iter(self._get_modules().values())

  def __getitem__(self, idx) -> Union[ModuleList, nn.Module]:
    from builtins import slice
    if isinstance(idx, slice):
      return self.__class__(dict(list(self._get_modules().items())[idx]))
    else:
      return list(self._get_modules().values())[idx]

  def __setitem__(self, idx: int, module: _ModT) -> None:
    key = list(self._get_modules().keys())[idx]
    return setattr(self, key, _convert_to_module(module))

  __call__ = nn.Module.__call__  # stays abstract


class Sequential(ModuleList):
  """
  Sequential Module, takes callable of Modules which are then executed in sequence
  """
  @nn.scoped
  def __call__(self, inp, **kwargs) -> nn.TensorRef:
    """
    Forward
    """
    for module in self:
      inp = module(inp, **kwargs)
    return inp


def sequential(source: nn.TensorRef, *modules) -> nn.TensorRef:
  """
  Wraps ``Sequential(*modules)(source)``
  """
  return Sequential(*modules)(source)


def _convert_to_module(obj: _ModT) -> nn.Module:
  if isinstance(obj, nn.Module):
    return obj
  elif callable(obj):
    return nn.Functional(obj)
  else:
    raise TypeError(f"did not expect {obj!r}")


def _is_iterable(obj) -> bool:
  # No good generic way, so do this ugly hack.
  # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
  try:
    iter(obj)
    return True
  except TypeError:
    return False
