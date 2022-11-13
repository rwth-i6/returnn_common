"""
Hooks.

Similar as register_forward_hook etc. in PyTorch.
Some code adapted from PyTorch.
"""

from __future__ import annotations
from typing import Union, Dict, Callable
from types import FunctionType, MethodType
from collections import OrderedDict
from functools import partial
import sys
import weakref
from ... import nn


def register_call_post_hook(
  func_or_module: Union[nn.Module, MethodType, FunctionType, HookedFunction],
  hook: Callable,
) -> RemovableHandle:
  """
  Executes the hook after func_or_module was executed

  :param func_or_module: function or module
  :param hook: hook function. ``hook(func_or_module, input, output, **kwargs) -> None or modified output``.
    You can access ``func_or_module.__self__`` in case of a method to get the object (e.g. the module).
  :return: removable-handle
  """
  if isinstance(func_or_module, nn.Module):
    hooked = HookedModuleCall.setup(func_or_module)
  else:
    hooked = HookedFunction.setup(func_or_module)
  handle = RemovableHandle(hooked, hooked.post_hooks)
  hooked.post_hooks[handle.id] = hook
  return handle


class _Hooked:

  def __init__(self):
    self.post_hooks: Dict[int, Callable] = OrderedDict()

  def call(self, call_func, arg_func, *args, **kwargs):
    """Call"""
    result = call_func(*args, **kwargs)
    for hook in self.post_hooks.values():
      hook_result = hook(arg_func, args, result, **kwargs)
      if hook_result is not None:
        result = hook_result
    return result

  def update_after_remove(self):
    """Remove"""
    raise NotImplementedError


class HookedFunction(_Hooked):
  """
  Encapsulates a list of hooks on some function.
  """

  @classmethod
  def setup(cls, func: Union[nn.Module, MethodType, FunctionType, HookedFunction]) -> HookedFunction:
    """Setup"""
    if isinstance(func, MethodType):
      assert getattr(func.__self__, func.__name__).__func__ is func.__func__
      hooked = HookedFunction(func)
      setattr(func.__self__, func.__name__, hooked)
      return hooked
    elif isinstance(func, FunctionType):
      mod = sys.modules[func.__module__]
      assert getattr(mod, func.__name__) is func
      hooked = HookedFunction(func)
      setattr(mod, func.__name__, hooked)
      return hooked
    elif isinstance(func, HookedFunction):
      return func
    else:
      raise TypeError(f"func_or_module must be a function, method, or module, got {func!r}")

  def _restore(self):
    if isinstance(self.func, MethodType):
      assert getattr(self.func.__self__, self.func.__name__) is self
      setattr(self.func.__self__, self.func.__name__, self.func)
    elif isinstance(self.func, FunctionType):
      mod = sys.modules[self.func.__module__]
      assert getattr(mod, self.func.__name__) is self
      setattr(mod, self.func.__name__, self.func)
    else:
      raise TypeError(f"{self}.func unexpected type {type(self.func)}")

  def __init__(self, func):
    super(HookedFunction, self).__init__()
    self.func = func

  def __call__(self, *args, **kwargs):
    return self.call(self.func, self.func, *args, **kwargs)

  def update_after_remove(self):
    """Remove"""
    if not self.post_hooks:
      self._restore()


class HookedModuleCall:
  """
  Per each instance of the module.
  """
  @classmethod
  def setup(cls, module: nn.Module) -> _Hooked:
    """Setup"""
    if isinstance(module.__class__.__call__, HookedModuleCall):
      hooked = module.__class__.__call__
    else:
      hooked = HookedModuleCall(module.__class__)
      module.__class__.__call__ = lambda *args, **kwargs: hooked(*args, **kwargs)
    return hooked.setdefault_by_module(module)

  def _restore(self):
    self.module_cls.__call__ = self.orig_call

  def __init__(self, module_cls):
    self.module_cls = module_cls
    self.orig_call = module_cls.__call__
    self.hooks: Dict[int, HookedModuleCallInstance] = OrderedDict()

  def __call__(self, module: nn.Module, *args, **kwargs):
    assert isinstance(module, nn.Module)
    hooked = self.hooks.get(id(module))
    if hooked:
      reg_module = hooked.module_ref()
      if reg_module is not None:
        assert reg_module is module
        return hooked.call(partial(self.orig_call, module), module, *args, **kwargs)
      else:
        del self.hooks[hooked.module_id]
    return self.orig_call(module, *args, **kwargs)

  def setdefault_by_module(self, module: nn.Module) -> HookedModuleCallInstance:
    """
    Get the hooking for the module, or create if it does not exist yet.
    """
    reg = self.hooks.get(id(module))
    if reg:
      reg_module_ref, hooked = reg
      reg_module = reg_module_ref()
      if reg_module is not None:
        assert reg_module is module
        return hooked
    hooked = HookedModuleCallInstance(self, module)
    self.hooks[id(module)] = hooked
    return hooked

  def update_after_remove(self):
    """Update"""
    # Remove outdated entries.
    for reg in list(self.hooks.values()):
      if reg.module_ref() is None:
        del self.hooks[reg.module_id]
    if not self.hooks:
      self._restore()


class HookedModuleCallInstance(_Hooked):
  """
    Per each instance of the module.
  """
  def __init__(self, parent: HookedModuleCall, module: nn.Module):
    super(HookedModuleCallInstance, self).__init__()
    self.parent = parent
    self.module_ref = weakref.ref(module)
    self.module_id = id(module)

  def update_after_remove(self):
    """update"""
    if not self.post_hooks:
      del self.parent.hooks[self.module_id]
      self.parent.update_after_remove()


class RemovableHandle:
  """
  Provides the API to remove the hook.
  """

  next_id: int = 0

  def __init__(self, hooked: Union[HookedFunction, HookedModuleCallInstance], hooks_dict: Dict[int, Callable]):
    self.hooked_ref = weakref.ref(hooked)
    self.hooks_dict_ref = weakref.ref(hooks_dict)
    self.id = RemovableHandle.next_id
    RemovableHandle.next_id += 1

  def remove(self) -> None:
    """
    Removes the hook.
    """
    hooked = self.hooked_ref()
    hooks_dict = self.hooks_dict_ref()
    if hooked is not None and hooks_dict is not None and self.id in hooks_dict:
      del hooks_dict[self.id]
      hooked.update_after_remove()

  def __enter__(self) -> RemovableHandle:
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.remove()
