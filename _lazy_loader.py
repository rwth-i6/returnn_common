"""
Lazy module loader.
Code adapted from TensorFlow.
"""

import importlib
import types
from typing import Dict, Any


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies."""

    def __init__(self, local_name: str, parent_module_globals: Dict[str, Any]):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        name = f'{parent_module_globals["__package__"]}.{local_name}'
        super(LazyLoader, self).__init__(name)
        parent_module_globals[local_name] = self

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)

        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
