"""
In case you do `import nn` and have set up the root of return_common as a src root
(e.g. while developing on returnn-common in PyCharm or elsewhere),
this does not work because it expects that there is a parent package.
This script fixes that.

This file is supposed to be symlinked into the sub-package directory,
and this is supposed to be in the __init__.py::

  from . import _absolute_import_fixup  # noqa

"""

from typing import Optional, List, Set
import sys
import os
import importlib
import importlib.machinery

# noinspection PyUnboundLocalVariable
__file__ = os.path.abspath(__file__)
assert os.path.islink(__file__), "should not be imported directly"
_ln = os.readlink(__file__)
assert _ln == "../" + os.path.basename(__file__), f"unexpected link: {_ln!r}"
__my_dir__ = os.path.dirname(__file__)
__rc_dir__ = os.path.dirname(__my_dir__)
__rc_parent_dir__ = os.path.dirname(__rc_dir__)
__rc_package_name__ = os.path.basename(__rc_dir__)


def _fixup_package_hierarchy():
    if "." in __package__:
        # We are already in a sub-package, so all should be fine.
        return

    old_package_name = __package__
    package = sys.modules[old_package_name]
    assert package.__name__ == package.__package__ == old_package_name

    # We want to load the parent package.
    # We could insert some custom MetaPathFinder into sys.meta_path
    # to find the parent package.
    # Or we could just extend sys.path to find the parent package.
    # We do the latter for simplicity now.

    new_package_search_path = __rc_parent_dir__
    if new_package_search_path not in sys.path:
        # Note: It might be a problem to add it at the end, when it is e.g. also installed via pip.
        sys.path.append(new_package_search_path)

    new_package_name = __rc_package_name__ + "." + old_package_name
    package.__name__ = package.__package__ = new_package_name
    sys.modules[package.__name__] = package

    import importlib.machinery

    assert isinstance(package.__loader__, importlib.machinery.SourceFileLoader)
    package.__loader__.name = new_package_name
    assert isinstance(package.__spec__, importlib.machinery.ModuleSpec)
    package.__spec__.name = new_package_name

    # Now we can import the parent package.
    importlib.import_module(__rc_package_name__)

    globals()["__package__"] = package.__name__  # fixup ourselves

    # Now, when other sub-packages are being imported by the user as absolute imports,
    # we also want that this works (e.g. `import asr.gt`).
    # For that, we need a meta-path-finder.
    if _MetaPathFinderOldAbsName not in sys.meta_path:
        sys.meta_path.insert(0, _MetaPathFinderOldAbsName)  # noqa
    _MetaPathFinderOldAbsName.old_package_names.add(old_package_name)


class _MetaPathFinderOldAbsName(importlib.abc.MetaPathFinder):
    old_package_names = set()  # type: Set[str]

    @classmethod
    def find_spec(cls, fullname: str, path: Optional[List[str]], target=None):
        """
        https://docs.python.org/3/library/importlib.html#importlib.abc.MetaPathFinder.find_spec
        """
        for old_package_name in cls.old_package_names:
            if fullname.startswith(old_package_name + "."):
                spec = importlib.machinery.PathFinder.find_spec(fullname, path=path, target=target)
                assert isinstance(spec, importlib.machinery.ModuleSpec)
                assert isinstance(spec.loader, importlib.machinery.SourceFileLoader)
                spec.name = spec.loader.name = __rc_package_name__ + "." + fullname
                spec.loader = _LoaderSetupOldAbsName(base_loader=spec.loader)
                return spec
        return None


class _LoaderSetupOldAbsName(importlib.abc.Loader):
    """
    https://docs.python.org/3/library/importlib.html#importlib.abc.Loader
    """

    def __init__(self, base_loader: importlib.abc.Loader):
        super(_LoaderSetupOldAbsName, self).__init__()
        self.base_loader = base_loader

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        """
        https://docs.python.org/3/library/importlib.html#importlib.abc.Loader.create_module
        """
        return self.base_loader.create_module(spec)

    def exec_module(self, module):
        """
        https://docs.python.org/3/library/importlib.html#importlib.abc.Loader.exec_module
        """
        self.base_loader.exec_module(module)
        assert module.__name__.startswith(__rc_package_name__ + ".")
        sys.modules[module.__name__[len(__rc_package_name__) + 1 :]] = module


_fixup_package_hierarchy()
