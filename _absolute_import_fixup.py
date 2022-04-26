"""
In case you do `import nn` and have set up the root of return_common as a src root
(e.g. while developing on returnn-common in PyCharm or elsewhere),
this does not work because it expects that there is a parent package.
This script fixes that.

This file is supposed to be symlinked into the sub-package directory,
and this is supposed to be in the __init__.py::

  from . import _absolute_import_fixup  # noqa

"""

import sys
import os

# noinspection PyUnboundLocalVariable
__file__ = os.path.abspath(__file__)
assert os.path.islink(__file__), "should not be imported directly"
_ln = os.readlink(__file__)
assert _ln == "../" + os.path.basename(__file__), f"unexpected link: {_ln!r}"
__my_dir__ = os.path.dirname(__file__)
__rc_dir__ = os.path.dirname(__my_dir__)
__rc_parent_dir__ = os.path.dirname(__rc_dir__)


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

  rc_package_name = os.path.basename(__rc_dir__)
  new_package_name = rc_package_name + "." + old_package_name
  package.__name__ = package.__package__ = new_package_name
  sys.modules[package.__name__] = package

  import importlib.machinery
  assert isinstance(package.__loader__, importlib.machinery.SourceFileLoader)
  package.__loader__.name = new_package_name
  assert isinstance(package.__spec__, importlib.machinery.ModuleSpec)
  package.__spec__.name = new_package_name

  # Now we can import the parent package.
  importlib.import_module(rc_package_name)

  globals()["__package__"] = package.__name__  # fixup ourselves


_fixup_package_hierarchy()
