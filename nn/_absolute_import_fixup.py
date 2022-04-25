"""
In case you do `import nn` and have set up the root of return_common as a src root
(e.g. while developing on returnn-common in PyCharm or elsewhere),
this does not work because it expects that there is a parent package.
This script fixes that.
"""

import sys
import os
import importlib

__nn_dir__ = os.path.dirname(os.path.abspath(__file__))
__rc_dir__ = os.path.dirname(__nn_dir__)
__rc_parent_dir__ = os.path.dirname(__rc_dir__)


def _fixup_package_hierarchy():
  if "." in __package__:
    # We are already in a sub-package, so all should be fine.
    return

  old_nn_abs_package_name = "nn"
  assert __package__ == old_nn_abs_package_name, f"__package__ {__package__!r} unexpected"
  nn_package = sys.modules[old_nn_abs_package_name]
  assert nn_package.__name__ == nn_package.__package__ == old_nn_abs_package_name

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
  nn_package.__name__ = nn_package.__package__ = rc_package_name + "." + old_nn_abs_package_name
  sys.modules[nn_package.__name__] = nn_package

  # Now we can import the parent package.
  importlib.import_module(rc_package_name)

  globals()["__package__"] = nn_package.__name__  # fixup ourselves


_fixup_package_hierarchy()
