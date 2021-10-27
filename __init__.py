"""
Common building blocks for RETURNN,
such as models or networks,
network definition code,
datasets, etc.
"""

import sys as _sys

# We require at least Python 3.7.
# See https://github.com/rwth-i6/returnn_common/issues/43.
# Current features we use:
# - Our code expects that the order of dict is deterministic, or even insertion order specifically.
# - Type annotations.
assert _sys.version_info[:2] >= (3, 7)
