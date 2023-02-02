"""
Common building blocks for RETURNN,
such as models or networks,
network definition code,
datasets, etc.
"""

import typing
import sys
from ._lazy_loader import LazyLoader

# We require at least Python 3.7.
# See https://github.com/rwth-i6/returnn_common/issues/43.
# Current features we use:
# - Our code expects that the order of dict is deterministic, or even insertion order specifically.
# - Type annotations.
assert sys.version_info[:2] >= (3, 7)

# Now all the imports.
# Use lazy imports, but only when not type checking.
if typing.TYPE_CHECKING:
    from . import nn  # noqa

else:
    LazyLoader("nn", globals())


del typing
del sys
del LazyLoader
