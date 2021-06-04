
"""
Import this module to have the side effect
that the RETURNN root dir will be added to ``sys.path``,
i.e. ``import returnn`` works afterwards.

In your code (e.g. tools or demos), you would have this::

    import _setup_returnn_env  # noqa

The ``# noqa`` is to ignore the warning that this module is not used.
"""

import os
import sys

my_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
root_dir = os.path.dirname(my_dir)
parent_root_dir = os.path.dirname(root_dir)

sys.path.insert(0, parent_root_dir)  # make `import returnn_common` work
