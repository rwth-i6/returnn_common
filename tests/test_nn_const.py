"""
test asr.specaugment
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


# Enables it globally now.
nn.enable_debug_eager_mode()


def test_ones():
  feat_dim = nn.FeatureDim("feat", 5)
  ones = nn.ones([feat_dim])
  assert ones.data.dtype == "float32"
