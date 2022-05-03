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
# nn.enable_debug_eager_mode()


def test_specaugment_v2():
  time_dim = nn.SpatialDim("time")
  feat_dim = nn.FeatureDim("feat", 50)
  audio = nn.get_extern_data(nn.Data("input", dim_tags=[nn.batch_dim, time_dim, feat_dim]))
  from ..asr import specaugment
  masked = specaugment.specaugment_v2(
    audio, spatial_dim=time_dim, global_train_step_dependent=False, only_on_train=False)
  print(masked)
  code_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  print(code_str)
