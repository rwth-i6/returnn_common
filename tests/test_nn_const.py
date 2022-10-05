"""
test asr.specaugment
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_ones():
  nn.enable_debug_eager_mode()
  feat_dim = nn.FeatureDim("feat", 5)
  ones = nn.ones([feat_dim])
  assert ones.data.dtype == "float32"
  nn.disable_debug_eager_mode()


def test_ones_like_new_spatial_dim():
  class _Net(nn.Module):
    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      # Make new spatial dim with unknown value (so no pool, conv or so).
      loop = nn.Loop(max_seq_len=nn.constant(value=5))
      loop.state.i = nn.zeros([nn.batch_dim])
      with loop:
        loop.state.i = loop.state.i + 1.
        loop.end(loop.state.i >= 5., include_eos=True)
        y = loop.stack(loop.state.i * nn.reduce(x, mode="mean", axis=axis))
      assert set(y.data.dim_tags) != set(x.data.dim_tags)
      return nn.ones_like(y) + y

  config, net_dict, net = dummy_config_net_dict(_Net, with_axis=True)
  dummy_run_net(config)
