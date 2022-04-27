"""
Test nn._generated_layers
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_range_from_length():
  # https://github.com/rwth-i6/returnn_common/issues/134
  class _Net(nn.Module):
    @nn.scoped
    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      durations = nn.cast(nn.reduce(x, axis=x.feature_dim, mode="sum"), dtype="int32")
      durations.verify_out_shape({nn.batch_dim, axis})
      t, _ = nn.range_from_length(durations)
      return t

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, with_axis=True)
  dummy_run_net(config, net=net)
