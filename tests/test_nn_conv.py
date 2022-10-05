"""
Test nn.conv
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_default_in_dim
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_conv1d():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      # Use some downsampling + valid padding to test dim tag math.
      self.conv = nn.Conv1d(dummy_default_in_dim, nn.FeatureDim("out", 13), 4, strides=3, padding="valid")

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """
      Forward
      """
      x, _ = self.conv(x, in_spatial_dim=axis)
      return x

  config, net_dict, net = dummy_config_net_dict(_Net, with_axis=True)
  dummy_run_net(config, net=net)
