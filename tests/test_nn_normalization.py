"""
Test nn.normalization
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_default_in_dim
from pprint import pprint
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_batch_norm():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.bn = nn.BatchNorm(dummy_default_in_dim, use_mask=False)

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      """forward"""
      return self.bn(x)

  config, net_dict, net = dummy_config_net_dict(_Net)
  pprint(net_dict)
  dummy_run_net(config, net=net)
