"""
Test nn.attention
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict
from pprint import pprint
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_self_attention():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.self_att = nn.SelfAttention(
        key_dim_total=nn.FeatureDim("key-dim-total", 21),
        value_dim_total=nn.FeatureDim("value-dim-total", 33),
        num_heads=3)

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """forward"""
      return self.self_att(x, axis=axis)

  net = _Net()
  config, net_dict = dummy_config_net_dict(net, with_axis=True)
  pprint(net_dict)
  dummy_run_net(config, net=net)
