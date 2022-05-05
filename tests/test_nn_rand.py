"""
Test nn.rand
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


def test_random_normal():
  nn.reset_default_root_name_ctx()

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.rnd = nn.Random()

    @nn.scoped
    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      return x + self.rnd.normal(x.shape_ordered)

  net = _Net()
  config, net_dict = dummy_config_net_dict(net, reset_name_ctx=False)
  pprint(net_dict)
  dummy_run_net(config, net=net)


def test_random_multi_call():
  # https://github.com/rwth-i6/returnn_common/issues/148
  # Actually we will not really test the non-determinism as this is difficult to test.
  # We just test whether we can run it multiple times without error.
  nn.reset_default_root_name_ctx()

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.rnd = nn.Random()

    @nn.scoped
    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      return x + self.rnd.normal(x.shape_ordered) - self.rnd.normal(x.shape_ordered)

  net = _Net()
  config, net_dict = dummy_config_net_dict(net, reset_name_ctx=False)
  pprint(net_dict)
  dummy_run_net(config, net=net)
