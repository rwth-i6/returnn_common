"""
Test nn.cond
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict

import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_cond():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      out_dim = nn.FeatureDim("linear-out", 13)
      self.linear_true = nn.Linear(out_dim)
      self.linear_false = nn.Linear(out_dim)

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      return x

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net)
  dummy_run_net(config, net=net)


def test_cond_shared_params():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear(x)
        cond.false = self.linear(x * 2.)
        x = cond.result
      return x

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net)
  engine = dummy_run_net(config, net=net)
  params = engine.network.get_params_list()
  print(params)
  assert len(params) == 2
  assert params[0].name == "linear/bias/param:0"


def test_cond_twice_shared_params():
  nn.reset_default_root_name_ctx()

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      out_dim = nn.FeatureDim("linear-out", 13)
      self.pre_linear = nn.Linear(out_dim)
      self.linear_true = nn.Linear(out_dim, in_dim=out_dim)
      self.linear_false = nn.Linear(out_dim, in_dim=out_dim)

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      x = self.pre_linear(x)
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 1) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      return x

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, reset_name_ctx=False)
  dummy_run_net(config, net=net)


def test_cond_random():
  nn.reset_default_root_name_ctx()

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.rnd = nn.Random()

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = x + self.rnd.normal(x.shape_ordered)
        cond.false = x
        x = cond.result
      return x

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, reset_name_ctx=False)
  dummy_run_net(config, net=net)
