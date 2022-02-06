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

    @nn.scoped
    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(x, axis=nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      return x

  config, net_dict = dummy_config_net_dict(net=_Net())
  dummy_run_net(config)


def test_cond_shared_params():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(x, axis=nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear(x)
        cond.false = self.linear(x * 2.)
        x = cond.result
      return x

  config, net_dict = dummy_config_net_dict(net=_Net())
  engine = dummy_run_net(config)
  params = engine.network.get_params_list()
  print(params)
  assert len(params) == 2
  assert params[0].name == "linear/bias/param:0"


def test_cond_twice_shared_params():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      out_dim = nn.FeatureDim("linear-out", 13)
      self.pre_linear = nn.Linear(out_dim)
      self.linear_true = nn.Linear(out_dim, in_dim=out_dim)
      self.linear_false = nn.Linear(out_dim, in_dim=out_dim)

    @nn.scoped
    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      x = self.pre_linear(x)
      with nn.Cond(nn.length(x, axis=nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      with nn.Cond(nn.length(x, axis=nn.batch_dim) % 2 == 1) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      return x

  config, net_dict = dummy_config_net_dict(net=_Net())
  dummy_run_net(config)
