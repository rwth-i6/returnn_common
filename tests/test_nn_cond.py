"""
Test nn.cond
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_run_net_single_custom, dummy_default_in_dim

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
      self.linear_true = nn.Linear(dummy_default_in_dim, out_dim)
      self.linear_false = nn.Linear(dummy_default_in_dim, out_dim)

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      return x

  config, net_dict, net = dummy_config_net_dict(_Net)
  dummy_run_net(config, net=net)


def test_cond_shared_params():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(dummy_default_in_dim, nn.FeatureDim("linear-out", 13))

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear(x)
        cond.false = self.linear(x * 2.)
        x = cond.result
      return x

  config, net_dict, net = dummy_config_net_dict(_Net)
  engine = dummy_run_net(config, net=net)
  params = engine.network.get_params_list()
  print(params)
  assert len(params) == 2
  assert params[0].name == "linear/bias/param:0"


def test_cond_twice_shared_params():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      out_dim = nn.FeatureDim("linear-out", 13)
      self.pre_linear = nn.Linear(dummy_default_in_dim, out_dim)
      self.linear_true = nn.Linear(out_dim, out_dim)
      self.linear_false = nn.Linear(out_dim, out_dim)

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

  config, net_dict, net = dummy_config_net_dict(_Net)
  dummy_run_net(config, net=net)


def test_cond_random():
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

  config, net_dict, net = dummy_config_net_dict(_Net)
  dummy_run_net(config, net=net)


def test_cond_new_axis():
  # Like in SelfAttention.
  nn.reset_default_root_name_ctx()
  in_dim = nn.FeatureDim("in", 12)
  time_dim = nn.SpatialDim("time")
  x = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
  net = nn.Linear(in_dim, in_dim)
  axis = time_dim

  with nn.Cond(nn.dim_value(nn.batch_dim) % 2 == 0) as cond:
    x_ = x
    x_ = net(x_)
    new_dim = nn.SpatialDim(f"{axis.description}:new-dim")
    x_, _ = nn.reinterpret_new_dim(x_, in_dim=axis, out_dim=new_dim)
    x_ = net(x_)
    cond.true = nn.reduce(x_, axis=new_dim, mode="max")
    cond.false = nn.reduce(x, axis=axis, mode="max")
  y = cond.result
  y.mark_as_default_output()

  config_str = nn.get_returnn_config().get_complete_py_code_str(net)
  dummy_run_net_single_custom(config_str)
