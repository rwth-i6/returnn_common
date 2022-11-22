"""
Test nn.cond
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_run_net_single_custom, \
  config_net_dict_via_serialized, dummy_default_in_dim

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


def test_cond_batch_norm():
  nn.reset_default_root_name_ctx()
  in_dim = nn.FeatureDim("in", 12)
  time_dim = nn.SpatialDim("time")
  x = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
  net = nn.BatchNorm(in_dim, use_mask=True)
  with nn.Cond(nn.dim_value(nn.batch_dim) % 2 == 0) as cond:
    cond.true = net(x)
    cond.false = x
  y = cond.result
  y.mark_as_default_output()
  config_str = nn.get_returnn_config().get_complete_py_code_str(net)
  dummy_run_net_single_custom(config_str)


def test_cond_chunking_conformer():
  # This test needs a huge stack size currently, due to the way RETURNN layer construction works currently.
  # On RETURNN side, there is the option flat_net_construction to solve this,
  # however, it's experimental and also does not work for this case.
  # https://github.com/rwth-i6/returnn/issues/957
  # https://stackoverflow.com/a/16248113/133374
  import resource
  import sys
  try:
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
  except Exception as exc:
    print(f"resource.setrlimit {type(exc).__name__}: {exc}")
  sys.setrecursionlimit(10 ** 6)

  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  input_dim = nn.FeatureDim("input", 10)
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))
  conformer = nn.ConformerEncoder(
    input_dim, nn.FeatureDim("out", 14), ff_dim=nn.FeatureDim("ff", 17),
    input_layer=None, num_heads=2, num_layers=1)
  window_dim = nn.SpatialDim("window", 50)

  with nn.Cond(nn.dim_value(nn.batch_dim) % 2 == 0) as cond:
    # chunking
    data_chunked, time_dim_ = nn.window(data, spatial_dim=time_dim, window_dim=window_dim, stride=25)

    # conformer on chunks
    out, _ = conformer(data_chunked, in_spatial_dim=window_dim)
    out.verify_out_shape({nn.batch_dim, time_dim_, window_dim, conformer.out_dim})

    # unchunking
    out_ = nn.inverse_window(out, in_spatial_dim=time_dim_, out_spatial_dim=time_dim, window_dim=window_dim, stride=25)
    out_.verify_out_shape({nn.batch_dim, time_dim, conformer.out_dim})
    cond.true = out_

    # no chunking
    out, _ = conformer(data, in_spatial_dim=time_dim)
    cond.false = out

  out_ = cond.result
  out_.mark_as_default_output()

  config_code = nn.get_returnn_config().get_complete_py_code_str(conformer)
  config, net_dict = config_net_dict_via_serialized(config_code)
  dummy_run_net(config, net=conformer)
