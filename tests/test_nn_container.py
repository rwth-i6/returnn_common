"""
Test nn.container
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_default_in_dim
from builtins import range as range_
from pprint import pprint
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_module_list():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.base_dim = nn.FeatureDim("linear-out", 3)
      dims = [self.base_dim + i for i in range_(4)]
      in_dims = [dummy_default_in_dim] + dims[:-1]
      self.ls = nn.ModuleList([nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, dims)])

    def __call__(self, out: nn.Tensor) -> nn.Tensor:
      """
      Forward
      """
      for layer in self.ls:
        out = layer(out)
      return out

  config, net_dict, net = dummy_config_net_dict(_Net)

  assert net_dict["ls"]["subnetwork"]["0"]["subnetwork"]["dot"]["from"][0] == "base:base:data:data"
  assert net_dict["ls"]["subnetwork"]["1"]["subnetwork"]["dot"]["from"][0] == "base:0"
  assert net_dict["ls"]["subnetwork"]["2"]["subnetwork"]["dot"]["from"][0] == "base:1"
  assert net_dict["ls"]["subnetwork"]["3"]["subnetwork"]["dot"]["from"][0] == "base:2"
  assert net_dict["output"]["from"] == "ls"

  dummy_run_net(config, net=net)


def test_sequential_base_case():
  class _TestSequential(nn.Module):
    def __init__(self):
      super().__init__()
      dims = [nn.FeatureDim("feat1", 1), nn.FeatureDim("feat2", 2), nn.FeatureDim("feat3", 3)]
      in_dims = [dummy_default_in_dim] + dims[:-1]
      self.seq = nn.Sequential(nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(in_dims, dims))

    def __call__(self, data: nn.Tensor) -> nn.Tensor:
      """
      Forward
      """
      seq = self.seq(data)
      return seq

  config, net_dict, net = dummy_config_net_dict(_TestSequential)
  pprint(net_dict)

  assert net_dict["seq"]["subnetwork"]["0"]["subnetwork"]["dot"]["from"][0] == "base:base:data:data"
  assert net_dict["seq"]["subnetwork"]["1"]["subnetwork"]["dot"]["from"][0] == "base:0"
  assert net_dict["seq"]["subnetwork"]["2"]["subnetwork"]["dot"]["from"][0] == "base:1"
  assert net_dict["seq"]["subnetwork"]["output"]["from"] == "2"
  assert net_dict["output"]["from"] == "seq"


def test_sequential_named_case():
  class _TestSequential(nn.Module):
    def __init__(self):
      super().__init__()
      from collections import OrderedDict
      dims = [nn.FeatureDim("linear1-out", 1), nn.FeatureDim("linear2-out", 2), nn.FeatureDim("linear3-out", 3)]
      x = OrderedDict()
      x["one"] = nn.Linear(dummy_default_in_dim, dims[0])
      x["two"] = nn.Linear(dims[0], dims[1])
      x["three"] = nn.Linear(dims[1], dims[2])
      self.seq = nn.Sequential(x)

    def __call__(self, data: nn.Tensor) -> nn.Tensor:
      """
      Forward
      """
      seq = self.seq(data)
      return seq

  config, net_dict, net = dummy_config_net_dict(_TestSequential)

  assert net_dict["seq"]["subnetwork"]["one"]["subnetwork"]["dot"]["from"][0] == "base:base:data:data"
  assert net_dict["seq"]["subnetwork"]["two"]["subnetwork"]["dot"]["from"][0] == "base:one"
  assert net_dict["seq"]["subnetwork"]["three"]["subnetwork"]["dot"]["from"][0] == "base:two"
  assert net_dict["seq"]["subnetwork"]["output"]["from"] == "three"
  assert net_dict["output"]["from"] == "seq"
  dummy_run_net(config, net=net)


def test_parameter_list():
  class _TestParameterList(nn.Module):
    def __init__(self):
      super().__init__()
      in_dim = nn.FeatureDim("input", 13)
      self.param_list = nn.ParameterList([nn.Parameter([in_dim]) for _ in range(3)])

    def __call__(self, data: nn.Tensor) -> nn.Tensor:
      """
      Forward
      """
      for param in self.param_list:
        data = nn.combine(data, param, kind="add", allow_broadcast_all_sources=True)
      return data

  config, net_dict, net = dummy_config_net_dict(_TestParameterList)
  pprint(net_dict)

  dummy_run_net(config, net=net)
