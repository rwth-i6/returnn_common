"""
Test nn.container
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict
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
      self.ls = nn.ModuleList([nn.Linear(self.base_dim + i) for i in range_(4)])

    @nn.scoped
    def __call__(self, out: nn.Tensor) -> nn.Tensor:
      """
      Forward
      """
      for layer in self.ls:
        out = layer(out)
      return out

  net = _Net()
  config, net_dict = dummy_config_net_dict(net)

  assert net_dict["ls.0"]["subnetwork"]["dot"]["from"][0] == "base:data:data"
  assert net_dict["ls.1"]["subnetwork"]["dot"]["from"][0] == "base:ls.0"
  assert net_dict["ls.2"]["subnetwork"]["dot"]["from"][0] == "base:ls.1"
  assert net_dict["ls.3"]["subnetwork"]["dot"]["from"][0] == "base:ls.2"
  assert net_dict["output"]["from"] == "ls.3"

  dummy_run_net(config)


def test_sequential_base_case():
  class _TestSequential(nn.Module):
    def __init__(self):
      super().__init__()
      self.seq = nn.Sequential(
        nn.Linear(nn.FeatureDim("feat1", 1)),
        nn.Linear(nn.FeatureDim("feat2", 2)),
        nn.Linear(nn.FeatureDim("feat3", 3)))

    @nn.scoped
    def __call__(self, data: nn.Tensor) -> nn.Tensor:
      """
      Forward
      """
      seq = self.seq(data)
      return seq

  net = _TestSequential()
  config, net_dict = dummy_config_net_dict(net)
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
      x = OrderedDict()
      x["one"] = nn.Linear(nn.FeatureDim("linear1-out", 1))
      x["two"] = nn.Linear(nn.FeatureDim("linear2-out", 2))
      x["three"] = nn.Linear(nn.FeatureDim("linear3-out", 3))
      self.seq = nn.Sequential(x)

    @nn.scoped
    def __call__(self, data: nn.Tensor) -> nn.Tensor:
      """
      Forward
      """
      seq = self.seq(data)
      return seq

  net = _TestSequential()
  config, net_dict = dummy_config_net_dict(net)

  assert net_dict["seq"]["subnetwork"]["one"]["subnetwork"]["dot"]["from"][0] == "base:base:data:data"
  assert net_dict["seq"]["subnetwork"]["two"]["subnetwork"]["dot"]["from"][0] == "base:one"
  assert net_dict["seq"]["subnetwork"]["three"]["subnetwork"]["dot"]["from"][0] == "base:two"
  assert net_dict["seq"]["subnetwork"]["output"]["from"] == "three"
  assert net_dict["output"]["from"] == "seq"
  dummy_run_net(config)
