"""
Test layers
"""

from . import _setup_test_env  # noqa
from returnn_common.models.layers import *
from returnn_common.models.base import get_extern_data
from pprint import pprint
from nose.tools import assert_equal


def test_simple_net():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.lstm = Lstm(n_out=13)

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x = self.lstm(x)
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert "lstm" in net_dict


def test_simple_net_share_params():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.lstm = Lstm(n_out=13)

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x = self.lstm(x)
      x = self.lstm(x)
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert "lstm" in net_dict
  assert "lstm_0" in net_dict
  assert_equal(net_dict["lstm_0"]["reuse_params"], "lstm")
