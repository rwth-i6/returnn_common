"""
Test layers
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net

from returnn_common.models.layers import *
from returnn_common.models.base import get_extern_data, NameCtx, Layer, LayerRef
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
  dummy_run_net(net_dict)


def test_simple_net_share_params():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.linear = Linear(n_out=13, activation=None)
      self.lstm = Lstm(n_out=13)

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x = self.linear(x)
      x = self.lstm(x)
      x = self.lstm(x)
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert "lstm" in net_dict
  assert "lstm_0" in net_dict
  assert_equal(net_dict["lstm_0"]["reuse_params"], "lstm")
  dummy_run_net(net_dict)


def test_explicit_root_ctx():
  class Net(Module):
    """
    Net
    """
    def __init__(self, l2=1e-07, dropout=0.1, n_out=13):
      super().__init__()
      self.linear = Linear(n_out=n_out, l2=l2, dropout=dropout, with_bias=False, activation=None)

    def forward(self, x: LayerRef) -> LayerRef:
      """
      forward
      """
      x = self.linear(x)
      return x

  with NameCtx.new_root() as name_ctx:
    net = Net()
    out = net(get_extern_data("data"))
    assert isinstance(out, Layer)
    assert_equal(out.get_name(), "Net")

    Copy()(out, name="output")  # make some dummy output layer
    net_dict = name_ctx.make_net_dict()
    pprint(net_dict)

  assert "Net" in net_dict
  sub_net_dict = net_dict["Net"]["subnetwork"]
  assert "linear" in sub_net_dict
  lin_layer_dict = sub_net_dict["linear"]
  assert_equal(lin_layer_dict["class"], "linear")
  assert_equal(lin_layer_dict["from"], "base:data:data")
  dummy_run_net(net_dict)


def test_root_mod_call_twice():
  class TestBlock(Module):
    """
    Test block
    """
    def __init__(self, l2=1e-07, dropout=0.1, n_out=13):
      super().__init__()
      self.linear = Linear(n_out=n_out, l2=l2, dropout=dropout, with_bias=False, activation=None)

    def forward(self, x: LayerRef) -> LayerRef:
      """
      forward
      """
      x = self.linear(x)
      return x

  with NameCtx.new_root() as name_ctx:
    test_block = TestBlock()
    y = test_block(get_extern_data("input1"))
    z = test_block(get_extern_data("input2"))

    print(y)
    assert isinstance(y, LayerRef)
    assert_equal(y.get_name(), "TestBlock")
    print(z)
    assert isinstance(z, LayerRef)
    assert_equal(z.get_name(), "TestBlock_0")

    net_dict = name_ctx.make_net_dict()
    pprint(net_dict)

  assert "TestBlock" in net_dict and "TestBlock_0" in net_dict
  assert_equal(net_dict["TestBlock_0"]["reuse_params"], "TestBlock")
