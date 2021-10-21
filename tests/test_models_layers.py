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


def test_simple_net_linear():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.linear = Linear(n_out=13)

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x = self.linear(x)
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert "linear" in net_dict
  dummy_run_net(net_dict)


def test_simple_net_arithmetic():
  class _Net(Module):
    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x = 1. / x + x * 2.
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  dummy_run_net(net_dict)


def test_eval():
  class _Net(Module):
    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x = eval(x, eval="source(0) * 2.5")
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  dummy_run_net(net_dict)


def test_simple_net_lstm():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.lstm = Lstm(n_out=13)

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x, _ = self.lstm(x)
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert "lstm" in net_dict
  dummy_run_net(net_dict)


def test_simple_net_explicit_root_ctx():
  lstm = Lstm(n_out=13)

  with NameCtx.new_root() as name_ctx:
    out, _ = lstm(get_extern_data("data"))
    assert isinstance(out, Layer)
    assert_equal(out.get_name(), "lstm")

    name_ctx.make_default_output(out)
    net_dict = name_ctx.make_net_dict()
    pprint(net_dict)

  assert "lstm" in net_dict
  lstm_layer_dict = net_dict["lstm"]
  assert_equal(lstm_layer_dict["class"], "rec")
  assert_equal(lstm_layer_dict["unit"], "nativelstm2")
  assert_equal(lstm_layer_dict["from"], "data:data")
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
      x, _ = self.lstm(x)
      x, _ = self.lstm(x)
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert "lstm" in net_dict
  assert "lstm_0" in net_dict
  assert_equal(net_dict["lstm_0"]["reuse_params"], "lstm")
  dummy_run_net(net_dict)


def test_explicit_root_ctx_sub():
  class Net(Module):
    """
    Net
    """
    # noinspection PyShadowingNames
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
    assert_equal(out.get_name(), "net")

    name_ctx.make_default_output(out)
    net_dict = name_ctx.make_net_dict()
    pprint(net_dict)

  assert "net" in net_dict
  sub_net_dict = net_dict["net"]["subnetwork"]
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
    # noinspection PyShadowingNames
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
    assert_equal(y.get_name(), "test_block")
    print(z)
    assert isinstance(z, LayerRef)
    assert_equal(z.get_name(), "test_block_0")

    net_dict = name_ctx.make_net_dict()
    pprint(net_dict)

  assert "test_block" in net_dict and "test_block_0" in net_dict
  assert_equal(net_dict["test_block_0"]["reuse_params"], "test_block")


def test_multiple_returns_depth_1():
  class _SubNet(Module):
    def __init__(self):
      super().__init__()
      self.linear = Linear(n_out=13, activation=None)

    def forward(self, x: LayerRef) -> Tuple[LayerRef, LayerRef]:
      """
      Forward
      """
      x = self.linear(x)
      return x, x

  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubNet()

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      out, add_out = self.sub(x)
      return out

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert net_dict["output"]["from"] == "sub/linear"
  assert net_dict["sub"]["subnetwork"]["linear"]["from"] == "base:data:data"


def test_multiple_returns_depth_2():
  class _SubSubNet(Module):
    def __init__(self):
      super().__init__()
      self.linear = Linear(n_out=13, activation=None)

    def forward(self, x: LayerRef) -> Tuple[LayerRef, LayerRef]:
      """
      Forward
      """
      x = self.linear(x)
      return x, x

  class _SubNet(Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubSubNet()

    def forward(self, x: LayerRef) -> Tuple[LayerRef, LayerRef]:
      """
      Forward
      """
      x, x_ = self.sub(x)
      return x, x_

  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubNet()

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      out, add_out = self.sub(x)
      return out

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert net_dict["output"]["from"] == "sub/sub/linear"
  assert net_dict["sub"]["subnetwork"]["output"]["from"] == "sub/linear"
  assert net_dict["sub"]["subnetwork"]["sub"]["subnetwork"]["linear"]["from"] == "base:base:data:data"


def test_from_call_variations():
  class _SubNet(Module):
    def __init__(self):
      super().__init__()
      self.linear = Linear(n_out=13, activation=None)
      self.linear2 = Linear(n_out=13, activation=None)

    def forward(self, x: LayerRef) -> Tuple[LayerRef, LayerRef]:
      """
      Forward
      """
      x = self.linear(x)
      x = self.linear2(x)
      return x, x

  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubNet()
      self.sub2 = _SubNet()

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      out, add_out = self.sub(x)
      out2, add_out2 = self.sub2(add_out)
      return out2

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert net_dict["output"]["from"] == "sub2/linear2"
  assert net_dict["sub"]["subnetwork"]["linear"]["from"] == "base:data:data"
  assert net_dict["sub"]["subnetwork"]["linear2"]["from"] == "linear"
  assert net_dict["sub2"]["subnetwork"]["linear"]["from"] == "base:sub/linear2"
  assert net_dict["sub2"]["subnetwork"]["linear2"]["from"] == "linear"


def test_from_call_variations2():
  class _SubNet(Module):
    def __init__(self):
      super().__init__()
      self.linear = Linear(n_out=13, activation=None)
      self.linear2 = Linear(n_out=13, activation=None)

    def forward(self, x: LayerRef) -> Tuple[LayerRef, LayerRef]:
      """
      Forward
      """
      x_ = self.linear(x)
      x = self.linear2(x_)
      return x, x_

  class _SubNet2(Module):
    def __init__(self):
      super().__init__()
      self.linear = Linear(n_out=13, activation=None)
      self.linear2 = Linear(n_out=13, activation=None)

    def forward(self, x: LayerRef, y: LayerRef) -> Tuple[LayerRef, LayerRef]:
      """
      Forward
      """
      assert_equal(x.get_name(), "base:sub/linear")
      assert_equal(y.get_name(), "base:linear")
      x_ = self.linear(x)
      x = self.linear2(x_)
      return x, x_

  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubNet()
      self.sub2 = _SubNet2()
      self.linear = Linear(n_out=13, activation=None)

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      out, add_out = self.sub(x)
      assert_equal(out.get_name(), "sub/linear2")
      assert_equal(add_out.get_name(), "sub/linear")
      lin = self.linear(out)
      assert_equal(lin.get_name(), "linear")
      out2, add_out2 = self.sub2(add_out, lin)
      assert_equal(out2.get_name(), "sub2/linear2")
      assert_equal(add_out2.get_name(), "sub2/linear")
      return out2

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)


def test_get_name_in_current_ctx():

  def make_ctx(parent: NameCtx = None, name: str = "", subnet=False):
    """
    helper that builds the different NameCtxs with correct attributes
    """
    if not parent:
      return NameCtx.new_root()
    ctx = NameCtx(parent=parent, name=name)
    if subnet:
      ctx.is_subnet_ctx = True
    return ctx

  root = make_ctx(name="root")
  sub_1 = make_ctx(parent=root, name="sub_1", subnet=True)
  same = make_ctx(parent=sub_1, name="same", subnet=True)
  child_1 = make_ctx(parent=same, name="child_1")
  sub_2 = make_ctx(parent=root, name="sub_2", subnet=True)
  child_2 = make_ctx(parent=sub_2, name="child_2")

  with root:
    with sub_1:
      assert_equal(same.get_name_in_current_ctx(), "same")
      assert_equal(child_1.get_name_in_current_ctx(), "same/child_1")
      assert_equal(sub_2.get_name_in_current_ctx(), "base:sub_2")
      assert_equal(child_2.get_name_in_current_ctx(), "base:sub_2/child_2")


def test_module_list():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.ls = ModuleList([Linear(i) for i in range(4)])

    def forward(self) -> LayerRef:
      """
      Forward
      """
      out = get_extern_data("data")
      for layer in self.ls:
        out = layer(out)
      return out

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)

  assert net_dict["linear"]["from"] == "data:data"
  assert net_dict["linear_0"]["from"] == "linear"
  assert net_dict["linear_1"]["from"] == "linear_0"
  assert net_dict["linear_2"]["from"] == "linear_1"
  assert net_dict["output"]["from"] == "linear_2"
