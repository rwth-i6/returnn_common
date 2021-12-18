"""
Test layers
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net
from builtins import range as range_
from pprint import pprint
from nose.tools import assert_equal
import typing
from typing import Tuple

if typing.TYPE_CHECKING:
  from .. import __init__ as rc
  from .. import nn
else:
  import returnn_common as rc  # noqa
  from returnn_common import nn  # noqa


def test_simple_net_linear():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      return self.linear(x)

  net = _Net()
  config = nn.make_root_net_dict(
    net, nn.Data("data", dim_tags=[nn.batch_dim, nn.SpatialDim("time"), nn.FeatureDim("input", 13)]))
  net_dict = config["network"]
  pprint(net_dict)
  assert "linear" in net_dict
  dummy_run_net(config)


def test_simple_net_module_explicit_root_ctx():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x) -> nn.LayerRef:
      """
      Forward
      """
      return self.linear(x)

  net = _Net()

  with nn.NameCtx.new_root() as name_ctx:
    out = net(nn.get_extern_data("data"), name=name_ctx)
    assert isinstance(out, nn.Layer)
    name_ctx.make_default_output(out)
    net_dict = name_ctx.make_net().make_net_dict_raw()
    pprint(net_dict)

  assert "linear" in net_dict
  linear_layer_dict = net_dict["linear"]
  assert_equal(linear_layer_dict["class"], "linear")
  assert_equal(linear_layer_dict["from"], "data:data")
  dummy_run_net(net_dict)


def test_simple_net_rc():
  class _Net(rc.nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = rc.nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: rc.nn.LayerRef) -> rc.nn.LayerRef:
      """
      Forward
      """
      x = self.linear(x)
      return x

  net = _Net()
  net_dict = nn.make_root_net_dict(net, x="data")
  pprint(net_dict)
  assert "linear" in net_dict
  dummy_run_net(net_dict)


def test_simple_net_arithmetic():
  class _Net(nn.Module):
    @nn.scoped
    def __call__(self, x) -> nn.LayerRef:
      """
      Forward
      """
      x = 1. / x + x * 2.
      return x

  net = _Net()
  net_dict = nn.make_root_net_dict(net, x="data")
  pprint(net_dict)
  dummy_run_net(net_dict)


def test_simple_net_lstm():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(nn.FeatureDim("lstm-out", 13))

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      x, _ = self.lstm(x, axis=nn.any_spatial_dim)
      return x

  net = _Net()
  config = nn.make_root_net_dict(
    net, nn.Data("data", dim_tags=[nn.batch_dim, nn.SpatialDim("time"), nn.FeatureDim("input", 13)]))
  net_dict = config["network"]
  pprint(net_dict)
  assert "lstm" in net_dict
  dummy_run_net(config)


def test_simple_net_share_params():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))
      self.linear2 = nn.Linear(nn.FeatureDim("linear2-out", 13))

    @nn.scoped
    def __call__(self, x) -> nn.LayerRef:
      """
      Forward
      """
      x = self.linear(x)
      x = self.linear2(x)
      x = self.linear2(x)
      return x

  net = _Net()
  net_dict = nn.make_root_net_dict(net, x="data")
  pprint(net_dict)
  assert "linear2" in net_dict
  assert "linear2_0" in net_dict
  assert_equal(net_dict["linear2_0"]["name_scope"], "linear2")
  dummy_run_net(net_dict)


def test_explicit_root_ctx_sub():
  class _Net(nn.Module):
    # noinspection PyShadowingNames
    def __init__(self, out_dim: nn.Dim, dropout=0.1):
      super().__init__()
      self.linear = nn.Linear(out_dim)
      self.dropout = dropout

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      forward
      """
      x = nn.dropout(x, self.dropout, name="pre")
      x = self.linear(x)
      return x

  with nn.NameCtx.new_root() as name_ctx:
    net = _Net(out_dim=nn.FeatureDim("linear-out", 13))
    out = net(nn.get_extern_data("data"), name=name_ctx)
    assert isinstance(out, nn.Layer)

    name_ctx.make_default_output(out)
    net_dict = name_ctx.make_net().make_net_dict_raw()
    pprint(net_dict)

  assert "linear" in net_dict
  lin_layer_dict = net_dict["linear"]
  assert_equal(lin_layer_dict["class"], "linear")
  assert_equal(lin_layer_dict["from"], "pre")
  assert "pre" in net_dict
  lin_layer_dict = net_dict["pre"]
  assert_equal(lin_layer_dict["class"], "dropout")
  assert_equal(lin_layer_dict["from"], "data:data")
  dummy_run_net(net_dict)


def test_root_mod_call_twice():
  class TestBlock(nn.Module):
    """
    Test block
    """
    # noinspection PyShadowingNames
    def __init__(self, out_dim: nn.Dim, dropout=0.1):
      super().__init__()
      self.linear = nn.Linear(out_dim)
      self.dropout = dropout

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      forward
      """
      x = nn.dropout(x, self.dropout)
      x = self.linear(x)
      return x

  with nn.NameCtx.new_root() as name_ctx:
    test_block = TestBlock(nn.FeatureDim("linear-out", 13))
    y = test_block(nn.get_extern_data("input1"), name=name_ctx)
    z = test_block(nn.get_extern_data("input2"))

    print(y)
    assert isinstance(y, nn.LayerRef)
    print(z)
    assert isinstance(z, nn.LayerRef)

    net_dict = name_ctx.make_net().make_net_dict_raw()
    pprint(net_dict)

  assert "linear" in net_dict and "test_block" in net_dict
  assert_equal(net_dict["test_block"]["name_scope"], "")


def test_multiple_returns_depth_1():
  class _SubNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> Tuple[nn.LayerRef, nn.LayerRef]:
      """
      Forward
      """
      x = self.linear(x)
      return x, x

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubNet()

    @nn.scoped
    def __call__(self, x) -> nn.LayerRef:
      """
      Forward
      """
      out, add_out = self.sub(x)
      return out

  net = _Net()
  net_dict = nn.make_root_net_dict(net, x="data")
  pprint(net_dict)
  assert net_dict["output"]["from"] == "sub/linear"
  assert net_dict["sub"]["subnetwork"]["linear"]["from"] == "base:data:data"


def test_multiple_returns_depth_2():
  class _SubSubNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> Tuple[nn.LayerRef, nn.LayerRef]:
      """
      Forward
      """
      x = self.linear(x)
      return x, x

  class _SubNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubSubNet()

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> Tuple[nn.LayerRef, nn.LayerRef]:
      """
      Forward
      """
      x, x_ = self.sub(x)
      return x, x_

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubNet()

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      out, add_out = self.sub(x)
      return out

  net = _Net()
  net_dict = nn.make_root_net_dict(net, x="data")
  pprint(net_dict)
  assert net_dict["output"]["from"] == "sub/sub/linear"
  assert net_dict["sub"]["subnetwork"]["output"]["from"] == "sub/linear"
  assert net_dict["sub"]["subnetwork"]["sub"]["subnetwork"]["linear"]["from"] == "base:base:data:data"


def test_from_call_variations():
  class _SubNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))
      self.linear2 = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> Tuple[nn.LayerRef, nn.LayerRef]:
      """
      Forward
      """
      x = self.linear(x)
      x = self.linear2(x)
      return x, x

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubNet()
      self.sub2 = _SubNet()

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      out, add_out = self.sub(x)
      out2, add_out2 = self.sub2(add_out)
      return out2

  net = _Net()
  net_dict = nn.make_root_net_dict(net, x="data")
  pprint(net_dict)
  assert net_dict["output"]["from"] == "sub2/linear2"
  assert net_dict["sub"]["subnetwork"]["linear"]["from"] == "base:data:data"
  assert net_dict["sub"]["subnetwork"]["linear2"]["from"] == "linear"
  assert net_dict["sub2"]["subnetwork"]["linear"]["from"] == "base:sub/linear2"
  assert net_dict["sub2"]["subnetwork"]["linear2"]["from"] == "linear"


def test_from_call_variations2():
  class _SubNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))
      self.linear2 = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> Tuple[nn.LayerRef, nn.LayerRef]:
      """
      Forward
      """
      x_ = self.linear(x)
      x = self.linear2(x_)
      return x, x_

  class _SubNet2(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))
      self.linear2 = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.LayerRef, y: nn.LayerRef) -> Tuple[nn.LayerRef, nn.LayerRef]:
      """
      Forward
      """
      assert_equal(x.get_name_in_current_ctx(), "base:sub/linear")
      assert_equal(y.get_name_in_current_ctx(), "base:linear")
      x_ = self.linear(x)
      x = self.linear2(x_)
      return x, x_

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.sub = _SubNet()
      self.sub2 = _SubNet2()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      out, add_out = self.sub(x)
      assert_equal(out.get_name_in_current_ctx(), "sub/linear2")
      assert_equal(add_out.get_name_in_current_ctx(), "sub/linear")
      lin = self.linear(out)
      assert_equal(lin.get_name_in_current_ctx(), "linear")
      out2, add_out2 = self.sub2(add_out, lin)
      assert_equal(out2.get_name_in_current_ctx(), "sub2/linear2")
      assert_equal(add_out2.get_name_in_current_ctx(), "sub2/linear")
      return out2

  net = _Net()
  net_dict = nn.make_root_net_dict(net, x="data")
  pprint(net_dict)


def test_get_name_in_current_ctx():

  def make_ctx(parent: nn.NameCtx = None, name: str = "", subnet=False):
    """
    helper that builds the different NameCtxs with correct attributes
    """
    if not parent:
      return nn.NameCtx.new_root()
    ctx = nn.NameCtx(parent=parent, name=name)
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
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.base_dim = nn.FeatureDim("linear-out", 3)
      self.ls = nn.ModuleList([nn.Linear(self.base_dim + i) for i in range_(4)])

    @nn.scoped
    def __call__(self, out: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      for layer in self.ls:
        out = layer(out)
      return out

  net = _Net()
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)

  assert net_dict["ls.0"]["from"] == "data:data"
  assert net_dict["ls.1"]["from"] == "ls.0"
  assert net_dict["ls.2"]["from"] == "ls.1"
  assert net_dict["ls.3"]["from"] == "ls.2"
  assert net_dict["output"]["from"] == "ls.3"

  assert net_dict["ls.0"]["out_dim"] == net.base_dim
  assert net_dict["ls.1"]["out_dim"] == net.base_dim + 1
  assert_equal(
    net_dict,
    {'ls.0': {'class': 'linear', 'from': 'data:data', 'out_dim': net.base_dim, 'name_scope': 'ls/0'},
     'ls.1': {'class': 'linear', 'from': 'ls.0', 'out_dim': net.base_dim + 1, 'name_scope': 'ls/1'},
     'ls.2': {'class': 'linear', 'from': 'ls.1', 'out_dim': net.base_dim + 2, 'name_scope': 'ls/2'},
     'ls.3': {'class': 'linear', 'from': 'ls.2', 'out_dim': net.base_dim + 3, 'name_scope': 'ls/3'},
     'output': {'class': 'copy', 'from': 'ls.3'}})


def test_sequential_base_case():
  class _TestSequential(nn.Module):
    def __init__(self):
      super().__init__()
      self.seq = nn.Sequential(
        nn.Linear(nn.FeatureDim("feat1", 1)),
        nn.Linear(nn.FeatureDim("feat2", 2)),
        nn.Linear(nn.FeatureDim("feat3", 3)))

    @nn.scoped
    def __call__(self, data: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      seq = self.seq(data)
      return seq

  net = _TestSequential()
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)

  assert net_dict["seq"]["subnetwork"]["0"]["from"] == "base:data:data"
  assert net_dict["seq"]["subnetwork"]["1"]["from"] == "0"
  assert net_dict["seq"]["subnetwork"]["2"]["from"] == "1"
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
    def __call__(self, data: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      seq = self.seq(data)
      return seq

  net = _TestSequential()
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)

  assert net_dict["seq"]["subnetwork"]["one"]["from"] == "base:data:data"
  assert net_dict["seq"]["subnetwork"]["two"]["from"] == "one"
  assert net_dict["seq"]["subnetwork"]["three"]["from"] == "two"
  assert net_dict["seq"]["subnetwork"]["output"]["from"] == "three"
  assert net_dict["output"]["from"] == "seq"


def test_split_glu():
  class _Net(nn.Module):
    @nn.scoped
    def __call__(self, x: nn.LayerRef, axis: nn.Dim) -> nn.Layer:
      """forward"""
      a, b = nn.split(x, axis=axis, out_dims=[axis // 2, axis // 2])
      return a * nn.sigmoid(b)

  net = _Net()
  # TODO how to get extern data dims?
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)

  assert_equal(
    net_dict,
    {'mul': {'class': 'combine', 'from': ['split/0', 'sigmoid'], 'kind': 'mul'},
     'output': {'class': 'copy', 'from': 'mul'},
     'sigmoid': {'activation': 'sigmoid', 'class': 'activation', 'from': 'split/1'},
     'split': {'axis': 'F', 'class': 'split', 'from': 'data:data', 'num_splits': 2}})


def test_self_attention():
  time_dim = nn.SpatialDim("time")

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.self_att = nn.SelfAttention(
        key_dim_total=nn.FeatureDim("key-dim-total", 21),
        value_dim_total=nn.FeatureDim("value-dim-total", 33),
        num_heads=3)

    @nn.scoped
    def __call__(self, x: nn.LayerRef) -> nn.Layer:
      """forward"""
      return self.self_att(x, axis=time_dim)

  net = _Net()
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)


def test_deepcopy():
  import copy

  dims = [nn.FeatureDim(f"linear{i}-out", i + 3) for i in range(3)]
  layers = nn.Sequential(copy.deepcopy(nn.Linear(dim)) for dim in dims)
  net_dict = nn.make_root_net_dict(layers, "data")
  pprint(net_dict)
  assert_equal(
    net_dict,
    {'0': {'class': 'linear', 'from': 'data:data', 'out_dim': dims[0]},
     '1': {'class': 'linear', 'from': '0', 'out_dim': dims[1]},
     '2': {'class': 'linear', 'from': '1', 'out_dim': dims[2]},
     'output': {'class': 'copy', 'from': '2'}})
