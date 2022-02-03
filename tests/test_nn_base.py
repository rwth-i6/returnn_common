"""
Test the base nn functionality
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict
from pprint import pprint
from nose.tools import assert_equal
import typing
from typing import Tuple

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_simple_net_linear():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x) -> nn.TensorRef:
      """
      Forward
      """
      return self.linear(x)

  config, net_dict = dummy_config_net_dict(_Net())
  assert "linear" in net_dict
  pprint(config)
  dummy_run_net(config)


def test_simple_net_arithmetic():
  class _Net(nn.Module):
    @nn.scoped
    def __call__(self, x) -> nn.TensorRef:
      """
      Forward
      """
      x = 1. / x + x * 2.
      return x

  config, net_dict = dummy_config_net_dict(_Net())
  dummy_run_net(config)


def test_simple_net_share_params():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.dim = nn.FeatureDim("linear-out", 13)
      self.linear = nn.Linear(self.dim)
      self.linear2 = nn.Linear(self.dim)

    @nn.scoped
    def __call__(self, x) -> nn.TensorRef:
      """
      Forward
      """
      x = self.linear(x)
      x = self.linear2(x)
      x = self.linear2(x)
      return x

  config, net_dict = dummy_config_net_dict(_Net())
  assert "linear2" in net_dict
  assert "linear2_0" in net_dict
  assert_equal(net_dict["linear2_0"]["name_scope"], "linear2")
  dummy_run_net(config)


def test_explicit_root_ctx_sub():
  class _Net(nn.Module):
    # noinspection PyShadowingNames
    def __init__(self, out_dim: nn.Dim, dropout=0.1):
      super().__init__()
      self.linear = nn.Linear(out_dim)
      self.dropout = dropout

    @nn.scoped
    def __call__(self, x: nn.TensorRef) -> nn.TensorRef:
      """
      forward
      """
      x = nn.dropout(x, self.dropout, axis=x.feature_dim, name="pre")
      x = self.linear(x)
      return x

  with nn.NameCtx.new_root() as name_ctx:
    net = _Net(out_dim=nn.FeatureDim("linear-out", 13))
    out = net(nn.get_extern_data(nn.Data("data", shape=(None, 5))), name=name_ctx)
    assert isinstance(out, nn.Tensor)
    out.mark_as_default_output()

  config = name_ctx.get_returnn_config()
  net_dict = config["network"]
  pprint(net_dict)

  assert "linear" in net_dict
  assert "pre" in net_dict
  lin_layer_dict = net_dict["pre"]
  assert_equal(lin_layer_dict["class"], "dropout")
  assert_equal(lin_layer_dict["from"], "data:data")
  dummy_run_net(config)


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
    def __call__(self, x: nn.TensorRef) -> nn.TensorRef:
      """
      forward
      """
      x = nn.dropout(x, self.dropout, axis=x.feature_dim)
      x = self.linear(x)
      return x

  with nn.NameCtx.new_root() as name_ctx:
    test_block = TestBlock(out_dim=nn.FeatureDim("linear-out", 13))
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("input", 5)
    y = test_block(nn.get_extern_data(nn.Data("input1", dim_tags=[nn.batch_dim, time_dim, in_dim])), name=name_ctx)
    z = test_block(nn.get_extern_data(nn.Data("input2", dim_tags=[nn.batch_dim, time_dim, in_dim])))

    print(y)
    assert isinstance(y, nn.TensorRef)
    print(z)
    assert isinstance(z, nn.TensorRef)

  config = name_ctx.get_returnn_config()
  net_dict = config["network"]
  pprint(net_dict)

  assert "linear" in net_dict and "test_block" in net_dict
  assert_equal(net_dict["test_block"]["name_scope"], "")


def test_multiple_returns_depth_1():
  class _SubNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.TensorRef) -> Tuple[nn.TensorRef, nn.TensorRef]:
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
    def __call__(self, x) -> nn.TensorRef:
      """
      Forward
      """
      out, add_out = self.sub(x)
      return out

  net = _Net()
  config, net_dict = dummy_config_net_dict(net)
  pprint(net_dict)
  assert net_dict["output"]["from"] == "sub/linear"
  assert net_dict["sub"]["subnetwork"]["linear"]["subnetwork"]["dot"]["from"][0] == "base:base:data:data"
  dummy_run_net(config)


def test_multiple_returns_depth_2():
  class _SubSubNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.TensorRef) -> Tuple[nn.TensorRef, nn.TensorRef]:
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
    def __call__(self, x: nn.TensorRef) -> Tuple[nn.TensorRef, nn.TensorRef]:
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
    def __call__(self, x: nn.TensorRef) -> nn.TensorRef:
      """
      Forward
      """
      out, add_out = self.sub(x)
      return out

  net = _Net()
  config, net_dict = dummy_config_net_dict(net)
  pprint(net_dict)
  assert net_dict["output"]["from"] == "sub/sub/linear"
  assert net_dict["sub"]["subnetwork"]["output"]["from"] == "sub/linear"
  assert (
      net_dict
      ["sub"]["subnetwork"]
      ["sub"]["subnetwork"]
      ["linear"]["subnetwork"]
      ["dot"]["from"][0] == "base:base:base:data:data")
  dummy_run_net(config)


def test_from_call_variations():
  class _SubNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))
      self.linear2 = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.TensorRef) -> Tuple[nn.TensorRef, nn.TensorRef]:
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
    def __call__(self, x: nn.TensorRef) -> nn.TensorRef:
      """
      Forward
      """
      out, add_out = self.sub(x)
      out2, add_out2 = self.sub2(add_out)
      return out2

  net = _Net()
  config, net_dict = dummy_config_net_dict(net)
  pprint(net_dict)
  assert net_dict["output"]["from"] == "sub2/linear2"
  assert net_dict["sub"]["subnetwork"]["linear"]["subnetwork"]["dot"]["from"][0] == "base:base:data:data"
  assert net_dict["sub"]["subnetwork"]["linear2"]["subnetwork"]["dot"]["from"][0] == "base:linear"
  assert net_dict["sub2"]["subnetwork"]["linear"]["subnetwork"]["dot"]["from"][0] == "base:base:sub/linear2"
  assert net_dict["sub2"]["subnetwork"]["linear2"]["subnetwork"]["dot"]["from"][0] == "base:linear"
  dummy_run_net(config)


def test_from_call_variations2():
  class _SubNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(nn.FeatureDim("linear-out", 13))
      self.linear2 = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.TensorRef) -> Tuple[nn.TensorRef, nn.TensorRef]:
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
    def __call__(self, x: nn.TensorRef, y: nn.TensorRef) -> Tuple[nn.TensorRef, nn.TensorRef]:
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
    def __call__(self, x: nn.TensorRef) -> nn.TensorRef:
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
  config, net_dict = dummy_config_net_dict(net)
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


def test_deepcopy():
  import copy

  dims = [nn.FeatureDim(f"linear{i}-out", i + 3) for i in range(3)]
  layers = nn.Sequential(copy.deepcopy(nn.Linear(dim)) for dim in dims)
  config, net_dict = dummy_config_net_dict(layers)
  pprint(net_dict)
  assert "0" in net_dict
  assert_equal(net_dict["0"]["subnetwork"]["dot"]["from"][0], "base:data:data")
  dummy_run_net(config)
