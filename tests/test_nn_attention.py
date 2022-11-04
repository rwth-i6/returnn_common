"""
Test nn.attention
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_default_in_dim, dummy_run_net_single_custom, \
  make_feed_dict
from pprint import pprint
import typing
import functools

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_self_attention():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.self_att = nn.SelfAttention(
        in_dim=dummy_default_in_dim, proj_dim=nn.FeatureDim("out", 5),
        key_dim_total=nn.FeatureDim("key-dim-total", 21),
        value_dim_total=nn.FeatureDim("value-dim-total", 33),
        num_heads=3)

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """forward"""
      return self.self_att(x, axis=axis)

  config, net_dict, net = dummy_config_net_dict(_Net, with_axis=True)
  pprint(net_dict)
  dummy_run_net(config, net=net)


def test_relative_positional_encoding():
  class _Net(nn.Module):
    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      x, _ = nn.relative_positional_encoding(axis, x.feature_dim)
      return x

  config, net_dict, net = dummy_config_net_dict(_Net, with_axis=True, in_dim=nn.FeatureDim("in", 12))
  pprint(net_dict)
  dummy_run_net(config, net=net)


def test_rel_pos_self_attention():
  class _Net(nn.Module):
    # noinspection PyShadowingNames
    def __init__(self, in_dim: nn.Dim):
      super().__init__()
      self.self_att = nn.RelPosSelfAttention(
        in_dim=in_dim, proj_dim=nn.FeatureDim("out", 5),
        key_dim_total=nn.FeatureDim("key-dim-total", 21),
        value_dim_total=nn.FeatureDim("value-dim-total", 33),
        num_heads=3)

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """forward"""
      return self.self_att(x, axis=axis)

  in_dim = nn.FeatureDim("in", 12)
  config, net_dict, net = dummy_config_net_dict(lambda: _Net(in_dim), with_axis=True, in_dim=in_dim)
  pprint(net_dict)
  dummy_run_net(config, net=net)


def test_rel_pos_self_attention_learnable():
  class _Net(nn.Module):
    # noinspection PyShadowingNames
    def __init__(self, in_dim: nn.Dim):
      super().__init__()
      self.self_att = nn.RelPosSelfAttention(
        in_dim=in_dim, proj_dim=nn.FeatureDim("out", 5),
        key_dim_total=nn.FeatureDim("key-dim-total", 21),
        value_dim_total=nn.FeatureDim("value-dim-total", 33),
        num_heads=3,
        # Shawn et al 2018 style, old RETURNN way.
        with_bias=False,
        with_linear_pos=False,
        with_pos_bias=False,
        learnable_pos_emb=True,
        learnable_pos_emb_clipping=3,
        separate_pos_emb_per_head=False,
      )

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """forward"""
      return self.self_att(x, axis=axis)

  in_dim = nn.FeatureDim("in", 12)
  config, net_dict, net = dummy_config_net_dict(lambda: _Net(in_dim), with_axis=True, in_dim=in_dim)
  pprint(net_dict)
  dummy_run_net(config, net=net, seq_len=3)  # ok
  dummy_run_net(config, net=net, seq_len=3)  # try again, to see that running again is ok.
  dummy_run_net(config, net=net, seq_len=1)  # ok
  dummy_run_net(config, net=net, seq_len=4)  # problem currently...


def test_learned_rel_pos_enc():
  class _Net(nn.Module):
    # noinspection PyShadowingNames
    def __init__(self, in_dim: nn.Dim):
      super().__init__()
      self.in_dim = in_dim
      self.self_att = nn.LearnedRelativePositionalEncoding(in_dim, clipping=3)

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      y, _ = self.self_att(axis)
      print("y:", y)
      return y + nn.reduce(x, axis=(axis, nn.batch_dim), mode="mean")

  nn.reset_default_root_name_ctx()
  net = _Net(in_dim=nn.FeatureDim("in", 12))
  time_dim = nn.SpatialDim("time")
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, net.in_dim]))
  out = net(data, axis=time_dim)
  out.mark_as_default_output()

  config_code_str = nn.get_returnn_config().get_complete_py_code_str(net)
  print(config_code_str)

  for seq_len in [1, 2, 3, 4, 5]:
    res = dummy_run_net_single_custom(
      config_code_str, make_feed_dict=functools.partial(make_feed_dict, n_time=seq_len))
    shape = res["layer:output"].shape
    print("res shape:", shape)
    assert shape == (2 * seq_len - 1, net.in_dim.dimension)
