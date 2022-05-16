"""
Test nn.math_.
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_run_net_single_custom
from .returnn_helpers import config_net_dict_via_serialized
from .utils import assert_equal
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_op_broadcasting():
  # https://github.com/rwth-i6/returnn_common/issues/136
  nn.reset_default_root_name_ctx()
  x = nn.get_extern_data(nn.Data('x', dim_tags=[nn.FeatureDim('x', 3)]))
  y = nn.get_extern_data(nn.Data('y', dim_tags=[nn.FeatureDim('y', 5)]))
  try:
    (x + y)
  except Exception as exc:
    print("Got expected exception on (x + y):", exc)
    assert "require explicit allow_broadcast_all_sources=True" in str(exc)
  else:
    raise Exception("Expected exception on (x + y)")
  # This should work:
  nn.combine(x, y, kind="add", allow_broadcast_all_sources=True).mark_as_default_output()
  config = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  dummy_run_net_single_custom(config)


def test_split_glu():
  class _Net(nn.Module):
    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """forward"""
      a, b = nn.split(x, axis=axis, out_dims=[axis // 2, axis // 2])
      return a * nn.sigmoid(b)

  nn.reset_default_root_name_ctx()
  net = _Net()
  time_dim = nn.SpatialDim("time")
  feat_dim = nn.FeatureDim("feature", 6)
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, feat_dim]))
  out = net(data, axis=feat_dim)
  out.mark_as_default_output()

  config, net_dict = config_net_dict_via_serialized(nn.get_returnn_config().get_complete_py_code_str(net))
  batch_dim = nn.batch_dim
  time_dim = config["time_dim"]
  feature_dim = config["feature_dim"]
  assert_equal(
    net_dict,
    {
      'split': {
        'class': 'split',
        'from': 'data:data',
        'axis': feature_dim,
        'out_dims': [feature_dim // 2, feature_dim // 2],
        'out_shape': {batch_dim, time_dim, feature_dim}
      },
      'sigmoid': {
        'class': 'activation',
        'from': 'split/1',
        'activation': 'sigmoid',
        'out_shape': {batch_dim, time_dim, feature_dim // 2}
      },
      'mul': {
        'class': 'combine',
        'from': ['split/0', 'sigmoid'],
        'kind': 'mul',
        'out_shape': {batch_dim, time_dim, feature_dim // 2}
      },
      'output': {
        'class': 'copy',
        'from': 'mul',
        'out_shape': {batch_dim, time_dim, feature_dim // 2}
      }
    })
  dummy_run_net(config)


def test_cumcum():
  # https://github.com/rwth-i6/returnn_common/issues/133
  class _Net(nn.Module):
    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """
      Forward
      """
      return nn.cumsum(x, axis=axis)

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, with_axis=True)
  dummy_run_net(config, net=net)
