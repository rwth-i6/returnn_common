"""
Test nn.math_.
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, config_net_dict_via_serialized
from nose.tools import assert_equal
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_split_glu():
  class _Net(nn.Module):
    @nn.scoped
    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """forward"""
      a, b = nn.split(x, axis=axis, out_dims=[axis // 2, axis // 2])
      return a * nn.sigmoid(b)

  with nn.NameCtx.new_root() as name_ctx:
    net = _Net()
    time_dim = nn.SpatialDim("time")
    feat_dim = nn.FeatureDim("feature", 6)
    data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, feat_dim]))
    out = net(data, axis=feat_dim, name=name_ctx)
    out.mark_as_default_output()

  config, net_dict = config_net_dict_via_serialized(name_ctx.get_returnn_config_serialized())
  batch_dim = nn.batch_dim
  extern_data_data_dim_tags_1_time_dim = config["extern_data_data_dim_tags_1_time_dim"]
  extern_data_data_dim_tags_2_feature_dim = config["extern_data_data_dim_tags_2_feature_dim"]
  assert_equal(
    net_dict,
    {
      'split': {
        'class': 'split',
        'from': 'data:data',
        'axis': extern_data_data_dim_tags_2_feature_dim,
        'out_dims': [
          extern_data_data_dim_tags_2_feature_dim // 2,
          extern_data_data_dim_tags_2_feature_dim // 2
        ],
        'out_shape': {batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_data_dim_tags_2_feature_dim}
      },
      'sigmoid': {
        'class': 'activation',
        'from': 'split/1',
        'activation': 'sigmoid',
        'out_shape': {batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_data_dim_tags_2_feature_dim // 2}
      },
      'mul': {
        'class': 'combine',
        'from': ['split/0', 'sigmoid'],
        'kind': 'mul',
        'out_shape': {batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_data_dim_tags_2_feature_dim // 2}
      },
      'output': {
        'class': 'copy',
        'from': 'mul',
        'out_shape': {batch_dim, extern_data_data_dim_tags_1_time_dim, extern_data_data_dim_tags_2_feature_dim // 2}
      }
    })
  dummy_run_net(config)
