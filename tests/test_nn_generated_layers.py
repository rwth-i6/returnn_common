"""
Test nn._generated_layers
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_run_net_single_custom
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_range_from_length():
  # https://github.com/rwth-i6/returnn_common/issues/134
  class _Net(nn.Module):
    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      durations = nn.cast(nn.reduce(x, axis=x.feature_dim, mode="sum"), dtype="int32")
      durations.verify_out_shape({nn.batch_dim, axis})
      t, _ = nn.range_from_length(durations)
      return t

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, with_axis=True)
  dummy_run_net(config, net=net)


def test_repeat_int():
  # https://github.com/rwth-i6/returnn_common/issues/162
  nn.reset_default_root_name_ctx()
  time = nn.SpatialDim("time")
  data = nn.get_extern_data(nn.Data('data', dim_tags=[nn.batch_dim, time]))
  rep, rep_dim = nn.repeat(data, repetitions=5, axis=time)
  rep.mark_as_default_output()
  config = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  dummy_run_net_single_custom(config)


def test_repeat_layer():
  # https://github.com/rwth-i6/returnn_common/issues/163
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  data = nn.get_extern_data(nn.Data('data', dim_tags=[nn.batch_dim, time_dim, nn.FeatureDim('F', 3)]))
  const = nn.constant(value=5, shape=[nn.batch_dim, time_dim])
  nn.repeat(data, repetitions=const, axis=time_dim)[0].mark_as_default_output()
  config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  dummy_run_net_single_custom(config_str)
