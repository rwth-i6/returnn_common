"""
Test nn.array_.
"""

from __future__ import annotations
from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net_single_custom
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_concat():
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("in", 3)
  x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)
  x = nn.get_extern_data(x)
  out, dim = nn.concat((x, x.feature_dim), (x * 2., x.feature_dim))
  assert dim == 2 * x.feature_dim
  out.mark_as_default_output()
  out.mark_as_loss("y")
  config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  dummy_run_net_single_custom(config_str, eval_flag=True)


def test_concat_features():
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("in", 3)
  x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)
  x = nn.get_extern_data(x)
  out = nn.concat_features(x, x * 2.)
  out.mark_as_default_output()
  out.mark_as_loss("y")
  config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  dummy_run_net_single_custom(config_str, eval_flag=True)
