"""
Test nn.conformer.
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, config_net_dict_via_serialized
from nose.tools import assert_equal
from tensorflow.python.util import nest
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_nn_conformer():
  # This test needs a huge stack size currently, due to the way RETURNN layer construction works currently.
  # On RETURNN side, there is the option flat_net_construction to solve this,
  # however, it's experimental and also does not work for this case.
  # https://github.com/rwth-i6/returnn/issues/957
  # https://stackoverflow.com/a/16248113/133374
  import resource
  import sys
  try:
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
  except Exception as exc:
    print(f"resource.setrlimit {type(exc).__name__}: {exc}")
  sys.setrecursionlimit(10 ** 6)

  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  input_dim = nn.FeatureDim("input", 10)
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))
  conformer = nn.ConformerEncoder(
    out_dim=nn.FeatureDim("out", 14), ff_dim=nn.FeatureDim("ff", 17),
    num_heads=2, num_layers=2)
  out, _ = conformer(data, in_spatial_dim=time_dim)
  out.mark_as_default_output()

  config_code = nn.get_returnn_config().get_complete_py_code_str(conformer)
  config, net_dict = config_net_dict_via_serialized(config_code)

  collected_name_scopes = {}  # path -> name_scope

  def _collect_name_scope(path, x):
    if path and path[-1] == "name_scope":
      collected_name_scopes[path] = x

  nest.map_structure_with_tuple_paths(_collect_name_scope, net_dict)
  assert_equal(collected_name_scopes, {
    ('conv_subsample_layer', 'subnetwork', 'conv_layers.0', 'name_scope'): 'conv_layers/0',
    ('conv_subsample_layer', 'subnetwork', 'conv_layers.1', 'name_scope'): 'conv_layers/1'})

  dummy_run_net(config, net=conformer)
