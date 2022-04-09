"""
Test nn.transformer.
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


def test_nn_transformer_search():
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  input_dim = nn.FeatureDim("input", 10)
  target_dim = nn.FeatureDim("target", 7)
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))
  transformer = nn.Transformer(
    model_dim=input_dim, ff_dim=input_dim * 4,
    num_heads=2, num_encoder_layers=2, num_decoder_layers=2,
    target_dim=target_dim)
  _, _, out_labels, _ = transformer(
    data, source_spatial_axis=time_dim,
    target=nn.SearchFunc(
      beam_size=3,
      max_seq_len=nn.reduce(
        nn.length(data, axis=time_dim),
        mode="max", axis=nn.batch_dim)))
  out_labels.mark_as_default_output()

  config_code = nn.get_returnn_config_serialized(transformer)
  config, net_dict = config_net_dict_via_serialized(config_code)

  dec_self_att_layer_dict = (
    net_dict["loop"]["unit"]["decoder"]["subnetwork"]["layers.0"]["subnetwork"]["self_attn"]["subnetwork"]["k_accum"])
  assert dec_self_att_layer_dict["class"] == "cum_concat"
  assert "state" not in dec_self_att_layer_dict  # optimization

  collected_name_scopes = {}  # path -> name_scope

  def _collect_name_scope(path, x):
    if path and path[-1] == "name_scope":
      collected_name_scopes[path] = x

  nest.map_structure_with_tuple_paths(_collect_name_scope, net_dict)
  assert_equal(collected_name_scopes, {
    ('encoder', 'subnetwork', 'layers.0', 'name_scope'): 'layers/0',
    ('encoder', 'subnetwork', 'layers.1', 'name_scope'): 'layers/1',
    ('loop', 'name_scope'): '',
    ('loop', 'unit', 'decoder', 'subnetwork', 'layers.0', 'name_scope'): 'layers/0',
    ('loop', 'unit', 'decoder', 'subnetwork', 'layers.1', 'name_scope'): 'layers/1'})

  dummy_run_net(config, net=transformer)
