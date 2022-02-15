"""
Test nn.transformer.
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, config_net_dict_via_serialized
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_nn_transformer_search():
  with nn.NameCtx.new_root() as name_ctx:
    time_dim = nn.SpatialDim("time")
    input_dim = nn.FeatureDim("input", 10)
    data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))
    transformer = nn.Transformer(
      output_dim=input_dim, dim_ff=input_dim * 4,
      num_heads=2, num_encoder_layers=2, num_decoder_layers=2)
    out, _ = transformer(data, source_spatial_axis=time_dim, search=True, beam_size=3, eos_symbol=0, name=name_ctx)
    out.mark_as_default_output()

  config_code = name_ctx.get_returnn_config_serialized()
  config, net_dict = config_net_dict_via_serialized(config_code)
  dummy_run_net(config)
