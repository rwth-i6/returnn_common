"""
Test nn.conformer.
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, config_net_dict_via_serialized
from .utils import assert_equal
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
    input_dim, nn.FeatureDim("out", 14), ff_dim=nn.FeatureDim("ff", 17),
    input_layer=nn.ConformerConvSubsample(
      input_dim,
      out_dims=[nn.FeatureDim("conv1", 32), nn.FeatureDim("conv2", 64)],
      filter_sizes=[(3, 3), (3, 3)], pool_sizes=[(2, 1), (2, 1)]
    ),
    num_heads=2, num_layers=2,
  )
  out, _ = conformer(data, in_spatial_dim=time_dim)
  out.mark_as_default_output()

  config_code = nn.get_returnn_config().get_complete_py_code_str(conformer)
  config, net_dict = config_net_dict_via_serialized(config_code)

  collected_var_names = set()
  for name, p in conformer.named_parameters():
    print(name, ":", p)
    collected_var_names.add(name.replace(".", "/"))

  assert_equal(
    collected_var_names,
    {'input_layer/conv_layers/0/bias',
     'input_layer/conv_layers/0/filter',
     'input_layer/conv_layers/1/bias',
     'input_layer/conv_layers/1/filter',
     'input_projection/weight',
     'layers/0/conv_block/depthwise_conv/bias',
     'layers/0/conv_block/depthwise_conv/filter',
     'layers/0/conv_block/norm/beta',
     'layers/0/conv_block/norm/gamma',
     'layers/0/conv_block/norm/running_mean',
     'layers/0/conv_block/norm/running_variance',
     'layers/0/conv_block/positionwise_conv1/bias',
     'layers/0/conv_block/positionwise_conv1/weight',
     'layers/0/conv_block/positionwise_conv2/bias',
     'layers/0/conv_block/positionwise_conv2/weight',
     'layers/0/conv_layer_norm/bias',
     'layers/0/conv_layer_norm/scale',
     'layers/0/ffn1/linear_ff/bias',
     'layers/0/ffn1/linear_ff/weight',
     'layers/0/ffn1/linear_out/bias',
     'layers/0/ffn1/linear_out/weight',
     'layers/0/ffn1_layer_norm/bias',
     'layers/0/ffn1_layer_norm/scale',
     'layers/0/ffn2/linear_ff/bias',
     'layers/0/ffn2/linear_ff/weight',
     'layers/0/ffn2/linear_out/bias',
     'layers/0/ffn2/linear_out/weight',
     'layers/0/ffn2_layer_norm/bias',
     'layers/0/ffn2_layer_norm/scale',
     'layers/0/final_layer_norm/bias',
     'layers/0/final_layer_norm/scale',
     'layers/0/self_att/proj/weight',
     'layers/0/self_att/proj/bias',
     'layers/0/self_att/qkv/bias',
     'layers/0/self_att/qkv/weight',
     'layers/0/self_att/pos_bias_u',
     'layers/0/self_att/pos_bias_v',
     'layers/0/self_att/linear_pos/weight',
     'layers/0/self_att_layer_norm/bias',
     'layers/0/self_att_layer_norm/scale',
     'layers/1/conv_block/depthwise_conv/bias',
     'layers/1/conv_block/depthwise_conv/filter',
     'layers/1/conv_block/norm/beta',
     'layers/1/conv_block/norm/gamma',
     'layers/1/conv_block/norm/running_mean',
     'layers/1/conv_block/norm/running_variance',
     'layers/1/conv_block/positionwise_conv1/bias',
     'layers/1/conv_block/positionwise_conv1/weight',
     'layers/1/conv_block/positionwise_conv2/bias',
     'layers/1/conv_block/positionwise_conv2/weight',
     'layers/1/conv_layer_norm/bias',
     'layers/1/conv_layer_norm/scale',
     'layers/1/ffn1/linear_ff/bias',
     'layers/1/ffn1/linear_ff/weight',
     'layers/1/ffn1/linear_out/bias',
     'layers/1/ffn1/linear_out/weight',
     'layers/1/ffn1_layer_norm/bias',
     'layers/1/ffn1_layer_norm/scale',
     'layers/1/ffn2/linear_ff/bias',
     'layers/1/ffn2/linear_ff/weight',
     'layers/1/ffn2/linear_out/bias',
     'layers/1/ffn2/linear_out/weight',
     'layers/1/ffn2_layer_norm/bias',
     'layers/1/ffn2_layer_norm/scale',
     'layers/1/final_layer_norm/bias',
     'layers/1/final_layer_norm/scale',
     'layers/1/self_att/proj/weight',
     'layers/1/self_att/proj/bias',
     'layers/1/self_att/qkv/bias',
     'layers/1/self_att/qkv/weight',
     'layers/1/self_att/pos_bias_u',
     'layers/1/self_att/pos_bias_v',
     'layers/1/self_att/linear_pos/weight',
     'layers/1/self_att_layer_norm/bias',
     'layers/1/self_att_layer_norm/scale',
    })

  dummy_run_net(config, net=conformer)
