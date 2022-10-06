"""
Test nn.encoder.
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


def test_nn_blstm_cnn_specaug():
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  input_dim = nn.FeatureDim("input", 10)
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))

  from returnn_common.nn.encoder.blstm_cnn_specaug import BlstmCnnSpecAugEncoder
  net = BlstmCnnSpecAugEncoder(input_dim, nn.FeatureDim("out", 14), num_layers=2)
  out, _ = net(data, spatial_dim=time_dim)
  assert isinstance(out, nn.Tensor)
  assert out.feature_dim == net.out_dim
  out.mark_as_default_output()

  config_code = nn.get_returnn_config().get_complete_py_code_str(net)
  config, net_dict = config_net_dict_via_serialized(config_code)

  collected_var_names = set()
  for name, p in net.named_parameters():
    print(name, ":", p)
    collected_var_names.add(name.replace(".", "/"))

  assert_equal(
    collected_var_names,
    {'layers/0/bw/param_W',
     'layers/0/bw/param_W_re',
     'layers/0/bw/param_b',
     'layers/0/fw/param_W',
     'layers/0/fw/param_W_re',
     'layers/0/fw/param_b',
     'layers/1/bw/param_W',
     'layers/1/bw/param_W_re',
     'layers/1/bw/param_b',
     'layers/1/fw/param_W',
     'layers/1/fw/param_W_re',
     'layers/1/fw/param_b',
     'pre_conv_net/conv0/bias',
     'pre_conv_net/conv0/filter',
     'pre_conv_net/conv1/bias',
     'pre_conv_net/conv1/filter'})

  dummy_run_net(config, net=net)
