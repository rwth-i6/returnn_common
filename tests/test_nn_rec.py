"""
Test nn.rec
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict
from nose.tools import assert_equal
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_simple_net_lstm():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(nn.FeatureDim("lstm-out", 13))

    @nn.scoped
    def __call__(self, x: nn.LayerRef, *, axis: nn.Dim) -> nn.LayerRef:
      """
      Forward
      """
      x, _ = self.lstm(x, axis=axis)
      return x

  config, net_dict = dummy_config_net_dict(_Net(), with_axis=True)
  assert "lstm" in net_dict
  extern_data_data_dim_tags_2_input_dim = config["extern_data_data_dim_tags_2_input_dim"]
  network_lstm_subnetwork_rec_out_dim_lstm_out_dim = config["network_lstm_subnetwork_rec_out_dim_lstm_out_dim"]
  lstm_subnet = net_dict["lstm"]["subnetwork"]
  param_input_weights_shape = lstm_subnet["param_W"]["shape"]
  param_rec_weights_shape = lstm_subnet["param_W_re"]["shape"]
  assert_equal(
    param_input_weights_shape,
    [extern_data_data_dim_tags_2_input_dim, 4 * network_lstm_subnetwork_rec_out_dim_lstm_out_dim])
  assert_equal(
    param_rec_weights_shape,
    [network_lstm_subnetwork_rec_out_dim_lstm_out_dim, 4 * network_lstm_subnetwork_rec_out_dim_lstm_out_dim])
  dummy_run_net(config)
