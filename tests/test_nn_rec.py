"""
Test nn.rec
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_default_in_dim
from .utils import assert_equal
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_simple_net_lstm():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(dummy_default_in_dim, nn.FeatureDim("lstm-out", 13))

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """
      Forward
      """
      x, _ = self.lstm(x, spatial_dim=axis)
      return x

  config, net_dict, net = dummy_config_net_dict(_Net, with_axis=True)
  assert "lstm" in net_dict
  input_dim = config["input_dim"]
  lstm_out_dim = config["lstm_out_dim"]
  lstm_subnet = net_dict["lstm"]["subnetwork"]
  param_input_weights_shape = lstm_subnet["param_W"]["shape"]
  param_rec_weights_shape = lstm_subnet["param_W_re"]["shape"]
  assert_equal(param_input_weights_shape, [input_dim, 4 * lstm_out_dim])
  assert_equal(param_rec_weights_shape, [lstm_out_dim, 4 * lstm_out_dim])
  dummy_run_net(config, net=net)


def test_simple_net_zoneout_lstm():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.ZoneoutLSTM(dummy_default_in_dim, nn.FeatureDim("lstm-out", 13))

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """
      Forward
      """
      x, _ = self.lstm(x, spatial_dim=axis)
      return x

  config, net_dict, net = dummy_config_net_dict(_Net, with_axis=True)
  dummy_run_net(config, net=net)
