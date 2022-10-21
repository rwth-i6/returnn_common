"""
Test nn.masked_computation
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_default_in_dim

import typing
from .utils import assert_equal

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_masked_computation_lstm():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(dummy_default_in_dim, nn.FeatureDim("out", 13))

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """
      Forward
      """
      loop = nn.Loop(axis=axis)
      loop.state.lstm_out = nn.constant(value=0.0, shape=[nn.batch_dim, self.lstm.out_dim])
      loop.state.lstm = self.lstm.default_initial_state(batch_dims=x.batch_dims_ordered(remove=(axis, x.feature_dim)))
      with loop:
        x_ = loop.unstack(x)
        mask = nn.reduce(x_, mode="mean", axis=x_.feature_dim) >= 0.  # [B]
        with nn.MaskedComputation(mask):
          loop.state.lstm_out, loop.state.lstm = self.lstm(x_, state=loop.state.lstm, spatial_dim=nn.single_step_dim)
        y = loop.stack(loop.state.lstm_out)
      return y

  config, net_dict, net = dummy_config_net_dict(_Net, with_axis=True)
  engine = dummy_run_net(config, net=net)
  params = engine.network.get_params_list()
  print(params)
  assert len(params) == 3
  assert_equal(params[0].name, "lstm/param_W/param:0")
