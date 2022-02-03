"""
Test nn.loop
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict

import typing
from nose.tools import assert_equal
from builtins import range as range_

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_rec_ff():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.rec_linear = nn.Linear(nn.FeatureDim("linear-out", 13))

    @nn.scoped
    def __call__(self, x: nn.TensorRef, *, axis: nn.Dim) -> nn.TensorRef:
      """
      Forward
      """
      # https://github.com/rwth-i6/returnn_common/issues/16
      loop = nn.Loop(axis=axis)
      loop.state.h = nn.zeros([nn.batch_dim, self.rec_linear.out_dim])
      with loop:
        x_ = loop.unstack(x)
        loop.state.h = self.rec_linear(nn.concat((x_, x_.feature_dim), (loop.state.h, self.rec_linear.out_dim)))
        y = loop.stack(loop.state.h)
      return y

  config, net_dict = dummy_config_net_dict(net=_Net(), with_axis=True)
  dummy_run_net(config)


def test_lstm_default_name():
  assert_equal(nn.LSTM(nn.FeatureDim("out", 3)).get_default_name(), "lstm")
  # no LSTMStep anymore, so nothing really to test here.
  # https://github.com/rwth-i6/returnn_common/issues/81
  # assert_equal(nn.LSTMStep(nn.FeatureDim("out", 3)).get_default_name(), "lstm")


def test_rec_inner_lstm():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(nn.FeatureDim("out", 13))

    @nn.scoped
    def __call__(self, x: nn.TensorRef, *, axis: nn.Dim) -> nn.TensorRef:
      """
      Forward
      """
      loop = nn.Loop(axis=axis)
      loop.state.lstm = self.lstm.default_initial_state()
      with loop:
        x_ = loop.unstack(x)
        y_, loop.state.lstm = self.lstm(x_, state=loop.state.lstm, axis=nn.single_step_dim)
        y = loop.stack(y_)
      return y

  config, net_dict = dummy_config_net_dict(net=_Net(), with_axis=True)
  dummy_run_net(config)


def test_rec_simple_iter():
  class _Net(nn.Module):
    @nn.scoped
    def __call__(self, x: nn.TensorRef, *, axis: nn.Dim) -> nn.TensorRef:
      """
      Forward
      """
      # https://github.com/rwth-i6/returnn_common/issues/16
      loop = nn.Loop(max_seq_len=10)
      loop.state.i = nn.zeros([nn.batch_dim])
      with loop:
        loop.state.i = loop.state.i + 1.
        loop.end(loop.state.i >= 5., include_eos=True)
        y = loop.stack(loop.state.i * nn.reduce(x, mode="mean", axis=axis))
      return y

  config, net_dict = dummy_config_net_dict(net=_Net(), with_axis=True)
  dummy_run_net(config)


def test_rec_hidden():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(nn.FeatureDim("lstm-out", 13))

    @nn.scoped
    def __call__(self, x: nn.TensorRef, *, axis: nn.Dim) -> nn.TensorRef:
      """
      Forward
      """
      y, state = self.lstm(x, axis=axis)
      res = nn.concat(
        (y, self.lstm.out_dim), (state.h, self.lstm.out_dim), (state.c, self.lstm.out_dim), allow_broadcast=True)
      return res

  config, net_dict = dummy_config_net_dict(net=_Net(), with_axis=True)
  dummy_run_net(config)


def test_rec_hidden_initial():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.out_dim = nn.FeatureDim("out", 13)
      self.linear = nn.Linear(self.out_dim)
      self.lstm = nn.LSTM(self.out_dim)

    @nn.scoped
    def __call__(self, x: nn.TensorRef, *, axis: nn.Dim) -> nn.TensorRef:
      """
      Forward
      """
      y = self.linear(x)
      state = None
      for _ in range_(3):
        y, state = self.lstm(y, state=state, axis=axis)
      return y

  config, net_dict = dummy_config_net_dict(net=_Net(), with_axis=True)
  dummy_run_net(config)
