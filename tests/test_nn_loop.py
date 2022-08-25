"""
Test nn.loop
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_run_net_single_custom

import typing
from .utils import assert_equal
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

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
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

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, with_axis=True)
  engine = dummy_run_net(config, net=net)
  params = engine.network.get_params_list()
  print(params)
  assert len(params) == 2
  assert_equal(params[0].name, "rec_linear/bias/param:0")


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

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
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

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, with_axis=True)
  engine = dummy_run_net(config, net=net)
  params = engine.network.get_params_list()
  print(params)
  assert len(params) == 3
  assert_equal(params[1].name, "lstm/param_W_re/param:0")


def test_rec_simple_iter():
  class _Net(nn.Module):
    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """
      Forward
      """
      # https://github.com/rwth-i6/returnn_common/issues/16
      loop = nn.Loop(max_seq_len=nn.constant(value=10))
      loop.state.i = nn.zeros([nn.batch_dim])
      with loop:
        loop.state.i = loop.state.i + 1.
        loop.end(loop.state.i >= 5., include_eos=True)
        y = loop.stack(loop.state.i * nn.reduce(x, mode="mean", axis=axis))
      return y

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, with_axis=True)
  dummy_run_net(config, net=net)


def test_rec_hidden():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(nn.FeatureDim("lstm-out", 13))

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """
      Forward
      """
      y, state = self.lstm(x, axis=axis)
      res = nn.concat(
        (y, self.lstm.out_dim), (state.h, self.lstm.out_dim), (state.c, self.lstm.out_dim), allow_broadcast=True)
      return res

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, with_axis=True)
  dummy_run_net(config, net=net)


def test_rec_hidden_initial():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.out_dim = nn.FeatureDim("out", 13)
      self.linear = nn.Linear(self.out_dim)
      self.lstm = nn.LSTM(self.out_dim)

    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      """
      Forward
      """
      y = self.linear(x)
      state = None
      for _ in range_(3):
        y, state = self.lstm(y, state=state, axis=axis)
      return y

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, with_axis=True)
  dummy_run_net(config, net=net)


def test_loop_axis_indices():
  class _Net(nn.Module):
    def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
      loop = nn.Loop(axis=axis)
      indices = nn.range_over_dim(axis)
      loop.state.x = nn.zeros([nn.batch_dim, x.feature_dim], dtype=indices.dtype)
      with loop:
        i = loop.unstack(indices)
        loop.state.x = loop.state.x + i
        loop.stack(i)  # loop needs some dummy output currently...
      return loop.state.x

  net = _Net()
  config, net_dict = dummy_config_net_dict(net=net, with_axis=True)
  dummy_run_net(config, net=net)


def test_loop_full_seq_last():
  nn.reset_default_root_name_ctx()

  feat_dim = nn.FeatureDim("feat", 5)
  time_dim = nn.SpatialDim("time")
  x = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, feat_dim]))

  # own name scope via function, this triggers the bug of need_last in sub layer inside rec loop
  def _relu(_x: nn.Tensor) -> nn.Tensor:
    _y = nn.where(_x < 0., 0., _x)
    # Check names. Note that these potentially might change at some later time, and then we need to update this here.
    # However, what we want to test is that the name is reasonable. Specifically:
    # - "test_loop_full_seq_last" should not be part of the name scope.
    assert _x.get_abs_name() == 'loop/prev:state.x'
    # 'where' twice due to current implementation of nn.where and nn.make_layer.
    # However, this is optimized (flattened) later, i.e. does not end up like that in the final net dict.
    assert _y.get_abs_name() == 'loop/where/where'
    return _y

  # feature mask
  mask_axis = x.feature_dim
  broadcast_axis = time_dim

  batch_dims = list(x.shape_ordered)
  batch_dims.remove(mask_axis)
  batch_dims.remove(broadcast_axis)
  num = nn.random_uniform(batch_dims, minval=1, maxval=3, dtype="int32")
  _, indices, k_dim = nn.top_k(
    nn.random_uniform(batch_dims + [mask_axis], minval=0., maxval=1.),
    axis=mask_axis,
    k=num if isinstance(num, int) else nn.reduce(num, mode="max", axis=num.shape_ordered))
  # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
  loop = nn.Loop(axis=k_dim)
  k_dim_indices = nn.range_over_dim(k_dim)
  loop.state.x = x
  with loop:
    i = loop.unstack(k_dim_indices)
    loop.state.x = _relu(loop.state.x)
    loop.stack(i)  # loop needs some dummy output currently...
  x = loop.state.x
  print(x)
  x.mark_as_default_output()

  code_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  code_str += "debug_runtime_sanity_checks = True\n\n"
  dummy_run_net_single_custom(code_str)
