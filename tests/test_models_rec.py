"""
Test rec module
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net

import typing
from pprint import pprint
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
      self.rec_linear = nn.Linear(n_out=13)

    @nn.scoped_method
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      # https://github.com/rwth-i6/returnn_common/issues/16
      with nn.Loop() as loop:
        x_ = loop.unstack(x, axis="T", declare_rec_time=True)
        loop.state.h = nn.State(initial=0)  # TODO proper initial...
        loop.state.h = self.rec_linear(nn.concat((x_, "F"), (loop.state.h, "F")))
        y = loop.stack(loop.state.h)
      return y

  net = _Net()
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)
  assert_equal(
    net_dict,
    {'loop': {'class': 'rec',
              'from': [],
              'unit': {'concat': {'class': 'concat',
                                  'from': (('rec_unstack', 'F'),
                                           ('prev:state.h', 'F'))},
                       'output': {'class': 'copy', 'from': 'state.h'},
                       'rec_unstack': {'axis': 'T', 'declare_rec_time': True,
                                       'class': 'rec_unstack',
                                       'from': 'base:data:data'},
                       'state.h': {'class': 'linear',
                                   'from': 'concat',
                                   'n_out': 13}}},
     'output': {'class': 'copy', 'from': 'loop/output'}})
  dummy_run_net(net_dict)


def test_lstm_default_name():
  assert_equal(nn.LSTM(3).get_default_name(), "lstm")
  assert_equal(nn.LSTMStep(3).get_default_name(), "lstm")


def test_rec_inner_lstm():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTMStep(13)

    @nn.scoped_method
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      with nn.Loop() as loop:
        x_ = loop.unstack(x, axis="T", declare_rec_time=True)
        loop.state.lstm = nn.State(initial=self.lstm.default_initial_state())
        y_, loop.state.lstm = self.lstm(x_, state=loop.state.lstm)
        y = loop.stack(y_)
      return y

  net = _Net()
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)
  dummy_run_net(net_dict)


def test_rec_simple_iter():
  class _Net(nn.Module):
    @nn.scoped_method
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      # https://github.com/rwth-i6/returnn_common/issues/16
      with nn.Loop(max_seq_len=10) as loop:
        loop.state.i = nn.State(initial=0.)
        loop.state.i = loop.state.i + 1.
        loop.end(loop.state.i >= 5., include_eos=True)
        y = loop.stack(loop.state.i * nn.reduce(x, mode="mean", axis="T"))
      return y

  net = _Net()
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)
  dummy_run_net(net_dict)


def test_rec_hidden():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.lstm = nn.LSTM(n_out=13)

    @nn.scoped_method
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      y, state = self.lstm(x)
      y_ = nn.reduce(y, mode="mean", axis="T")  # TODO just because concat allow_broadcast=True does not work yet...
      res = nn.concat((y_, "F"), (state.h, "F"), (state.c, "F"))
      return res

  net = _Net()
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)
  dummy_run_net(net_dict)


def test_rec_hidden_initial():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(13)
      self.lstm = nn.LSTM(13)

    @nn.scoped_method
    def __call__(self, x: nn.LayerRef) -> nn.LayerRef:
      """
      Forward
      """
      y = self.linear(x)
      state = None
      for _ in range_(3):
        y, state = self.lstm(y, initial_state=state)
      return y

  net = _Net()
  net_dict = nn.make_root_net_dict(net, "data")
  pprint(net_dict)
  dummy_run_net(net_dict)
