"""
Test rec module
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net

from returnn_common.nn import *
from pprint import pprint


def test_rec_ff():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.rec_linear = Linear(n_out=13)

    def forward(self, x: LayerRef) -> LayerRef:
      """
      Forward
      """
      # https://github.com/rwth-i6/returnn_common/issues/16
      with Loop() as loop:
        x_ = loop.unstack(x, axis="T")
        loop.state.h = self.rec_linear(concat(x_, loop.state.h))
        y = loop.stack(loop.state.h)
      return y

  net = _Net()
  net_dict = make_root_net_dict(net, "data")
  pprint(net_dict)
  dummy_run_net(net_dict)


def test_rec_simple_iter():
  class _Net(Module):
    def forward(self, x: LayerRef) -> LayerRef:
      """
      Forward
      """
      # https://github.com/rwth-i6/returnn_common/issues/16
      with Loop(max_seq_len=10) as loop:
        loop.state.i = State()
        loop.state.i = loop.state.i + 1.
        loop.end(loop.state.i >= 5.)
        y = loop.stack(loop.state.i * reduce(x, mode="mean", axis="T"))
      return y

  net = _Net()
  net_dict = make_root_net_dict(net, "data")
  pprint(net_dict)
  dummy_run_net(net_dict)


def test_rec_hidden():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.lstm = Lstm(n_out=13)

    def forward(self, x: LayerRef) -> LayerRef:
      """
      Forward
      """
      y, state = self.lstm(x)
      return concat(y, state.h, state.c)

  net = _Net()
  net_dict = make_root_net_dict(net, "data")
  pprint(net_dict)
  dummy_run_net(net_dict)


def test_rec_hidden_initial():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.linear = Linear(13)
      self.lstm = Lstm(13)

    def forward(self, x: LayerRef) -> LayerRef:
      """
      Forward
      """
      y = self.linear(x)
      state = None
      for i in range(3):
        y, state = self.lstm(y, state=state)
      return y

  net = _Net()
  net_dict = make_root_net_dict(net, "data")
  pprint(net_dict)
  dummy_run_net(net_dict)
