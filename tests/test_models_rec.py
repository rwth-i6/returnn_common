"""
Test rec module
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net

from returnn_common.models.layers import *
from returnn_common.models.base import *
from pprint import pprint


def test_rec_ff():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.rec_linear = Linear(n_out=13)

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      # https://github.com/rwth-i6/returnn_common/issues/16
      with Loop() as loop:
        x_ = loop.unstack(x, axis="T")
        loop.state.h = y_ = self.rec_linear([x_, loop.state.h])
        y = loop.stack(y_)
      return y

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  dummy_run_net(net_dict)
