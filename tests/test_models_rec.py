"""
Test rec module
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net

from returnn_common.models.layers import *
from returnn_common.models.base import get_extern_data, LayerRef
from pprint import pprint


def test_rec_ff():
  class _MyRec(Rec):
    def __init__(self):
      super(_MyRec, self).__init__()
      self.lin = Linear(n_out=13, activation=None)

    def step(self, x: LayerRef) -> LayerRef:
      """step"""
      x = unroll(x)  # TODO ...
      return self.lin(x)

  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.rec = _MyRec()

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x = self.rec(x)
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  dummy_run_net(net_dict)
