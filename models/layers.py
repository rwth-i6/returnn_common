"""
Wrap RETURNN layers
"""

from ._generated_layers import *  # noqa
from .base import Module, Rec  # noqa


class Lstm(RecUnit):
  """
  LSTM
  """
  def __init__(self, rec_weight_dropout=0, unit_opts=NotSpecified, **kwargs):
    if rec_weight_dropout:
      if unit_opts is not NotSpecified and unit_opts:
        assert isinstance(unit_opts, dict)
        unit_opts = unit_opts.copy()
      else:
        unit_opts = {}
      assert "rec_weight_dropout" not in unit_opts
      unit_opts["rec_weight_dropout"] = rec_weight_dropout
    super(Lstm, self).__init__(
      unit="nativelstm2", unit_opts=unit_opts, **kwargs)
