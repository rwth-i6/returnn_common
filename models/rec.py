"""
Basic RNNs.
"""

from ._generated_layers import Rec


class Lstm(Rec):
  """
  LSTM
  """
  def __init__(self, *, rec_weight_dropout=0, rec_weight_dropout_shape=None, **kwargs):
    assert "unit_opts" not in kwargs, "we handle that here"
    unit_opts = {}
    if rec_weight_dropout:
      unit_opts["rec_weight_dropout"] = rec_weight_dropout
    if rec_weight_dropout_shape:
      unit_opts["rec_weight_dropout_shape"] = rec_weight_dropout_shape
    super(Lstm, self).__init__(
      unit="nativelstm2", unit_opts=unit_opts, **kwargs)
