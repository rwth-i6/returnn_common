"""
Basic RNNs.
"""

from ._generated_layers import _Rec


class Lstm(_Rec):
  """
  LSTM
  """
  def __init__(self, n_out: int, **kwargs):
    super().__init__(n_out=n_out, unit="nativelstm2", **kwargs)
