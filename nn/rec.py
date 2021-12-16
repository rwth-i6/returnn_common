"""
Basic RNNs.
"""

from .. import nn
from ._generated_layers import _Rec


class LSTM(_Rec):
  """
  LSTM operating on a sequence. returns (output, final_state) tuple, where final_state is (h,c).
  """
  def __init__(self, out_dim: nn.Dim, **kwargs):
    super().__init__(out_dim=out_dim, unit="nativelstm2", **kwargs)


class ZoneoutLSTM(_Rec):
  """
  LSTM with zoneout operating on a sequence. returns (output, final_state) tuple, where final_state is (h,c).
  """
  def __init__(self, n_out: int, zoneout_factor_cell: float = 0., zoneout_factor_output: float = 0., **kwargs):
    super().__init__(
      n_out=n_out, unit="zoneoutlstm",
      unit_opts={'zoneout_factor_cell': zoneout_factor_cell, 'zoneout_factor_output': zoneout_factor_output},
      **kwargs)
