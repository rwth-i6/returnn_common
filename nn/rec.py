"""
Basic RNNs.
"""

from typing import Optional, Union, Dict, Tuple, Any
from .. import nn


class _Rec(nn.Module):
  def __init__(self, *, out_dim: nn.Dim, unit: str, unit_opts: Optional[Dict[str, Any]] = None):
    super().__init__()
    self.out_dim = out_dim
    self.unit = unit
    self.unit_opts = unit_opts

  def __call__(self, source: nn.LayerRef, *,
               in_dim: Optional[nn.Dim] = None,
               axis: nn.Dim,
               state: Optional[Union[nn.LayerRef, Dict[str, nn.LayerRef], nn.NotSpecified]] = nn.NotSpecified,
               initial_state: Optional[Union[nn.LayerRef, Dict[str, nn.LayerRef], nn.NotSpecified]] = nn.NotSpecified,
               ) -> Tuple[nn.Layer, nn.LayerState]:
    pass  # TODO


class LSTM(_Rec):
  """
  LSTM operating on a sequence. returns (output, final_state) tuple, where final_state is (h,c).
  """
  def __init__(self, out_dim: nn.Dim):
    super().__init__(out_dim=out_dim, unit="nativelstm2")


class ZoneoutLSTM(_Rec):
  """
  LSTM with zoneout operating on a sequence. returns (output, final_state) tuple, where final_state is (h,c).
  """
  def __init__(self, out_dim: nn.Dim, zoneout_factor_cell: float = 0., zoneout_factor_output: float = 0.):
    super().__init__(
      out_dim=out_dim, unit="zoneoutlstm",
      unit_opts={'zoneout_factor_cell': zoneout_factor_cell, 'zoneout_factor_output': zoneout_factor_output},
      )
