"""
Basic RNNs.
"""

from typing import Optional
from .. import nn
from ._generated_layers import _Rec


class LSTM(_Rec):
  """
  LSTM operating on a sequence. returns (output, final_state) tuple, where final_state is (h,c).
  """
  def __init__(self, out_dim: nn.Dim, **kwargs):
    super().__init__(out_dim=out_dim, unit="nativelstm2", **kwargs)

  # noinspection PyMethodOverriding
  def make_layer_dict(
        self, source: nn.LayerRef, *, axis: nn.Dim, initial_state: Optional[nn.LayerState] = None) -> nn.LayerDictRaw:
    """make layer"""
    return super().make_layer_dict(source, axis=axis, initial_state=initial_state)


class LSTMStep(_Rec):
  """
  LSTM operating one step. returns (output, state) tuple, where state is (h,c).
  """
  default_name = "lstm"  # make consistent to LSTM

  def __init__(self, out_dim: nn.Dim, **kwargs):
    super().__init__(out_dim=out_dim, unit="nativelstm2", **kwargs)

  # noinspection PyMethodOverriding
  def make_layer_dict(
        self, source: nn.LayerRef, *, state: nn.LayerState) -> nn.LayerDictRaw:
    """make layer"""
    # TODO specify per-step, how? this should also work without rec loop, when there is no time dim.
    #  https://github.com/rwth-i6/returnn/issues/847
    return super().make_layer_dict(source, state=state, axis=None)


class ZoneoutLSTM(_Rec):
  """
  LSTM with zoneout operating on a sequence. returns (output, final_state) tuple, where final_state is (h,c).
  """
  def __init__(self, n_out: int, zoneout_factor_cell: int = 0., zoneout_factor_output: int = 0., **kwargs):
    super().__init__(
      unit_opts={'zoneout_factor_cell': zoneout_factor_cell, 'zoneout_factor_output': zoneout_factor_output},
      n_out=n_out, unit="zoneoutlstm", **kwargs)

  # noinspection PyMethodOverriding
  def make_layer_dict(
        self, source: nn.LayerRef, *, initial_state: Optional[nn.LayerState] = None) -> nn.LayerDictRaw:
    """make layer"""
    return super().make_layer_dict(source, initial_state=initial_state)


class ZoneoutLSTMStep(_Rec):
  """
  LSTM with zoneout operating one step. returns (output, state) tuple, where state is (h,c).
  """
  default_name = "zoneoutlstm"  # make consistent to ZoneoutLSTM

  def __init__(self, n_out: int, zoneout_factor_cell: int = 0., zoneout_factor_output: int = 0., **kwargs):
    super().__init__(
      unit_opts={'zoneout_factor_cell': zoneout_factor_cell, 'zoneout_factor_output': zoneout_factor_output},
      n_out=n_out, unit="zoneoutlstm", **kwargs)

  # noinspection PyMethodOverriding
  def make_layer_dict(
        self, source: nn.LayerRef, *, state: nn.LayerState) -> nn.LayerDictRaw:
    """make layer"""
    return super().make_layer_dict(source, state=state)
