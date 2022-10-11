"""
Basic RNNs.
"""

from typing import Optional, Union, Dict, Sequence, List, Tuple, Callable, Any
from .. import nn


class _Rec(nn.Module):
  """
  Wraps the RecLayer in RETURNN for specific units like LSTM.
  This can operate both single-step and on a sequence.
  See :func:`__call__`.
  """

  unit: str = None
  param_list: Tuple[str] = ()

  def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim, *,
               unit: Optional[str] = None, unit_opts: Optional[Dict[str, Any]] = None,
               param_list: Optional[Tuple[str]] = None):
    """
    :param out_dim: dimension tag for the output feature dimension
    :param unit: unit description string, see documentation for available recurrent cells
    :param in_dim: input feature dimension
    :param unit_opts: additional options for the recurrent unit
    """
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    if unit is not None:
      self.unit = unit
    else:
      assert self.unit is not None
    self.unit_opts = unit_opts
    if param_list is not None:
      self.param_list = param_list

  def __call__(self, source: nn.Tensor, *,
               axis: nn.Dim,
               state: Union[nn.LayerState, Dict[str, nn.Tensor], nn.Tensor, None, nn.NotSpecified] = nn.NotSpecified,
               direction: int = 1,
               ) -> Tuple[nn.Tensor, nn.LayerState]:
    """
    :param source:
    :param axis: nn.single_step_dim specifies to operate for a single step
    :param state: prev state when operating a single step or initial state when operating on an axis
    :param direction: 1 for forward direction, -1 for backward direction
    :return: out, out_state. out_state is the new or last state.
    """
    assert self.in_dim in source.shape
    rec_layer_dict = {
      "class": "rec", "from": source,
      "in_dim": self.in_dim, "axis": axis, "out_dim": self.out_dim,
      "unit": self.unit}
    if self.unit_opts:
      rec_layer_dict["unit_opts"] = self.unit_opts
    # We use the reuse_params mechanism from RETURNN to explicitly pass the parameters.
    reuse_params = {}
    for param in self.param_list:
      param_ = getattr(self, f"param_{param}")
      assert isinstance(param_, nn.Parameter)
      reuse_params[param] = {"layer_output": param_, "shape": param_.shape_ordered}
    rec_layer_dict["reuse_params"] = {"map": reuse_params}
    assert direction in [1, -1]
    if direction == -1:
      assert axis is not nn.single_step_dim, "Can not reverse direction for single step recurrent layers"
      rec_layer_dict["direction"] = -1
    nn.ReturnnWrappedLayerBase.handle_recurrent_state(rec_layer_dict, axis=axis, state=state)
    out = nn.make_layer(rec_layer_dict, name="rec")
    out_state = nn.ReturnnWrappedLayerBase.returnn_layer_get_recurrent_state(out)
    return out, out_state

  def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> nn.LayerState:
    """
    :return: default initial state
    """
    from .const import zeros
    if "lstm" in self.unit.lower():
      return nn.LayerState(h=zeros(list(batch_dims) + [self.out_dim]), c=zeros(list(batch_dims) + [self.out_dim]))
    raise NotImplementedError(f"{self}.default_initial_state for RecLayer with unit {self.unit!r}")


class LSTM(_Rec):
  """
  LSTM. returns (output, state) tuple, where state is (h,c).
  """

  unit = "nativelstm2"
  param_list = ("W_re", "W", "b")

  def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim):
    super().__init__(in_dim=in_dim, out_dim=out_dim)
    self.param_W_re = nn.Parameter((self.out_dim, 4 * self.out_dim))
    self.param_W_re.initial = nn.init.Glorot()
    self.param_W = nn.Parameter((self.in_dim, 4 * self.out_dim))
    self.param_W.initial = nn.init.Glorot()
    self.param_b = nn.Parameter((4 * self.out_dim,))
    self.param_b.initial = 0.


class ZoneoutLSTM(_Rec):
  """
  LSTM with zoneout. returns (output, state) tuple, where state is (h,c).
  """
  unit = "zoneoutlstm"
  param_list = ("kernel", "bias")

  def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim, *,
               zoneout_factor_cell: float = 0., zoneout_factor_output: float = 0.):
    super().__init__(
      in_dim=in_dim, out_dim=out_dim,
      unit_opts={'zoneout_factor_cell': zoneout_factor_cell, 'zoneout_factor_output': zoneout_factor_output})
    self.param_kernel = nn.Parameter((self.in_dim + self.out_dim, 4 * self.out_dim))
    self.param_kernel.initial = nn.init.Glorot()
    self.param_bias = nn.Parameter((4 * self.out_dim,))
    self.param_bias.initial = 0.
