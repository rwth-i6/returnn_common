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

  def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim, *,
               unit: str, param_list: List[Tuple[str, Callable[[], Tuple[nn.Dim, ...]]]],
               unit_opts: Optional[Dict[str, Any]] = None):
    """
    :param out_dim: dimension tag for the output feature dimension
    :param unit: unit description string, see documentation for available recurrent cells
    :param param_list: (str, callable) pairs where the callable (usually lambda) returns all parameter dims
      (e.g. 1 for bias and 2 for weight matrices)
    :param in_dim: input feature dimension
    :param unit_opts: additional options for the recurrent unit
    """
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.unit = unit
    self.unit_opts = unit_opts
    self.param_list = param_list
    for param, shape_func in self.param_list:
      shape = shape_func()
      param_ = nn.Parameter(shape)
      param_.initial = nn.init.Glorot()
      setattr(self, f"param_{param}", param_)

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
    rec_layer_dict = {
      "class": "rec", "from": source,
      "in_dim": self.in_dim, "axis": axis, "out_dim": self.out_dim,
      "unit": self.unit}
    if self.unit_opts:
      rec_layer_dict["unit_opts"] = self.unit_opts
    # We use the reuse_params mechanism from RETURNN to explicitly pass the parameters.
    reuse_params = {}
    for param, shape_func in self.param_list:
      param_ = getattr(self, f"param_{param}")
      assert isinstance(param_, nn.Parameter)
      shape = shape_func()
      reuse_params[param] = {"layer_output": param_, "shape": shape}
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
  def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim):
    self.param_W_re = None  # type: Optional[nn.Parameter]
    self.param_W = None  # type: Optional[nn.Parameter]
    self.param_b = None  # type: Optional[nn.Parameter]
    super().__init__(
      in_dim=in_dim, out_dim=out_dim,
      unit="nativelstm2",
      param_list=[
        ("W_re", lambda: (self.out_dim, 4 * self.out_dim)),
        ("W", lambda: (self.in_dim, 4 * self.out_dim)),
        ("b", lambda: (4 * self.out_dim,))])


class ZoneoutLSTM(_Rec):
  """
  LSTM with zoneout. returns (output, state) tuple, where state is (h,c).
  """
  def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim, *,
               zoneout_factor_cell: float = 0., zoneout_factor_output: float = 0.):
    self.param_kernel = None  # type: Optional[nn.Parameter]
    self.param_bias = None  # type: Optional[nn.Parameter]
    super().__init__(
      in_dim=in_dim, out_dim=out_dim,
      unit="zoneoutlstm",
      unit_opts={'zoneout_factor_cell': zoneout_factor_cell, 'zoneout_factor_output': zoneout_factor_output},
      param_list=[
        ("kernel", lambda: (self.in_dim + self.out_dim, 4 * self.out_dim)),
        ("bias", lambda: (4 * self.out_dim,))])
