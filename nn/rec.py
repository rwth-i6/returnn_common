"""
Basic RNNs.
"""

from typing import Optional, Union, Dict, List, Tuple, Callable, Any
from .. import nn


class _Rec(nn.Module):
  def __init__(self, *, out_dim: nn.Dim, unit: str, param_list: List[Tuple[str, Callable[[], Tuple[nn.Dim, ...]]]],
               in_dim: Optional[nn.Dim] = None,
               unit_opts: Optional[Dict[str, Any]] = None):
    super().__init__()
    self.out_dim = out_dim
    self.in_dim = in_dim
    self.unit = unit
    self.unit_opts = unit_opts
    self.param_list = param_list
    if in_dim:
      self._lazy_init(in_dim)

  def _lazy_init(self, in_dim: nn.Dim):
    if self.in_dim:
      assert self.in_dim == in_dim
    else:
      self.in_dim = in_dim
      for param, shape_func in self.param_list:
        shape = shape_func()
        setattr(self, f"param_{param}", nn.Parameter(shape))

  @nn.scoped
  def __call__(self, source: nn.LayerRef, *,
               axis: nn.Dim,
               state: Optional[Union[nn.LayerRef, Dict[str, nn.LayerRef], nn.NotSpecified]] = nn.NotSpecified,
               initial_state: Optional[Union[nn.LayerRef, Dict[str, nn.LayerRef], nn.NotSpecified]] = nn.NotSpecified,
               ) -> Tuple[nn.Layer, nn.LayerState]:
    self._lazy_init(source.dim)
    rec_layer_dict = {
      "class": "rec", "from": source,
      "in_dim": self.in_dim, "axis": axis, "out_dim": self.out_dim,
      "unit": self.unit}
    if self.unit_opts:
      rec_layer_dict["unit_opts"] = self.unit_opts
    reuse_params = {}
    for param, _ in self.param_list:
      param_ = getattr(self, f"param_{param}")
      reuse_params[param] = {"layer_output": param_}
    rec_layer_dict["reuse_params"] = {"map": reuse_params}
    nn.ReturnnWrappedLayerBase.handle_recurrent_state(
      rec_layer_dict, axis=axis, state=state, initial_state=initial_state)
    out = nn.make_layer(rec_layer_dict, name="rec")
    out_state = nn.ReturnnWrappedLayerBase.returnn_layer_get_recurrent_state(out)
    return out, out_state


class LSTM(_Rec):
  """
  LSTM operating on a sequence. returns (output, final_state) tuple, where final_state is (h,c).
  """
  def __init__(self, out_dim: nn.Dim, *, in_dim: Optional[nn.Dim] = None):
    self.param_W_re = None  # type: Optional[nn.Parameter]
    self.param_W = None  # type: Optional[nn.Parameter]
    self.param_b = None  # type: Optional[nn.Parameter]
    super().__init__(
      unit="nativelstm2",
      out_dim=out_dim, in_dim=in_dim,
      param_list=[
        ("W_re", lambda: (self.out_dim, 4 * self.out_dim)),
        ("W", lambda: (self.in_dim, 4 * self.out_dim)),
        ("b", lambda: (4 * self.out_dim,))])


class ZoneoutLSTM(_Rec):
  """
  LSTM with zoneout operating on a sequence. returns (output, final_state) tuple, where final_state is (h,c).
  """
  def __init__(self, out_dim: nn.Dim,
               *,
               in_dim: Optional[nn.Dim] = None,
               zoneout_factor_cell: float = 0., zoneout_factor_output: float = 0.):
    self.param_kernel = None  # type: Optional[nn.Parameter]
    self.param_bias = None  # type: Optional[nn.Parameter]
    super().__init__(
      unit="zoneoutlstm",
      out_dim=out_dim, in_dim=in_dim,
      unit_opts={'zoneout_factor_cell': zoneout_factor_cell, 'zoneout_factor_output': zoneout_factor_output},
      param_list=[
        ("kernel", lambda: (self.in_dim + self.out_dim, 4 * self.out_dim)),
        ("bias", lambda: (4 * self.out_dim,))])
