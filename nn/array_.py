"""
Array (Tensor) functions
"""

from typing import Optional, Tuple, List, Union
from returnn.util.basic import NotSpecified
from .. import nn


def concat(*sources: Tuple[nn.LayerRef, nn.Dim],
           allow_broadcast=False,
           name: Optional[str] = None) -> nn.Layer:
  """
  Concatenates multiple sources (by default in feature axis).
  """
  opts = {}
  if allow_broadcast:
    opts["allow_broadcast"] = True
  return nn.make_layer({"class": "concat", "from": sources, **opts}, name=name or "concat")


def cum_concat_step(
      source: nn.LayerRef, *, state: nn.LayerState,
      out_spatial_dim: nn.Dim,
      name: Optional[str] = None) -> Tuple[nn.Layer, nn.LayerState]:
  """
  Concatenates all previous frames of a time-axis.
  See RETURNN :class:`CumConcatLayer` for details.
  """
  from ._generated_layers import rec_cum_concat
  return rec_cum_concat(
    source=source, axis=nn.single_step_dim,
    state=state, out_spatial_dim=out_spatial_dim, name=name)


def split(source: nn.LayerRef, *,
          axis: nn.Dim,
          out_dims: Union[List[nn.Dim], Tuple[nn.Dim, ...]],
          name: Optional[str] = None) -> Tuple[nn.LayerRef, ...]:
  """
  Split the input on the specified axis (by default feature).
  Basically a wrapper around tf.split.
  """
  from ._generated_layers import _split
  from .base import _get_sub_layer
  res = _split(source, axis=axis, out_dims=out_dims, name=name)
  src_axis_int = source.data.get_axis_from_description(axis)
  return tuple(
    _get_sub_layer(
      layer=res, name=str(i),
      data=source.data.copy_template_replace_dim_tag(
        axis=src_axis_int, new_dim_tag=dim,
        name=f"{source.data.name}/split:{i}:{dim.description}"))
    for i, dim in enumerate(out_dims))


def window(
      source: nn.LayerRef, *,
      axis: nn.Dim,
      window_dim: nn.Dim,
      window_left: Optional[int] = NotSpecified,
      window_right: Optional[int] = NotSpecified,
      padding: str = NotSpecified,
      stride: int = NotSpecified,
      name: Optional[str] = None) -> nn.Layer:
  """
  Window. See :func:`_generated_layers._window`.
  """
  from ._generated_layers import rec_window
  layer, state = rec_window(
    source,
    window_dim=window_dim, window_left=window_left, window_right=window_right,
    axis=axis, padding=padding, stride=stride,
    name=name)
  del state
  return layer


def window_step(
      source: nn.LayerRef, *, state: nn.LayerState,
      window_dim: nn.Dim,
      padding: str = NotSpecified,
      stride: int = NotSpecified,
      name: Optional[str] = None) -> Tuple[nn.Layer, nn.LayerState]:
  """
  Window into the past when iterating.
  See :func:`_generated_layers._window`.
  """
  from ._generated_layers import rec_window
  return rec_window(
    source, state=state,
    window_dim=window_dim, window_left=window_dim.dimension - 1, window_right=0,
    axis=nn.single_step_dim, padding=padding, stride=stride,
    name=name)
