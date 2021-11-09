"""
Array (Tensor) functions
"""

from typing import Optional, Tuple, List, Union
from returnn.util.basic import NotSpecified
from .. import nn


def concat(*sources: Tuple[nn.LayerRef, Union[str, nn.DimensionTag]],
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
      new_dim: nn.DimensionTag,
      name: Optional[str] = None) -> Tuple[nn.Layer, nn.LayerState]:
  """
  Concatenates all previous frames of a time-axis.
  See RETURNN :class:`CumConcatLayer` for details.
  """
  from ._generated_layers import _cum_concat
  return _cum_concat(source=source, state=state, new_dim=new_dim, name=name)


def split(source: nn.LayerRef, *,
          axis: Optional[str] = NotSpecified,
          num_splits: Optional[int] = NotSpecified,
          size_splits: Optional[Union[List[int], Tuple[int, ...]]] = NotSpecified,
          name: Optional[str] = None) -> Tuple[nn.LayerRef, ...]:
  """
  Split the input on the specified axis (by default feature).
  Basically a wrapper around tf.split.
  """
  from ._generated_layers import _split
  from .base import get_sub_layer
  res = _split(source, axis=axis, num_splits=num_splits, size_splits=size_splits, name=name)
  if num_splits is None or num_splits is NotSpecified:
    assert isinstance(size_splits, (tuple, list))
    num_splits = len(size_splits)
  return tuple(get_sub_layer(res, str(i)) for i in range(num_splits))


def window(
      source: nn.LayerRef, *,
      window_size: int,
      window_left: Optional[int] = NotSpecified,
      window_right: Optional[int] = NotSpecified,
      axis: str = NotSpecified,
      padding: str = NotSpecified,
      stride: int = NotSpecified,
      name: Optional[str] = None) -> nn.Layer:
  """
  Window. See :func:`_generated_layers._window`.
  """
  from ._generated_layers import _window
  layer, state = _window(
    source,
    window_size=window_size, window_left=window_left, window_right=window_right,
    axis=axis, padding=padding, stride=stride,
    name=name)
  del state
  return layer


def window_step(
      source: nn.LayerRef, *, state: nn.LayerState,
      window_size: int,
      axis: str = NotSpecified,
      padding: str = NotSpecified,
      stride: int = NotSpecified,
      name: Optional[str] = None) -> Tuple[nn.Layer, nn.LayerState]:
  """
  Window into the past when iterating.
  See :func:`_generated_layers._window`.
  """
  from ._generated_layers import _window
  return _window(
    source, state=state,
    window_size=window_size, window_left=window_size - 1, window_right=0,
    axis=axis, padding=padding, stride=stride,
    name=name)
