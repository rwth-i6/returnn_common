"""
Array (Tensor) functions
"""

from typing import Optional, Union, Tuple, List
from returnn.util.basic import NotSpecified
from .base import LayerRef, Layer


def concat(sources: Union[List[LayerRef], Tuple[LayerRef, ...]], *,
           axis: Optional[str] = NotSpecified,
           name: Optional[str] = None) -> Layer:
  """
  Concatenates multiple sources (by default in feature axis).
  """
  if axis is NotSpecified or axis is None or axis.upper() == "F":
    # standard case
    from .base import make_layer
    return make_layer({"class": "copy", "from": sources}, name=name or "concat")
  else:
    raise NotImplementedError(f"Cannot handle concat with axis {axis!r} yet")


def split(source: LayerRef, *,
          axis: Optional[str] = NotSpecified,
          num_splits: Optional[int] = NotSpecified,
          size_splits: Optional[List[int]] = NotSpecified,
          ) -> Tuple[LayerRef, ...]:
  """
  Split the input on the specified axis (by default feature).
  Basically a wrapper around tf.split.
  """
  from ._generated_layers import _split
  from .base import get_sub_layer
  res = _split(source, axis=axis, num_splits=num_splits, size_splits=size_splits)
  if num_splits is None:
    assert isinstance(size_splits, (tuple, list))
    num_splits = len(size_splits)
  return tuple(get_sub_layer(res, str(i)) for i in range(num_splits))
