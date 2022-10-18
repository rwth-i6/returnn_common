"""
Array (Tensor) functions
"""

from typing import Optional, Sequence, Tuple, List, Union
from returnn.util.basic import NotSpecified
from .. import nn


def dim_value(dim: nn.Dim) -> Union[nn.Tensor, int]:
  """
  :return: like tf.shape(source)[axis], or specifically max(nn.length(source, axis=axis))
  """
  if dim.dimension is not None:
    return dim.dimension
  length_ = nn.length(dim)
  if not length_.shape:
    return length_
  return nn.reduce(length_, mode="max", axis=length_.shape_ordered)


def length(dim: nn.Dim,
           *,
           dtype: str = NotSpecified,
           sparse: bool = False,
           ) -> Union[nn.Tensor, int]:
  """
  :param nn.Dim dim:
  :param str dtype: default is int32
  :param bool sparse:
  :return: individual sequence lengths of dim tag (commonly shape [B])
  """
  if dim.dimension is not None:
    return dim.dimension
  args = {}
  if dtype is not nn.NotSpecified:
    args["dtype"] = dtype
  if sparse:
    args["sparse"] = True
  return nn.make_layer({
    'class': 'length',
    'from': nn.get_dim_deps(dim),
    'axis': dim,
    **args}, name='length')


def reshape(source: nn.Tensor, old_dims: Sequence[nn.Dim], new_dims: Sequence[nn.Dim]) -> nn.Tensor:
  """
  Wraps tf.reshape.

  You should use :func:`split_dims` or :func:`merge_dims`
  when you want to split or merge dimensions.
  This here is for doing any other kind of reshape.
  This can be used for clever indexing, slicing, padding tricks.

  :param source: e.g. (..., old_dims, ...)
  :param old_dims: the old dims which should be reshaped into new_dims.
    This should only cover those dims which should be reshaped,
    not all the dims of the source.
  :param new_dims: the new dims which should be reshaped from old_dims.
    This is excluding any of the other dims in the source.
  :return: e.g. (..., new_dims, ...)
  """
  return nn.make_layer(
    {
      "class": "reshape",
      "from": source,
      "old_dims": old_dims,
      "new_dims": new_dims,
      "extra_deps": nn.get_dim_deps(new_dims),
    },
    name="reshape")


def expand_dim(source: nn.Tensor, *, dim: nn.Dim, name: Optional[str] = None) -> nn.Tensor:
  """
  Expand the source by the given dimension,
  which should be 1.

  Note that this is *never* needed for broadcasting.
  All broadcasting should always happen automatically.

  This might be needed for convolution or concatenation.

  This can be reversed via :func:`squeeze`.
  """
  if dim.dimension != 1:
    raise ValueError(f"{dim} is not a 1-dim")
  # We use SplitDimsLayer for this.
  # ExpandDimsLayer in RETURNN currently would allow to use a dim tag.
  # Now search for a good axis to split via some heuristics.
  source_dims = [d for d in source.shape_ordered if not d.is_batch_dim()]
  if not source_dims:
    # Unfortunately, for scalars (ignoring batch), split_dims does not work.
    return source + nn.zeros(source.shape_ordered + (dim,), dtype=source.dtype)
  if dim.is_spatial_dim():
    if any(d.is_spatial_dim() for d in source_dims):
      axis = [d for d in source_dims if d.is_spatial_dim()][-1]
      return nn.split_dims(source, axis=axis, dims=(axis, dim), name=name)
    else:
      axis = source_dims[0]
      return nn.split_dims(source, axis=axis, dims=(dim, axis), name=name)
  elif dim.is_feature_dim():
    if any(d.is_feature_dim() for d in source_dims):
      axis = [d for d in source_dims if d.is_feature_dim()][-1]
      return nn.split_dims(source, axis=axis, dims=(axis, dim), name=name)
    else:
      axis = source_dims[-1]
      return nn.split_dims(source, axis=axis, dims=(axis, dim), name=name)
  else:
    raise ValueError(f"{dim} is not a spatial or feature dim")


def concat(*sources: Tuple[nn.Tensor, nn.Dim],
           allow_broadcast=False,
           name: Optional[str] = None
           ) -> (nn.Tensor, nn.Dim):
  """
  Concatenates multiple sources in the specified dimension.
  """
  assert sources
  opts = {}
  if allow_broadcast:
    opts["allow_broadcast"] = True
  else:
    dims = sources[0][0].shape - {sources[0][1]}
    for src, dim in sources:
      assert src.shape - {dim} == dims, f"concat {sources}, need allow_broadcast=True"
  out_dim = sum(d for _, d in sources)
  res = nn.make_layer(
    {"class": "concat", "from": sources, "out_dim": out_dim, **opts},
    name=name or "concat", name_ctx_ignore_top_stack_frames=1)
  out_dim = res.data.get_dim_tag_from_description(out_dim)  # maybe adapt batch info
  return res, out_dim


def concat_features(*sources: nn.Tensor, allow_broadcast=False) -> nn.Tensor:
  """
  Concatenates multiple sources, using feature_dim of each source,
  so make sure that the feature_dim is correctly set.
  """
  src_pairs = []
  for src in sources:
    assert src.feature_dim is not None
    src_pairs.append((src, src.feature_dim))
  res, out_dim = concat(*src_pairs, allow_broadcast=allow_broadcast)
  assert res.feature_dim == out_dim
  return res


def cum_concat_step(
      source: nn.Tensor, *, state: nn.LayerState,
      out_spatial_dim: Optional[nn.Dim] = None,
      name: Optional[str] = None) -> Tuple[nn.Tensor, nn.Dim, nn.LayerState]:
  """
  Concatenates all previous frames of a time-axis.
  See RETURNN :class:`CumConcatLayer` for details.
  """
  nn.auto_setup_name_ctx_ignore_func(cum_concat_step)
  from ._generated_layers import rec_cum_concat
  return rec_cum_concat(
    source=source, axis=nn.single_step_dim,
    state=state, out_spatial_dim=out_spatial_dim, name=name)


def split(source: nn.Tensor, *,
          axis: nn.Dim,
          out_dims: Union[List[nn.Dim], Tuple[nn.Dim, ...]],
          name: Optional[str] = None) -> Tuple[nn.Tensor, ...]:
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
      source: nn.Tensor, *,
      axis: nn.Dim,
      window_dim: nn.Dim,
      window_left: Optional[int] = NotSpecified,
      window_right: Optional[int] = NotSpecified,
      padding: str = NotSpecified,
      stride: int = NotSpecified,
      name: Optional[str] = None) -> Tuple[nn.Tensor, nn.Dim]:
  """
  Window. See :func:`rec_window`.
  """
  nn.auto_setup_name_ctx_ignore_func(window)
  from ._generated_layers import rec_window
  layer, (window_dim, out_spatial_dim), state = rec_window(
    source,
    window_dim=window_dim, window_left=window_left, window_right=window_right,
    axis=axis, padding=padding, stride=stride,
    name=name)
  del state
  return layer, out_spatial_dim


def window_step(
      source: nn.Tensor, *, state: nn.LayerState,
      window_dim: nn.Dim,
      name: Optional[str] = None) -> Tuple[nn.Tensor, nn.LayerState]:
  """
  Window into the past when iterating.
  See :func:`rec_window`.
  """
  nn.auto_setup_name_ctx_ignore_func(window_step)
  from ._generated_layers import rec_window
  out, _, state = rec_window(
    source, state=state,
    window_dim=window_dim, window_left=window_dim.dimension - 1, window_right=0,
    axis=nn.single_step_dim,
    name=name)
  return out, state


def boolean_mask(source: nn.Tensor, *,
                 mask: nn.Tensor,
                 in_spatial_dim: nn.Dim,
                 out_spatial_dim: Optional[nn.Dim] = None
                 ) -> Tuple[nn.Tensor, nn.Dim]:
  """
  Applies the mask on the source tensor, i.e. reducing the axis.

  For mask of shape [B,T], source of shape [B,T,D],
  it would return shape [B,T',D], where T' = sum(mask, axis=T).
  """
  if not out_spatial_dim:
    out_spatial_dim = nn.SpatialDim(f"{mask.name_ctx.get_abs_name()}:spatial")
  return nn.make_layer(
    {
      "class": "masked_computation",
      "mask": mask,
      "in_spatial_dim": in_spatial_dim,
      "out_spatial_dim": out_spatial_dim,
      "unit": {"class": "copy", "from": source}
    }, name="boolean_mask"), out_spatial_dim
