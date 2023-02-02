"""
Utilities for dimension tags, dimensions, axes.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple
from ... import nn


# noinspection PyShadowingNames
def make_dim_from_length(length: nn.Tensor, dim: Optional[nn.Dim] = None) -> nn.Dim:
    """
    Given some length tensor, creates a dim tag for it.
    """
    if dim is None:
        dim = nn.SpatialDim(length.get_abs_name())
    # The actual range tensor is never used, but this has the side effect to set up the dim tag.
    r, dim = nn.range_from_length(length, out_spatial_dim=dim)
    from ..base import _register_dim_deps_when_novel

    _register_dim_deps_when_novel(dim, [r])
    return dim


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


def length(
    dim: nn.Dim,
    *,
    dtype: str = nn.NotSpecified,
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
    return nn.make_layer({"class": "length", "from": nn.get_dim_deps(dim), "axis": dim, **args}, name="length")


def reinterpret_new_dim(
    source: nn.Tensor, *, in_dim: nn.Dim, out_dim: Optional[nn.Dim] = None, name: Optional[str] = None
) -> Tuple[nn.Tensor, nn.Dim]:
    """
    :return: source with in_dim replaced by out_dim.
      this does not work for the sparse_dim. see :func:`reinterpret_set_sparse_dim` for that case.
    """
    if not out_dim:
        out_dim = in_dim.copy(same_as_self=False, description="new-dim")
    out = nn.make_layer(
        {"class": "reinterpret_data", "set_dim_tags": {in_dim: out_dim}, "from": source}, name=name or "new_dim"
    )
    return out, out_dim


def reinterpret_set_sparse_dim(source: nn.Tensor, out_dim: nn.Dim, *, name: str = "set_sparse_dim") -> nn.Tensor:
    """
    :return: source with sparse_dim set to out_dim
    """
    return nn.make_layer(
        {"class": "reinterpret_data", "set_sparse": True, "set_sparse_dim": out_dim, "from": source}, name=name
    )


def dim_match_priority_when_needed(dim: nn.Dim, *other_dims: nn.Dim) -> nn.Dim:
    """
    :return: maybe copy of dim with higher match_priority if needed to distinguish from other_dims
    """
    if dim in other_dims:
        return dim.copy(match_priority=1)
    return dim


def range_over_dim(
    dim: nn.Dim,
    *,
    dtype: str = nn.NotSpecified,
    sparse: bool = False,
) -> nn.Tensor:
    """
    Creates a tensor with shape [dim] with values 0,1,2,...,dim-1.
    In RETURNN, this is the range_in_axis layer.

    :param nn.Dim dim:
    :param str dtype: default is int32
    :param bool sparse:
    :return: layer
    """
    args = {}
    if sparse:
        args["sparse"] = True
    if dtype is not nn.NotSpecified:
        args["dtype"] = dtype
    return nn.make_layer(
        {"class": "range_in_axis", "from": nn.get_dim_deps(dim), "axis": dim, **args}, name="range_over_dim"
    )
