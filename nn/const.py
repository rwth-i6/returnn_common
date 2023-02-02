"""
Const helpers
"""

from typing import Optional, Union, Sequence
import numpy
from .. import nn


def constant(
    value: Union[int, float, bool, numpy.ndarray],
    *,
    shape: Optional[Sequence[nn.Dim]] = None,
    dtype: Optional[str] = None,
    sparse_dim: Optional[nn.Dim] = None,
    name: Optional[Union[str, nn.NameCtx]] = None,
) -> nn.Tensor:
    """
    Output is a constant value.

    :param int|float|bool|numpy.ndarray value:
    :param Sequence[nn.Dim] shape: for verification, and defining dim tags
    :param str|None dtype:
    :param nn.Dim|None sparse_dim:
    :param str|nn.NameCtx|None name:
    :return: layer
    """
    args = {
        "value": value,
        "shape": shape,
        "dtype": dtype,
        "sparse_dim": sparse_dim,
        "shape_deps": nn.get_dim_deps(shape) if shape else None,
    }
    args = {key: value for (key, value) in args.items() if value is not None}
    return nn.make_layer({"class": "constant", **args}, name=name or "constant")


def zeros(
    shape: Sequence[nn.Dim], dtype: Optional[str] = nn.NotSpecified, *, name: Optional[Union[str, nn.NameCtx]] = None
) -> nn.Tensor:
    """
    zeros
    """
    value = 0
    if dtype is None or dtype is nn.NotSpecified:
        dtype = "float32"
    if dtype == "bool":
        value = False
    return nn.constant(value=value, shape=shape, dtype=dtype, name=name or "zeros")


def zeros_like(value: nn.Tensor, *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
    """
    zeros with shape and dtype from value. But there is no dependency on value in the computation graph.
    """
    return zeros(shape=value.shape_ordered, dtype=value.dtype, name=name)


def ones(
    shape: Sequence[nn.Dim], dtype: Optional[str] = nn.NotSpecified, *, name: Optional[Union[str, nn.NameCtx]] = None
) -> nn.Tensor:
    """
    ones
    """
    value = 1
    if dtype is None or dtype is nn.NotSpecified:
        dtype = "float32"
    if dtype == "bool":
        value = True
    return nn.constant(value=value, shape=shape, dtype=dtype, name=name or "ones")


def ones_like(value: nn.Tensor, *, name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
    """
    ones with shape and dtype from value. But there is no dependency on value in the computation graph.
    """
    return ones(shape=value.shape_ordered, dtype=value.dtype, name=name)
