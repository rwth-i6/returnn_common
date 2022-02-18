"""
Convolution
"""

from typing import Optional, Sequence, Tuple, Union
from .. import nn


# noinspection PyAbstractClass
class _ConvOrTransposedConv(nn.Module):
  """
  Base class for both convolution and transposed convolution.
  """
  nd: Optional[int] = None
  _transposed: bool

  def __init__(self,
               out_dim: nn.Dim,
               filter_size: Union[Sequence[Union[int, nn.Dim]], int, nn.Dim],
               *,
               in_dim: Optional[nn.Dim],
               padding: str,
               with_bias: bool,
               ):
    """
    :param Dim out_dim:
    :param filter_size: (width,), (height,width) or (depth,height,width) for 1D/2D/3D conv.
      the input data ndim must match, or you can add dimensions via input_expand_dims or input_add_feature_dim.
      it will automatically swap the batch-dim to the first axis of the input data.
    :param Dim|None in_dim:
    """
    super().__init__()
    self.out_dim = out_dim
    if isinstance(filter_size, (int, nn.Dim)):
      if self.nd in (None, 1):
        filter_size = [filter_size]
      else:
        filter_size = [filter_size] * self.nd
    assert isinstance(filter_size, (tuple, list))
    if self.nd:
      assert self.nd == len(filter_size)
    else:
      self.nd = len(filter_size)
    self.filter_size = [
      s if isinstance(s, nn.Dim) else nn.SpatialDim(f"filter-dim{i}", s)
      for i, s in enumerate(filter_size)]
    self.in_dim = in_dim
    self.padding = padding
    self.with_bias = with_bias
    self.filter = None  # type: Optional[nn.Parameter]
    self.bias = None  # type: Optional[nn.Parameter]
    if in_dim:
      self._lazy_init(in_dim)

  def _lazy_init(self, in_dim: nn.Dim):
    self.in_dim = in_dim
    self.filter = nn.Parameter(
      self.filter_size +
      ([nn.dim_match_priority_when_needed(self.in_dim, self.out_dim), self.out_dim]
       if not self._transposed
       else [self.out_dim, nn.dim_match_priority_when_needed(self.in_dim, self.out_dim)]))
    self.filter.initial = nn.init.Glorot()
    if self.with_bias:
      self.bias = nn.Parameter([self.out_dim])
      self.bias.initial = 0.

  def _call_nd1(self, source: nn.Tensor, *,
                in_dim: Optional[nn.Dim] = None,
                in_spatial_dim: nn.Dim,
                out_spatial_dim: Optional[nn.Dim] = None) -> Tuple[nn.Tensor, nn.Dim]:
    assert self.nd == 1
    out, (out_spatial_dim,) = self.__class__.__base__.__call__(
      self, source, in_dim=in_dim, in_spatial_dims=[in_spatial_dim],
      out_spatial_dims=[out_spatial_dim] if out_spatial_dim else None)
    return out, out_spatial_dim


class _Conv(_ConvOrTransposedConv):
  """
  A generic convolution layer which supports 1D, 2D and 3D convolution.
  Base class for :class:`Conv1d`, :class:`Conv2d`, :class:`Conv3d`.
  """

  _transposed = False

  # noinspection PyShadowingBuiltins,PyShadowingNames
  def __init__(self,
               out_dim: nn.Dim,
               filter_size: Union[Sequence[Union[int, nn.Dim]], int, nn.Dim],
               *,
               in_dim: Optional[nn.Dim] = None,
               padding: str,
               strides: Optional[Union[int, Sequence[int]]] = None,
               dilation_rate: Optional[Union[int, Sequence[int]]] = None,
               groups: Optional[int] = None,
               with_bias: bool = True,
               ):
    """
    :param Dim out_dim:
    :param filter_size: (width,), (height,width) or (depth,height,width) for 1D/2D/3D conv.
      the input data ndim must match, or you can add dimensions via input_expand_dims or input_add_feature_dim.
      it will automatically swap the batch-dim to the first axis of the input data.
    :param Dim|None in_dim:
    :param str padding: "same" or "valid"
    :param int|Sequence[int] strides: strides for the spatial dims,
      i.e. length of this tuple should be the same as filter_size, or a single int.
    :param int|Sequence[int] dilation_rate: dilation for the spatial dims
    :param int groups: grouped convolution
    :param bool with_bias: if True, will add a bias to the output features
    """
    super().__init__(out_dim=out_dim, filter_size=filter_size, in_dim=in_dim, padding=padding, with_bias=with_bias)
    self.strides = strides
    self.dilation_rate = dilation_rate
    self.groups = groups

  @nn.scoped
  def __call__(self, source: nn.Tensor, *,
               in_dim: Optional[nn.Dim] = None,
               in_spatial_dims: Sequence[nn.Dim],
               out_spatial_dims: Optional[Sequence[nn.Dim]] = None
               ) -> Tuple[nn.Tensor, Sequence[nn.Dim]]:
    source = nn.check_in_feature_dim_lazy_init(source, in_dim, self.in_dim, self._lazy_init)
    for in_spatial_dim in in_spatial_dims:
      if in_spatial_dim not in source.shape:
        raise ValueError(f"{self}: source {source} does not have spatial dim {in_spatial_dim}")
    if not out_spatial_dims:
      out_spatial_dims = [
        nn.SpatialDim(f"{nn.NameCtx.current_ctx().layer_abs_name_scope}:out-spatial-dim{i}")
        for i, s in enumerate(self.filter_size)]
      for i in range(len(self.filter_size)):
        s = 1 if not self.strides else self.strides[i]
        if self.filter_size[i].dimension == s == 1 or (s == 1 and self.padding.lower() == "same"):
          out_spatial_dims[i] = in_spatial_dims[i]
    layer_dict = {
      "class": "conv", "from": source,
      "in_dim": self.in_dim, "in_spatial_dims": in_spatial_dims,
      "out_dim": self.out_dim, "out_spatial_dims": out_spatial_dims,
      "filter_size": [d.dimension for d in self.filter_size], "padding": self.padding}
    if self.strides:
      layer_dict["strides"] = self.strides
    if self.dilation_rate:
      layer_dict["dilation_rate"] = self.dilation_rate
    if self.groups:
      layer_dict["groups"] = self.groups
    layer_dict.update({"filter": self.filter, "with_bias": self.with_bias})
    if self.with_bias:
      layer_dict["bias"] = self.bias
    out = nn.make_layer(layer_dict, name="conv")
    return out, out_spatial_dims


class Conv1d(_Conv):
  """
  1D convolution
  """

  nd = 1

  def __init__(self,
               out_dim: nn.Dim,
               filter_size: Union[int, nn.Dim],
               *,
               in_dim: Optional[nn.Dim] = None,
               padding: str,
               strides: Optional[int] = None,
               dilation_rate: Optional[int] = None,
               groups: Optional[int] = None,
               with_bias: bool = True,
               ):
    """
    :param Dim out_dim:
    :param int|Dim filter_size:
    :param str padding: "same" or "valid"
    :param int|None strides: strides for the spatial dims,
      i.e. length of this tuple should be the same as filter_size, or a single int.
    :param int|None dilation_rate: dilation for the spatial dims
    :param int groups: grouped convolution
    :param Dim|None in_dim:
    :param bool with_bias: if True, will add a bias to the output features
    """
    super().__init__(
      out_dim=out_dim, filter_size=[filter_size],
      in_dim=in_dim, padding=padding, strides=strides, dilation_rate=dilation_rate,
      groups=groups, with_bias=with_bias)

  __call__ = _ConvOrTransposedConv._call_nd1


class Conv2d(_Conv):
  """
  2D convolution
  """
  nd = 2


class Conv3d(_Conv):
  """
  3D convolution
  """
  nd = 3


class _TransposedConv(_ConvOrTransposedConv):
  """
  Transposed convolution, sometimes also called deconvolution.
  See :func:`tf.nn.conv2d_transpose` (currently we support 1D/2D).
  """

  nd: Optional[int] = None
  _transposed = True

  # noinspection PyShadowingBuiltins,PyShadowingNames
  def __init__(self,
               out_dim: nn.Dim,
               filter_size: Sequence[Union[int, nn.Dim]],
               *,
               in_dim: Optional[nn.Dim] = None,
               padding: str,
               remove_padding: Union[Sequence[int], int] = 0,
               output_padding: Optional[Union[Sequence[Optional[int]], int]] = None,
               strides: Optional[Sequence[int]] = None,
               with_bias: bool = True,
               ):
    """
    :param Dim out_dim:
    :param list[int] filter_size:
    :param list[int]|None strides: specifies the upscaling. by default, same as filter_size
    :param str padding: "same" or "valid"
    :param list[int]|int remove_padding:
    :param list[int|None]|int|None output_padding:
    :param Dim|None in_dim:
    :param bool with_bias: whether to add a bias. enabled by default
    """
    super().__init__(out_dim=out_dim, filter_size=filter_size, in_dim=in_dim, padding=padding, with_bias=with_bias)
    self.strides = strides
    self.remove_padding = remove_padding
    self.output_padding = output_padding

  @nn.scoped
  def __call__(self, source: nn.Tensor, *,
               in_dim: Optional[nn.Dim] = None,
               in_spatial_dims: Sequence[nn.Dim],
               out_spatial_dims: Optional[Sequence[nn.Dim]] = None
               ) -> Tuple[nn.Tensor, Sequence[nn.Dim]]:
    source = nn.check_in_feature_dim_lazy_init(source, in_dim, self.in_dim, self._lazy_init)
    if not out_spatial_dims:
      out_spatial_dims = [
        nn.SpatialDim(f"{nn.NameCtx.current_ctx().layer_abs_name_scope}:out-spatial-dim{i}")
        for i, s in enumerate(self.filter_size)]
      for i in range(len(self.filter_size)):
        s = self.filter_size[i].dimension if not self.strides else self.strides[i]
        if self.filter_size[i].dimension == s == 1 or (s == 1 and self.padding.lower() == "same"):
          out_spatial_dims[i] = in_spatial_dims[i]
    layer_dict = {
      "class": "transposed_conv", "from": source,
      "in_dim": self.in_dim, "in_spatial_dims": in_spatial_dims,
      "out_dim": self.out_dim, "out_spatial_dims": out_spatial_dims,
      "filter_size": [d.dimension for d in self.filter_size], "padding": self.padding}
    if self.remove_padding:
      layer_dict["remove_padding"] = self.remove_padding
    if self.output_padding:
      layer_dict["output_padding"] = self.output_padding
    if self.strides:
      layer_dict["strides"] = self.strides
    layer_dict.update({"filter": self.filter, "with_bias": self.with_bias})
    if self.with_bias:
      layer_dict["bias"] = self.bias
    out = nn.make_layer(layer_dict, name="conv")
    return out, out_spatial_dims


class TransposedConv1d(_TransposedConv):
  """
  1D transposed convolution
  """
  nd = 1

  __call__ = _ConvOrTransposedConv._call_nd1


class TransposedConv2d(_TransposedConv):
  """
  2D transposed convolution
  """
  nd = 2


class TransposedConv3d(_TransposedConv):
  """
  3D transposed convolution
  """
  nd = 3


def _pool_nd(
      source: nn.Tensor,
      *,
      nd: int,
      mode: str,
      pool_size: Union[Sequence[int], int],
      padding: str = nn.NotSpecified,
      dilation_rate: Union[Sequence[int], int] = nn.NotSpecified,
      strides: Optional[Union[Sequence[int], int]] = nn.NotSpecified,
      in_spatial_dims: Sequence[nn.Dim],
      out_spatial_dims: Optional[Sequence[nn.Dim]] = nn.NotSpecified,
      name: Optional[Union[str, nn.NameCtx]] = None) -> Tuple[nn.Tensor, Sequence[nn.Dim]]:
  """
  A generic N-D pooling layer.
  This would usually be done after a convolution for down-sampling.

  :param Tensor source:
  :param str mode: "max" or "avg"
  :param tuple[int] pool_size: shape of the window of each reduce
  :param str padding: "valid" or "same"
  :param tuple[int]|int dilation_rate:
  :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
  :param Sequence[Dim] in_spatial_dims:
  :param Sequence[Dim]|None out_spatial_dims:
  :param str|NameCtx|None name:
  :return: layer, out_spatial_dims
  """
  if isinstance(pool_size, int):
    pool_size = [pool_size] * nd
  assert isinstance(pool_size, (list, tuple))
  assert len(pool_size) == nd

  if out_spatial_dims is None or out_spatial_dims is nn.NotSpecified:
    out_spatial_dims = [
      d.copy(same_as_self=False, description=f'{nn.NameCtx.current_ctx().get_abs_name()}:out_spatial_dims{i}')
      for i, d in enumerate(in_spatial_dims)]
  args = {
    'mode': mode,
    'pool_size': pool_size,
    'padding': padding,
    'dilation_rate': dilation_rate,
    'strides': strides,
    'in_spatial_dims': in_spatial_dims,
    'out_spatial_dims': out_spatial_dims,
    }
  args = {key: value for (key, value) in args.items() if value is not nn.NotSpecified}
  layer = nn.make_layer({
    'class': 'pool',
    'from': source,
    **args}, name=name or 'pool')
  return layer, out_spatial_dims


def pool1d(
      source: nn.Tensor,
      *,
      mode: str,
      pool_size: int,
      padding: str = nn.NotSpecified,
      dilation_rate: Union[int] = nn.NotSpecified,
      strides: Optional[int] = nn.NotSpecified,
      in_spatial_dim: nn.Dim,
      out_spatial_dim: Optional[nn.Dim] = nn.NotSpecified,
      name: Optional[Union[str, nn.NameCtx]] = None) -> Tuple[nn.Tensor, nn.Dim]:
  """
  1D pooling.

  :param Tensor source:
  :param str mode: "max" or "avg"
  :param tuple[int] pool_size: shape of the window of each reduce
  :param str padding: "valid" or "same"
  :param tuple[int]|int dilation_rate:
  :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
  :param Sequence[Dim] in_spatial_dim:
  :param Sequence[Dim]|None out_spatial_dim:
  :param str|NameCtx|None name:
  :return: layer, out_spatial_dim
  """
  out, (out_spatial_dim,) = _pool_nd(
    source=source, nd=1, mode=mode, pool_size=pool_size, padding=padding,
    dilation_rate=dilation_rate, strides=strides, in_spatial_dims=[in_spatial_dim],
    out_spatial_dims=[out_spatial_dim], name=name)
  return out, out_spatial_dim


def pool2d(
      source: nn.Tensor,
      *,
      mode: str,
      pool_size: Union[Sequence[int], int],
      padding: str = nn.NotSpecified,
      dilation_rate: Union[Sequence[int], int] = nn.NotSpecified,
      strides: Optional[Union[Sequence[int], int]] = nn.NotSpecified,
      in_spatial_dims: Sequence[nn.Dim],
      out_spatial_dims: Optional[Sequence[nn.Dim]] = nn.NotSpecified,
      name: Optional[Union[str, nn.NameCtx]] = None) -> Tuple[nn.Tensor, Sequence[nn.Dim]]:
  """
  2D pooling.

  :param Tensor source:
  :param str mode: "max" or "avg"
  :param tuple[int] pool_size: shape of the window of each reduce
  :param str padding: "valid" or "same"
  :param tuple[int]|int dilation_rate:
  :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
  :param Sequence[Dim] in_spatial_dims:
  :param Sequence[Dim]|None out_spatial_dims:
  :param str|NameCtx|None name:
  :return: layer, out_spatial_dims
  """
  return _pool_nd(
    source=source, nd=2, mode=mode, pool_size=pool_size, padding=padding,
    dilation_rate=dilation_rate, strides=strides,
    in_spatial_dims=in_spatial_dims, out_spatial_dims=out_spatial_dims,
    name=name)


def pool3d(
      source: nn.Tensor,
      *,
      mode: str,
      pool_size: Union[Sequence[int], int],
      padding: str = nn.NotSpecified,
      dilation_rate: Union[Sequence[int], int] = nn.NotSpecified,
      strides: Optional[Union[Sequence[int], int]] = nn.NotSpecified,
      in_spatial_dims: Sequence[nn.Dim],
      out_spatial_dims: Optional[Sequence[nn.Dim]] = nn.NotSpecified,
      name: Optional[Union[str, nn.NameCtx]] = None) -> Tuple[nn.Tensor, Sequence[nn.Dim]]:
  """
  3D pooling.

  :param Tensor source:
  :param str mode: "max" or "avg"
  :param tuple[int] pool_size: shape of the window of each reduce
  :param str padding: "valid" or "same"
  :param tuple[int]|int dilation_rate:
  :param tuple[int]|int|None strides: in contrast to tf.nn.pool, the default (if it is None) will be set to pool_size
  :param Sequence[Dim] in_spatial_dims:
  :param Sequence[Dim]|None out_spatial_dims:
  :param str|NameCtx|None name:
  :return: layer, out_spatial_dims
  """
  return _pool_nd(
    source=source, nd=3, mode=mode, pool_size=pool_size, padding=padding,
    dilation_rate=dilation_rate, strides=strides,
    in_spatial_dims=in_spatial_dims, out_spatial_dims=out_spatial_dims,
    name=name)
