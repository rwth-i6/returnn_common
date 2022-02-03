"""
Some generic utils (which doesn't fit into math_, array_, etc)
"""

from typing import Optional, Union, Sequence, Tuple, Callable, Any
from .. import nn


def convert_to_layer_ref(x: Union[nn.Tensor, int, float, complex, bool, str]) -> nn.Tensor:
  """
  In case it is not a layer ref yet, it will make some constant.
  """
  if isinstance(x, nn.Tensor):
    return x
  return nn.constant(value=x)


def where(cond: nn.Tensor,
          true_: Union[nn.Tensor, float, int],
          false_: Union[nn.Tensor, float, int],
          *, name: Optional[str] = None) -> nn.Tensor:
  """
  Wraps tf.where, which is SwitchLayer in RETURNN.

  :return: true_ if cond else false_, elemwise.
  """
  return nn.make_layer({
    'class': 'switch', "condition": cond, "true_from": true_, "false_from": false_}, name=name or 'where')


# noinspection PyShadowingNames
def dropout(source: nn.Tensor,
            dropout: float,
            *,
            axis: Union[nn.Dim, Sequence[nn.Dim]],
            on_forward: bool = False,
            name: Optional[str] = None
            ) -> nn.Tensor:
  """
  Applies dropout.

  Dropout will only be applied during training (unless you set on_forward=True).

  When dropout is applied, the output will be scaled by 1/dropout.

  :param nn.Tensor source:
  :param float dropout: 0.0 means to apply no dropout. 100% would mask everything.
    For every value in the tensor, the probability of it being dropped is drawn independently given this probability.
    The broadcasted axes are those not specified in ``axis``.
  :param axis: axis to apply dropout on. multiple axes can be specified.
    This defines the set of axes where the dropout mask is not broadcasted to.
    (RETURNN also has the ``noise_shape`` option but the ``axis`` option provides the same functionality.)
  :param bool on_forward: apply dropout during inference
  :param str|None name:
  """
  assert isinstance(source, nn.Tensor)
  if not dropout:
    return source
  opts = {"dropout": dropout, "dropout_axis": axis}
  if on_forward:
    opts["dropout_on_forward"] = True
  from .base import make_layer
  return make_layer(
    {"class": "dropout", "from": source, **opts},
    name=name or "dropout")


def stop_gradient(source: nn.Tensor, name: Optional[str] = None) -> nn.Tensor:
  """wraps tf.stop_gradient"""
  return nn.scaled_gradient(source, scale=0, name=name)


def reinterpret_new_dim(source: nn.Tensor, *, in_dim: nn.Dim, out_dim: Optional[nn.Dim] = None,
                        name: Optional[str] = None) -> Tuple[nn.Tensor, nn.Dim]:
  """
  :return: source with in_dim replaced by out_dim
  """
  if not out_dim:
    out_dim = in_dim.copy(same_as_self=False, description="new-dim")
  out = nn.make_layer(
    {"class": "reinterpret_data", "set_dim_tags": {in_dim: out_dim}, "from": source},
    name=name or "new_dim")
  return out, out_dim


def check_in_feature_dim_lazy_init(
      source: nn.Tensor, in_dim: Optional[nn.Dim], lazy_init: Callable[[nn.Dim], Any]) -> nn.Tensor:
  """
  This is a helper function for modules which want to lazily support assigning the in_dim.
  """
  if in_dim:
    if in_dim in source.shape:
      return source  # all fine
    raise ValueError(f"{source} does not have feature dim {in_dim}")
  # Not yet initialized.
  if not source.feature_dim:
    raise ValueError(f"{source} has no feature dim. define the in_dim explicitly")
  lazy_init(source.feature_dim)
  return source


def range_for_dim(dim: nn.Dim, *, dim_source: Optional[nn.Tensor] = None, sparse: bool = False) -> nn.Tensor:
  """
  range [0,dim-1] for dim

  :param nn.Dim dim:
  :param nn.Tensor dim_source: only needed for dynamic dims currently. might not be needed at some later point.
  :param bool sparse:
  """
  if dim.dimension is None:
    # We need to wrap nn.range_in_axis. For that, we need some source data which contains this dim.
    # This is also needed for proper dependency resolution.
    if dim_source:
      assert dim in dim_source.shape
      return nn.range_in_axis(dim_source, axis=dim, sparse=sparse)
    # We might get a layer via dim.src_data when we keep some history of data sources.
    # But we don't have that yet and also not sure if this is the best solution.
    raise NotImplementedError(f"range_for_dim({dim}) not implemented for dynamic dims yet")
  out, _ = nn.range(start=0, limit=dim.dimension, out_spatial_dim=dim, sparse=sparse)
  return out


@nn.scoped
def sparse_to_dense(source: nn.Tensor, *,
                    label_value: Union[nn.Tensor, int, float],
                    other_value: Union[nn.Tensor, int, float]) -> nn.Tensor:
  """
  Converts a sparse tensor to a dense one.

  This is a more generic variant of "one_hot".

  Note that usually this is not needed as most other functions should handle sparse tensors just fine
  and much more efficiently than they would be with dense tensors.
  """
  assert source.data.sparse
  axis = source.data.sparse_dim
  indices = range_for_dim(axis, dim_source=source, sparse=True)
  return nn.where(source == indices, label_value, other_value)


def one_hot(source: nn.Tensor, *, name: Optional[str] = None) -> nn.Tensor:
  """
  one_hot. special case of :func:`sparse_to_dense`.

  Note that usually this is not needed as most other functions should handle sparse tensors just fine
  and much more efficiently than they would be with dense tensors.
  """
  return sparse_to_dense(source, label_value=1., other_value=0., name=name or "one_hot")


def smooth_one_hot(source: nn.Tensor, *, label_prob: Union[nn.Tensor, float],
                   name: Optional[str] = None) -> nn.Tensor:
  """
  Smooth variant of :func:`one_hot`.
  Uses ``label_prob`` for the labels and ``(1 - label_prob) / (dim - 1)`` for the remaining values.
  This is used for label smoothing.
  """
  assert source.data.sparse
  if source.data.sparse_dim.dimension is None:
    raise NotImplementedError(f"smooth_one_hot({source}) not implemented for dynamic dims")
  return sparse_to_dense(
    source, label_value=label_prob, other_value=(1. - label_prob) / (source.data.sparse_dim.dimension - 1),
    name=name or "smooth_one_hot")


def label_smoothing(source: nn.Tensor, smoothing: Union[nn.Tensor, float],
                    *, axis: Optional[nn.Dim] = None) -> nn.Tensor:
  """
  label smoothing
  """
  if not axis:
    assert source.feature_dim
    axis = source.feature_dim
  if source.data.sparse:
    assert source.data.sparse_dim == axis
    return smooth_one_hot(source, label_prob=1. - smoothing + smoothing / axis.dimension)
  else:
    assert axis in source.shape
    return source * (1. - smoothing) + smoothing / axis.dimension
