"""
Some generic utils (which doesn't fit into math_, array_, etc)
"""

from typing import Optional, Union, Sequence, Callable, Any
from .. import nn


# noinspection PyShadowingNames
def dropout(source: nn.LayerRef,
            dropout: float,
            *,
            axis: Union[nn.Dim, Sequence[nn.Dim]],
            on_forward: bool = False,
            name: Optional[str] = None
            ) -> nn.LayerRef:
  """
  Applies dropout.

  Dropout will only be applied during training (unless you set on_forward=True).

  When dropout is applied, the output will be scaled by 1/dropout.

  :param nn.LayerRef source:
  :param float dropout: 0.0 means to apply no dropout. 100% would mask everything.
    For every value in the tensor, the probability of it being dropped is drawn independently given this probability.
    The broadcasted axes are those not specified in ``axis``.
  :param axis: axis to apply dropout on. multiple axes can be specified.
    This defines the set of axes where the dropout mask is not broadcasted to.
    (RETURNN also has the ``noise_shape`` option but the ``axis`` option provides the same functionality.)
  :param bool on_forward: apply dropout during inference
  :param str|None name:
  """
  assert isinstance(source, nn.LayerRef)
  if not dropout:
    return source
  opts = {"dropout": dropout, "dropout_axis": axis}
  if on_forward:
    opts["dropout_on_forward"] = True
  from .base import make_layer
  return make_layer(
    {"class": "dropout", "from": source, **opts},
    name=name or "dropout")


def stop_gradient(source: nn.LayerRef, name: Optional[str] = None) -> nn.LayerRef:
  """wraps tf.stop_gradient"""
  return nn.scaled_gradient(source, scale=0, name=name)


def reinterpret_new_dim(source: nn.LayerRef, *, in_dim: nn.Dim, out_dim: nn.Dim,
                        name: Optional[str] = None) -> nn.Layer:
  """
  :return: source with in_dim replaced by out_dim
  """
  return nn.make_layer(
    {"class": "reinterpret_data", "set_dim_tags": {out_dim: in_dim}, "from": source},
    name=name or "new_dim")


def reinterpret_set_feature_dim(source: nn.LayerRef, in_dim: nn.Dim) -> nn.LayerRef:
  """
  :return: source with feature_dim set to in_dim
  """
  if in_dim == source.feature_dim:
    return source  # nothing to do
  assert in_dim in source.shape
  if not in_dim.is_feature_dim():
    in_dim_ = in_dim.copy(same_as_self=True, kind=nn.Dim.Types.Feature)
    source = nn.reinterpret_new_dim(source, in_dim=in_dim, out_dim=in_dim_)
  assert not source.data.sparse  # otherwise, it should have been correct before
  source = nn.make_layer(
    {"class": "reinterpret_data", "set_axes": {"F": in_dim}, "from": source},
    name="set_feature_dim")
  assert source.feature_dim == in_dim
  return source


def check_in_feature_dim_lazy_init(
      source: nn.LayerRef, in_dim: Optional[nn.Dim], lazy_init: Callable[[nn.Dim], Any]) -> nn.LayerRef:
  """
  This is a helper function for modules which want to lazily support assigning the in_dim.
  """
  if in_dim:
    if source.feature_dim == in_dim:
      return source  # all fine
    if in_dim in source.shape:
      return reinterpret_set_feature_dim(source, in_dim)
    raise ValueError(f"{source} does not have feature dim {in_dim}")
  # Not yet initialized.
  if not source.feature_dim:
    raise ValueError(f"{source} has no feature dim. define the in_dim explicitly")
  lazy_init(source.feature_dim)
  return source
