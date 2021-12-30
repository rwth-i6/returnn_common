"""
Normalization functions such as batch norm
"""


from typing import Optional, Sequence, Union, Tuple
from .. import nn


@nn.scoped
def moments(x: nn.LayerRef, axis: Union[nn.Dim, Sequence[nn.Dim]]) -> Tuple[nn.Layer, nn.Layer]:
  """
  :return: mean, variance
  """
  mean = nn.reduce(x, mode="mean", axis=axis, name="mean")
  # stop_gradient does not change the gradient here
  variance = nn.reduce(
    nn.squared_difference(x, nn.stop_gradient(mean)),
    mode="mean", axis=axis, name="variance")
  return mean, variance


class BatchNorm(nn.Module):
  """
  Batch normalization. https://arxiv.org/abs/1502.03167

  Note that the default arguments differ from corresponding batch norm in RETURNN.
  See here for discussion on defaults: https://github.com/rwth-i6/returnn/issues/522
  """

  def __init__(self, feature_dim: Optional[nn.Dim] = None):
    super().__init__()
    self.feature_dim = feature_dim
    self.mean = None  # type: Optional[nn.Parameter]
    self.var = None  # type: Optional[nn.Parameter]
    if feature_dim:
      self._lazy_init(feature_dim)

  def _lazy_init(self, feature_dim: nn.Dim):
    self.feature_dim = feature_dim
    self.mean = nn.Parameter([feature_dim], trainable=False)
    self.var = nn.Parameter([feature_dim], trainable=False)

  def __call__(self, source: nn.LayerRef, *, feature_dim: Optional[nn.Dim], spatial_dims: Sequence[nn.Dim],
               epsilon: float = 1e-5) -> nn.Layer:
    assert self.feature_dim or source.feature_dim or feature_dim
    if feature_dim:
      self._lazy_init(feature_dim)
    elif not self.feature_dim:
      self._lazy_init(source.feature_dim)

    reduce_dims = [d for d in source.data.dim_tags if d != feature_dim]
    mean, variance = moments(source, reduce_dims)
    # TODO: handle running mean/var ...
    return (source - mean) * nn.rsqrt(variance + epsilon)
