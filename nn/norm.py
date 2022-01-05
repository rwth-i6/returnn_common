"""
Normalization functions such as batch norm
"""


from typing import Optional, Sequence, Union, Tuple
from .. import nn


@nn.scoped
def moments(x: nn.LayerRef, axis: Union[nn.Dim, Sequence[nn.Dim]]) -> Tuple[nn.Layer, nn.Layer]:
  """
  :param x: input
  :param axis: the axis to be reduced, to calculate statistics over
  :return: mean, variance. it has the same shape as the input with the axis removed
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

  We calculate statistics over all axes except the given in_dim.
  I.e. all other axes are reduced for the statistics.

  To compensate the normalization, there are learnable parameters gamma and beta
  (optional, used when option `affine` is True).

  The usual behavior depends on whether this is used in training or evaluation,
  although this often configurable in other frameworks.
  The usual behavior, in training::

      # Using statistics from current batch.
      mean_cur_batch, variance_cur_batch = moments(source, reduce_dims)
      y = (x - mean_cur_batch) / sqrt(variance_cur_batch + epsilon)
      y = gamma * y + beta

      # Updating running statistics for later use.
      mean = (1 - momentum) * mean + momentum * mean_cur_batch
      variance = (1 - momentum) * variance + momentum * variance_cur_batch

  The usual behavior, not in training (i.e. in evaluation)::

      # Using collected statistics. Not using statistics from current batch.
      y = (x - mean) / sqrt(variance + epsilon)
      y = gamma * y + beta

  """

  def __init__(self, in_dim: Optional[nn.Dim] = None, *, affine: bool = True):
    """
    :param in_dim: the feature dimension of the input
    :param affine: whether to use learnable parameters gamma and beta
    """
    super().__init__()
    self.in_dim = in_dim
    self.mean = None  # type: Optional[nn.Parameter]
    self.var = None  # type: Optional[nn.Parameter]
    self.affine = affine
    self.gamma = None  # type: Optional[nn.Parameter]
    self.beta = None  # type: Optional[nn.Parameter]
    if in_dim:
      self._lazy_init(in_dim)

  def _lazy_init(self, in_dim: nn.Dim):
    self.in_dim = in_dim
    self.mean = nn.Parameter([in_dim], auxiliary=True)
    self.var = nn.Parameter([in_dim], auxiliary=True)
    if self.affine:
      self.gamma = nn.Parameter([in_dim])
      self.beta = nn.Parameter([in_dim])

  def __call__(self, source: nn.LayerRef, *,
               epsilon: float = 1e-5) -> nn.Layer:
    source = nn.check_in_feature_dim_lazy_init(source, self.in_dim, self._lazy_init)
    reduce_dims = [d for d in source.data.dim_tags if d != self.in_dim]
    mean_cur_batch, variance_cur_batch = moments(source, reduce_dims)
    mean_cur_batch.verify_out_shape({self.in_dim})
    variance_cur_batch.verify_out_shape({self.in_dim})
    # TODO: handle running mean/var ...
    return (source - mean_cur_batch) * nn.rsqrt(variance_cur_batch + epsilon)
