"""
Common functions for random number generation.
"""

from typing import Union, Sequence
from .. import nn


class Random(nn.Module):
  """
  Random number generator with state.
  """
  def __init__(self):
    super(Random, self).__init__()
    # No explicit seed, so RETURNN uses its global seed.
    init_state = nn.random_state_init()
    self.state_var = nn.Parameter(init_state.data.dim_tags, init_state.dtype)

  @nn.scoped
  def __call__(self, **kwargs) -> nn.Layer:
    return nn.random(
      explicit_state=self.state_var, auto_update_state=True,
      **kwargs)

  def uniform(self,
              shape, dtype=None,
              *,
              minval: Union[int, float, nn.LayerRef] = 0, maxval: Union[int, float, nn.LayerRef]):
    """uniform"""
    return self(distribution="uniform", shape=shape, dtype=dtype, minval=minval, maxval=maxval)

  def normal(self,
             shape: Sequence[nn.Dim], dtype=None,
             *,
             mean: Union[int, float, nn.LayerRef],
             stddev: Union[int, float, nn.LayerRef],
             ) -> nn.Layer:
    """normal"""
    return self(distribution="normal", shape=shape, dtype=dtype, mean=mean, stddev=stddev)

  def truncated_normal(self,
                       shape: Sequence[nn.Dim], dtype=None,
                       *,
                       mean: Union[int, float, nn.LayerRef],
                       stddev: Union[int, float, nn.LayerRef],
                       ) -> nn.Layer:
    """truncated normal"""
    return self(distribution="truncated_normal", shape=shape, dtype=dtype, mean=mean, stddev=stddev)
