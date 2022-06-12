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
    self._call_counter = 0

  _state_dim = nn.FeatureDim("random-state", None)

  def __call__(self, **kwargs) -> nn.Tensor:
    # For every call, we create a new state var to make sure there is no non-determinism.
    # https://github.com/rwth-i6/returnn_common/issues/148
    # No explicit seed, so RETURNN uses its global seed.
    init_state, _ = nn.random_state_init(out_dim=self._state_dim)
    state_var = nn.Parameter(init_state.shape_ordered, init_state.dtype)
    setattr(self, f"state_var{self._call_counter}", state_var)
    state_var.initial = init_state
    self._call_counter += 1
    return nn.random(
      explicit_state=state_var, auto_update_state=True,
      **kwargs)

  def uniform(self,
              shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
              *,
              minval: Union[int, float, nn.Tensor] = 0,
              maxval: Union[int, float, nn.Tensor]
              ) -> nn.Tensor:
    """
    Random uniform.

    :param shape:
    :param dtype:
    :param minval: inclusive
    :param maxval: exclusive
    """
    return self(distribution="uniform", shape=shape, dtype=dtype, minval=minval, maxval=maxval)

  def normal(self,
             shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
             *,
             mean: Union[int, float, nn.Tensor] = 0.,
             stddev: Union[int, float, nn.Tensor] = 1.,
             ) -> nn.Tensor:
    """normal"""
    return self(distribution="normal", shape=shape, dtype=dtype, mean=mean, stddev=stddev)

  def truncated_normal(self,
                       shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
                       *,
                       mean: Union[int, float, nn.Tensor] = 0.,
                       stddev: Union[int, float, nn.Tensor] = 1.,
                       ) -> nn.Tensor:
    """truncated normal"""
    return self(distribution="truncated_normal", shape=shape, dtype=dtype, mean=mean, stddev=stddev)

  def bernoulli(self, shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
                *,
                p: Union[float, nn.Tensor]
                ) -> nn.Tensor:
    """bernoulli. 0 or 1. p: prob for 1. uniform(1) <= p -> 1, else 0"""
    x = self.uniform(shape=shape, dtype=dtype, minval=0., maxval=1.)
    return nn.where(x <= p, nn.ones(shape=(), dtype=dtype), nn.zeros(shape=(), dtype=dtype))


def random_uniform(shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
                   *,
                   minval: Union[int, float, nn.Tensor] = 0,
                   maxval: Union[int, float, nn.Tensor]
                   ) -> nn.Tensor:
  """
  Random uniform.

  :param shape:
  :param dtype:
  :param minval: inclusive
  :param maxval: exclusive
  """
  return Random().uniform(shape=shape, dtype=dtype, minval=minval, maxval=maxval)


def random_normal(shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
                  *,
                  mean: Union[int, float, nn.Tensor] = 0.,
                  stddev: Union[int, float, nn.Tensor] = 1.,
                  ) -> nn.Tensor:
  """Random normal"""
  return Random().normal(shape=shape, dtype=dtype, mean=mean, stddev=stddev)


def random_truncated_normal(
      shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
      *,
      mean: Union[int, float, nn.Tensor] = 0.,
      stddev: Union[int, float, nn.Tensor] = 1.,
      ) -> nn.Tensor:
  """Random truncated normal"""
  return Random().truncated_normal(shape=shape, dtype=dtype, mean=mean, stddev=stddev)


def random_bernoulli(
      shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
      *,
      p: Union[float, nn.Tensor]
      ) -> nn.Tensor:
  """Random Bernoulli. 0 or 1. p: prob for 1. uniform(1) <= p -> 1, else 0"""
  return Random().bernoulli(shape=shape, dtype=dtype, p=p)
