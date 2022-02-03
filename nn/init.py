"""
Common parameter initialization functions.

https://github.com/rwth-i6/returnn/wiki/Parameter-initialization
"""

from .. import nn
import math


class VarianceScaling:
  """
  Provides a generalized way for initializing weights.
  All the common initialization methods are special cases
  such as Xavier Glorot and Kaiming He.

  Code adopted from TensorFlow VarianceScaling.
  """
  scale = 1.0
  mode = "fan_in"  # fan_in, fan_out, fan_avg
  distribution = "truncated_normal"  # normal, untruncated_normal, truncated_normal, uniform
  dtype = "float32"

  def __init__(self, scale: float = None, mode: str = None, distribution: str = None, dtype: str = None):
    if scale is not None:
      self.scale = scale
    if mode is not None:
      self.mode = mode
    if distribution is not None:
      self.distribution = distribution
    if dtype is not None:
      self.dtype = dtype

    if self.scale <= 0.:
      raise ValueError(f"Argument `scale` must be a positive float. Received: {self.scale}")
    if self.mode not in {"fan_in", "fan_out", "fan_avg"}:
      raise ValueError(
        f"Argument `mode` should be one of ('fan_in', 'fan_out', 'fan_avg'). Received: {self.mode}")
    if self.distribution not in {"normal", "uniform", "truncated_normal", "untruncated_normal"}:
      raise ValueError(
        "Argument `distribution` should be one of ('normal', 'uniform', 'truncated_normal', 'untruncated_normal'). "
        f"Received: {self.distribution}")

  def __call__(self, shape, dtype=None) -> nn.Tensor:
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    fan_in, fan_out = _compute_fans(shape)
    if self.mode == "fan_in":
      scale /= max(1., fan_in)
    elif self.mode == "fan_out":
      scale /= max(1., fan_out)
    else:
      scale /= max(1., (fan_in + fan_out) / 2.)
    return self._random(shape=shape, dtype=dtype, scale=scale)

  def _random(self, shape, scale, dtype=None) -> nn.Tensor:
    if self.distribution in {"truncated_normal", "normal"}:
      # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
      stddev = math.sqrt(scale) / .87962566103423978
      return nn.random(
        distribution=self.distribution, static=True,
        shape=shape, mean=0.0, stddev=stddev, dtype=dtype)
    elif self.distribution == "untruncated_normal":
      stddev = math.sqrt(scale)
      return nn.random(
        distribution=self.distribution, static=True,
        shape=shape, mean=0.0, stddev=stddev, dtype=dtype)
    elif self.distribution == "uniform":
      limit = math.sqrt(3.0 * scale)
      return nn.random(
        distribution=self.distribution, static=True,
        shape=shape, minval=-limit, maxval=limit, dtype=dtype)
    else:
      raise ValueError(f"invalid distribution {self.distribution!r}")


class Glorot(VarianceScaling):
  """
  Xavier Glorot
  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
  """
  scale = 1.0
  mode = "fan_avg"
  distribution = "uniform"


class He(VarianceScaling):
  """
  Kaiming He
  https://arxiv.org/pdf/1502.01852.pdf
  """
  scale = 2.0
  mode = "fan_in"
  distribution = "normal"


HeNormal = He


class HeUniform(He):
  """
  He-init but using a uniform distribution.
  """
  distribution = "uniform"


def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.

  Args:
    shape: Integer shape tuple or TF tensor shape.

  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)
