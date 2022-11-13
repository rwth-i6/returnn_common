"""
Variational weight noise

https://github.com/rwth-i6/returnn_common/issues/240
"""

from __future__ import annotations
from typing import TypeVar
from ... import nn


T_module = TypeVar('T_module', bound=nn.Module)


def variational_weight_noise(module: T_module, name: str, weight_noise_std: float) -> T_module:
  """
  :param module: module
  :param name: name of the weight parameter
  :param weight_noise_std: standard deviation of the weight noise

  Example::

      vn = 0.0075
      for mod in self.encoder.modules():
          if isinstance(mod, nn.LSTM):
              nn.variational_weight_noise(mod, "param_W_re", vn)
              nn.variational_weight_noise(mod, "param_W", vn)
          elif isinstance(mod, nn.Linear):
              nn.variational_weight_noise(mod, "weight", vn)

  """
  assert weight_noise_std > 0
  assert hasattr(module, name)
  weight = getattr(module, name)
  assert isinstance(weight, nn.Parameter)

  assert not hasattr(module, f"{name}_raw")
  setattr(module, f"{name}_raw", weight)

  with nn.Cond(nn.train_flag()) as cond:
    weight_noise = nn.random_normal(weight.shape_ordered, weight.dtype, stddev=weight_noise_std)
    cond.true = weight + weight_noise
    cond.false = weight
  setattr(module, name, cond.result)
  return module
