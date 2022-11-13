"""
Weight dropout.

Also known as "variational dropout" or "Bayesian dropout",
sometimes applied for LSTM weights,
but this can also be applied to any other weights.

https://github.com/rwth-i6/returnn_common/issues/100
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, TypeVar
from ... import nn


T_module = TypeVar('T_module', bound=nn.Module)


def weight_dropout(
  module: T_module, name: str, dropout: float,
  *,
  axis: Optional[Union[nn.Dim, Sequence[nn.Dim]]] = None,
) -> T_module:
  """
  :param module: module
  :param name: name of the weight parameter
  :param dropout: dropout probability
  :param axis: axis to apply dropout on. see :func:`nn.dropout`

  Example usage::

      wdrop = 0.1
      for mod in self.encoder.modules():
          if isinstance(mod, nn.LSTM):
              nn.weight_dropout(mod, "param_W_re", wdrop)
              nn.weight_dropout(mod, "param_W", wdrop)
          elif isinstance(mod, nn.Linear):
              nn.weight_dropout(mod, "weight", wdrop)

  """
  assert hasattr(module, name)
  weight = getattr(module, name)
  assert isinstance(weight, nn.Parameter)
  if not axis:
    axis = weight.shape_ordered

  assert not hasattr(module, f"{name}_raw")
  setattr(module, f"{name}_raw", weight)
  weight = nn.dropout(weight, dropout, axis=axis)
  setattr(module, name, weight)
  return module
