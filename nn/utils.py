"""
Some generic utils (which doesn't fit into math_, array_, etc)
"""

from typing import Any, Optional, Tuple
from .. import nn
from .base import LayerRef, NotSpecified


# noinspection PyShadowingNames
def dropout(source: LayerRef,
            dropout: float,
            *,
            noise_shape: Any = NotSpecified,
            on_forward: bool = False,
            name: Optional[str] = None
            ) -> LayerRef:
  """
  Applies dropout.
  Dropout will only be applied during training (unless you set on_forward=True).

  :param LayerRef source:
  :param float dropout: 0.0 means to apply no dropout.
  :param dict[str|tuple,int|None] noise_shape: see :func:`returnn.tf.util.data.get_bc_shape`
  :param bool on_forward: apply dropout during inference
  :param str|None name:
  """
  assert isinstance(source, LayerRef)
  if not dropout:
    return source
  opts = {"dropout": dropout}
  if noise_shape is not NotSpecified:
    opts["dropout_noise_shape"] = noise_shape
  if on_forward:
    opts["dropout_on_forward"] = True
  from .base import make_layer
  return make_layer(
    {"class": "dropout", "from": source, **opts},
    name=name or "dropout")


def ken_lm_state_step(source: nn.LayerRef, *, state: nn.LayerState, **kwargs) -> Tuple[nn.Layer, nn.LayerState]:
  """
  See :func:`._generated_layers._ken_lm_state`.
  """
  from ._generated_layers import _ken_lm_state
  return _ken_lm_state(source, state=state, initial_state=None, **kwargs)
