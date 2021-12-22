"""
Some generic utils (which doesn't fit into math_, array_, etc)
"""

from typing import Optional, Tuple, Dict, Union
from .. import nn
from ..nn import NotSpecified


# noinspection PyShadowingNames
def dropout(source: nn.LayerRef,
            dropout: float,
            *,
            axis: nn.Dim = NotSpecified,
            noise_shape: Dict[Union[str, nn.Dim], Optional[int]] = NotSpecified,
            on_forward: bool = False,
            name: Optional[str] = None
            ) -> nn.LayerRef:
  """
  Applies dropout.
  Dropout will only be applied during training (unless you set on_forward=True).

  :param nn.LayerRef source:
  :param float dropout: 0.0 means to apply no dropout.
  :param Dim axis:
  :param dict[str|nn.Dim,int|None] noise_shape: see :func:`returnn.tf.util.data.get_bc_shape`
  :param bool on_forward: apply dropout during inference
  :param str|None name:
  """
  assert isinstance(source, nn.LayerRef)
  if not dropout:
    return source
  opts = {"dropout": dropout}
  if axis is not NotSpecified:
    assert noise_shape is NotSpecified, "cannot provide both axis and noise_shape"
    noise_shape = {"*": 1, axis: None}
  if noise_shape is not NotSpecified:
    opts["dropout_noise_shape"] = noise_shape
  if on_forward:
    opts["dropout_on_forward"] = True
  from .base import make_layer
  return make_layer(
    {"class": "dropout", "from": source, **opts},
    name=name or "dropout")
