"""
Some generic utils (which doesn't fit into math_, array_, etc)
"""

from typing import Optional, Union, Sequence
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
