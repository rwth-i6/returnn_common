"""
Losses and distances

There is nothing specific about the functions here
except that they are commonly used as loss functions.
But they can also be used in other context when needed.

There is no reduction on batch or spatial axes.
E.g. cross_entropy just reduces the feature axis.

"""

from typing import Optional
from .. import nn


@nn.scoped
def cross_entropy(*, target: nn.LayerRef, estimated: nn.LayerRef, estimated_type: str,
                  label_smoothing: float = 0.0,
                  axis: Optional[nn.Dim] = None) -> nn.Layer:
  """
  Cross entropy H(target,estimated) (https://en.wikipedia.org/wiki/Cross_entropy).

  ``target`` is supposed to be in std prob space (normalized). It can also be sparse.
  ``estimated`` can be probs, log-probs or logits, specified via ``estimated_type``.

  Assuming both are in prob space, the cross entropy is:

    H(target,estimated) = -reduce_sum(target * log(estimated), axis=axis)
                        = -dot(target, log(estimated), reduce=axis)

  :param target: probs, normalized. can also be sparse
  :param estimated: probs, log-probs or logits, specified via ``estimated_type``
  :param estimated_type: "probs", "log-probs" or "logits"
  :param label_smoothing:
  :param axis: the axis to reduce over
  :return: cross entropy
  """
  if not axis:
    assert target.feature_dim
    axis = target.feature_dim
  if label_smoothing:
    target = nn.label_smoothing(target, label_smoothing, axis=axis)
  if estimated_type == "probs":
    return -nn.dot(target, nn.safe_log(estimated), reduce=axis)
  elif estimated_type == "log-probs":
    return -nn.dot(target, estimated, reduce=axis)
  elif estimated_type == "logits":
    return -nn.dot(target, nn.log_softmax(estimated, axis=axis), reduce=axis)
  else:
    raise ValueError("estimated_kind must be 'probs', 'log-probs' or 'logits'")
