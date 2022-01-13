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
                  axis: Optional[nn.Dim] = None) -> nn.Layer:
  """
  Cross entropy H(target,estimated) (https://en.wikipedia.org/wiki/Cross_entropy).

  ``target`` is supposed to be in std prob space (normalized). It can also be sparse.
  ``estimated`` can be probs, log-probs or logits, specified via ``estimated_type``.

  Assuming both are in prob space, the cross entropy is:

    H(target,estimated) = -reduce_sum(target * log(estimated), axis=axis)
                        = -dot(target, log(estimated), reduce=axis)

  In case you want label smoothing, you can use e.g.::

      ce = nn.cross_entropy(
        target=nn.label_smoothing(target, 0.1),
        estimated=estimated)

  :param target: probs, normalized. can also be sparse
  :param estimated: probs, log-probs or logits, specified via ``estimated_type``
  :param estimated_type: "probs", "log-probs" or "logits"
  :param axis: the axis to reduce over
  :return: cross entropy
  """
  if not axis:
    assert target.feature_dim
    axis = target.feature_dim
  if estimated_type == "logits" and target.data.sparse:
    # This is a common case and TF provides an optimized function for it, so use that directly.
    return nn.sparse_softmax_cross_entropy_with_logits(logits=estimated, targets=target, axis=axis)
  if estimated_type == "probs":
    log_prob = nn.safe_log(estimated)
  elif estimated_type == "log-probs":
    log_prob = estimated
  elif estimated_type == "logits":
    log_prob = nn.log_softmax(estimated, axis=axis)
  else:
    raise ValueError("estimated_kind must be 'probs', 'log-probs' or 'logits'")
  return -nn.dot(target, log_prob, reduce=axis)
