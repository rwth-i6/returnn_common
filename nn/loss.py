"""
Losses and distances

There is nothing specific about the functions here
except that they are commonly used as loss functions.
But they can also be used in other context when needed.

There is no reduction on batch or spatial axes.
E.g. cross_entropy just reduces the feature axis.

Reduction on batch or spatial axes is not necessary
and should *not* be done when this is used as a loss function
because RETURNN will handle the proper accumulation.

To use some tensor as a loss in RETURNN,
call :func:`nn.Tensor.mark_as_loss`.

Despite the functions in this module,
also see:
* :func:`nn.ctc_loss`

"""

from typing import Optional, Union
from .. import nn


def cross_entropy(*, target: nn.Tensor, estimated: nn.Tensor, estimated_type: str,
                  axis: Optional[nn.Dim] = None) -> nn.Tensor:
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


def binary_cross_entropy(*,
                         target: nn.Tensor,
                         pos_estimated: nn.Tensor, pos_estimated_type: str,
                         pos_weight: Optional[Union[float, nn.Tensor]] = None):
  """
  Binary cross entropy, or also called sigmoid cross entropy.

  :param target: (sparse) target labels, 0 (positive) or 1 (negative), i.e. binary.
  :param pos_estimated: positive class logits. probs = sigmoid(logits).
  :param pos_estimated_type: "logits" only supported currently
  :param pos_weight: weight for positive class.

  Code and documentation partly borrowed from TensorFlow.

  A value `pos_weight > 1` decreases the false negative count, hence increasing
  the recall.
  Conversely, setting `pos_weight < 1` decreases the false positive count and
  increases the precision.
  This can be seen from the fact that `pos_weight` is introduced as a
  multiplicative coefficient for the positive labels term
  in the loss expression:

      labels * -log(sigmoid(logits)) * pos_weight +
          (1 - labels) * -log(1 - sigmoid(logits))

  For brevity, let `x = logits`, `z = labels`.  The logistic loss is

        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  For x < 0, to avoid overflow in exp(-x), we reformulate the above

        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))

  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation

      max(x, 0) - x * z + log(1 + exp(-abs(x)))
  """
  if pos_estimated_type != "logits":
    raise NotImplementedError(
      f"binary_cross_entropy, pos_estimated_type {pos_estimated_type!r}, only 'logits' supported")
  logits = pos_estimated

  if pos_weight is not None:
    # Code adapted from tf.nn.weighted_cross_entropy_with_logits.
    # The logistic loss formula from above is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
    # To avoid branching, we use the combined version
    #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
    log_weight = 1 + (pos_weight - 1) * target
    return (
      (1 - target) * logits +
      log_weight * (nn.log1p(nn.exp(-nn.abs(logits))) + nn.relu(-logits))
      )

  # Code adapted from tf.nn.sigmoid_cross_entropy_with_logits.
  # The logistic loss formula from above is
  #   x - x * z + log(1 + exp(-x))
  # For x < 0, a more numerically stable formula is
  #   -x * z + log(1 + exp(x))
  # Note that these two expressions can be combined into the following:
  #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
  # To allow computing gradients at zero, we define custom versions of max and
  # abs functions.
  cond = (logits >= 0)
  relu_logits = nn.where(cond, logits, 0)
  neg_abs_logits = nn.where(cond, -logits, logits)  # pylint: disable=invalid-unary-operand-type
  return (
    relu_logits - logits * target +
    nn.log1p(nn.exp(neg_abs_logits))
    )


def kl_div(*, target: nn.Tensor, target_type: str,
           estimated: nn.Tensor, estimated_type: str,
           axis: Optional[nn.Dim] = None) -> nn.Tensor:
  """
  Kullback-Leibler divergence (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence)

  L(target, estimated) = target * log(target / estimated)
                       = target * (log(target) - log(estimated)

  :param target: probs, normalized. can also be sparse
  :param target_type: "probs", "log-probs" or "logits"
  :param estimated: probs, log-probs or logits, specified via ``estimated_type``
  :param estimated_type: "probs", "log-probs" or "logits"
  :param axis: the axis to reduce over
  :return: KL-div
  """
  if not axis:
    assert target.feature_dim
    axis = target.feature_dim

  if target.data.sparse:
    raise NotImplementedError(f"Sparse target {target} not supported for KL. Use cross entropy instead?")
  if target_type == "probs":
    log_target = nn.safe_log(target)
  elif estimated_type == "log-probs":
    log_target = target
  elif estimated_type == "logits":
    log_target = nn.log_softmax(target, axis=axis)
  else:
    raise ValueError("target_kind must be 'probs', 'log-probs' or 'logits'")

  if estimated_type == "probs":
    log_est = nn.safe_log(estimated)
  elif estimated_type == "log-probs":
    log_est = estimated
  elif estimated_type == "logits":
    log_est = nn.log_softmax(estimated, axis=axis)
  else:
    raise ValueError("estimated_kind must be 'probs', 'log-probs' or 'logits'")

  # Assuming target = softmax(...):
  # Using nn.exp(log_target) instead of target (but not nn.safe_exp!)
  # to avoid calculating softmax twice (efficiency)
  # (because nn.safe_log(target) = log_softmax(...), so a separate softmax calculation).
  kl = nn.dot(nn.exp(log_target), log_target - log_est, reduce=axis)

  return kl


def mean_absolute_difference(a: nn.Tensor, b: nn.Tensor, *, axis: Optional[nn.Dim] = None) -> nn.Tensor:
  """
  Mean absolute difference, mean absolute error (MAE), or L1 loss between two tensors,
  i.e. mean_{axis}( abs(a - b) ), where axis is the feature dim by default.
  """
  if not axis:
    assert a.feature_dim
    axis = a.feature_dim
  return nn.reduce(nn.abs(a - b), mode="mean", axis=axis)


def mean_squared_difference(a: nn.Tensor, b: nn.Tensor, *, axis: Optional[nn.Dim] = None) -> nn.Tensor:
  """
  Mean squared difference between two tensors,
  i.e. mean_{axis}( (a - b) ** 2 ), where axis is the feature dim by default.
  """
  if not axis:
    assert a.feature_dim
    axis = a.feature_dim
  return nn.reduce(nn.squared_difference(a, b), mode="mean", axis=axis)
