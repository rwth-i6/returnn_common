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

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, List
from .. import nn

if TYPE_CHECKING:
    import tensorflow as tf
    from returnn.tf.layers.basic import LayerBase


def cross_entropy(
    *, target: nn.Tensor, estimated: nn.Tensor, estimated_type: str, axis: Optional[nn.Dim] = None
) -> nn.Tensor:
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


def binary_cross_entropy(
    *,
    target: nn.Tensor,
    pos_estimated: nn.Tensor,
    pos_estimated_type: str,
    pos_weight: Optional[Union[float, nn.Tensor]] = None,
):
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
            f"binary_cross_entropy, pos_estimated_type {pos_estimated_type!r}, only 'logits' supported"
        )
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
        return (1 - target) * logits + log_weight * (nn.log1p(nn.exp(-nn.abs(logits))) + nn.relu(-logits))

    # Code adapted from tf.nn.sigmoid_cross_entropy_with_logits.
    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.
    cond = logits >= 0
    relu_logits = nn.where(cond, logits, 0)
    neg_abs_logits = nn.where(cond, -logits, logits)  # pylint: disable=invalid-unary-operand-type
    return relu_logits - logits * target + nn.log1p(nn.exp(neg_abs_logits))


def kl_div(
    *, target: nn.Tensor, target_type: str, estimated: nn.Tensor, estimated_type: str, axis: Optional[nn.Dim] = None
) -> nn.Tensor:
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


def transducer_time_sync_full_sum_neg_log_prob(
    log_probs: nn.Tensor,
    *,
    labels: nn.Tensor,
    input_spatial_dim: nn.Dim,
    labels_spatial_dim: nn.Dim,
    blank_index: int = -1,
) -> nn.Tensor:
    """
    Computes the RNA loss between a sequence of activations and a
    ground truth labeling.

    https://github.com/rwth-i6/warp-rna

    Args:
        log_probs: A >=3-D Tensor of floats.  The dimensions
                     should be (B..., T, U, V), where B is the minibatch index,
                     T is the time index, U is the prediction network sequence
                     length, and V indexes over activations for each
                     symbol in the alphabet.
        labels: A >=1-D Tensor of ints, shape (B...,U) a padded label sequences to make sure
                     labels for the minibatch are same length.
        input_spatial_dim:
        labels_spatial_dim:
        blank_index: int, the label value/index that the RNA
                     calculation should use as the blank label
    Returns:
        >=0-D float Tensor, shape (B...), the cost of each example in the minibatch
        (as negative log probabilities).
    """
    assert isinstance(log_probs, nn.Tensor) and isinstance(labels, nn.Tensor)
    assert isinstance(input_spatial_dim, nn.Dim) and isinstance(labels_spatial_dim, nn.Dim)
    assert isinstance(blank_index, int)
    return nn.make_layer(
        {
            "class": "eval",
            "from": [log_probs, labels],
            # Pickling/serialization of the func ref should work when this is a global function of this module.
            # But depending on your setup, there might anyway not be any serialization.
            "eval": _transducer_full_sum_log_prob_eval_layer_func,
            "eval_locals": {
                "blank_index": blank_index,
                "input_spatial_dim": input_spatial_dim,
                "labels_spatial_dim": labels_spatial_dim,
            },
            "out_type": _transducer_full_sum_log_prob_eval_layer_out,
        },
        name="transducer_time_sync_full_sum_neg_log_prob",
    )


def _transducer_full_sum_log_prob_eval_layer_func(
    *,
    self: LayerBase,
    source,
    input_spatial_dim: nn.Dim,
    labels_spatial_dim: nn.Dim,
    blank_index: int,
) -> tf.Tensor:
    from returnn.tf.layers.basic import LayerBase

    assert isinstance(self, LayerBase)
    log_probs = source(0, auto_convert=False, as_data=True)
    labels = source(1, auto_convert=False, as_data=True)
    assert isinstance(log_probs, nn.Data) and isinstance(labels, nn.Data)
    batch_dims = list(self.output.dim_tags)
    feat_dim = log_probs.feature_dim_or_sparse_dim
    if blank_index < 0:
        blank_index += feat_dim.dimension
    assert 0 <= blank_index < feat_dim.dimension
    assert labels.sparse_dim.dimension <= feat_dim.dimension
    # Move axes into the right order (no-op if they already are).
    log_probs = log_probs.copy_compatible_to(
        nn.Data("log_probs", dim_tags=batch_dims + [input_spatial_dim, 1 + labels_spatial_dim, feat_dim]),
        check_dtype=False,
    )
    labels = labels.copy_compatible_to(
        nn.Data("labels", dim_tags=batch_dims + [labels_spatial_dim], sparse_dim=labels.sparse_dim), check_dtype=False
    )
    input_lengths = input_spatial_dim.get_dyn_size_ext_for_batch_ctx(
        log_probs.batch, log_probs.control_flow_ctx
    ).copy_compatible_to(nn.Data("input_lengths", dim_tags=batch_dims), check_dtype=False)
    label_lengths = labels_spatial_dim.get_dyn_size_ext_for_batch_ctx(
        log_probs.batch, log_probs.control_flow_ctx
    ).copy_compatible_to(nn.Data("label_lengths", dim_tags=batch_dims), check_dtype=False)
    from returnn.extern.WarpRna import rna_loss  # noqa

    return rna_loss(
        log_probs=log_probs.placeholder,
        labels=labels.placeholder,
        input_lengths=input_lengths.placeholder,
        label_lengths=label_lengths.placeholder,
        blank_label=blank_index,
    )


def _transducer_full_sum_log_prob_eval_layer_out(
    *, name: str, sources: List[LayerBase], input_spatial_dim: nn.Dim, labels_spatial_dim: nn.Dim, **_kwargs
) -> nn.Data:
    from returnn.tf.layers.basic import LayerBase

    log_probs, labels = sources
    assert isinstance(log_probs, LayerBase) and isinstance(labels, LayerBase)
    dim_tags = list(log_probs.output.dim_tags)
    # Remove all dims used here -- batch dim(s) remain.
    dim_tags.remove(log_probs.output.feature_dim_or_sparse_dim)
    dim_tags.remove(input_spatial_dim)
    dim_tags.remove(1 + labels_spatial_dim)
    assert set(dim_tags + [labels_spatial_dim]) == set(labels.output.dim_tags)  # same batch dims
    return nn.Data("%s_output" % name, dim_tags=dim_tags)
