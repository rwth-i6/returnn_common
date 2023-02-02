"""
Helpers to wrap TensorFlow (TF) code in returnn-common.
On RETURNN level, this is via :class:`EvalLayer`.
"""


from .. import nn
from typing import Tuple


def wrap_tf_function(func, *args: Tuple[nn.Tensor, nn.Dim]):
    """
    wrap TF function

    :return: function
    """
    func, args  # noqa  # TODO
    # TODO...
    # nn.make_layer({
    #   "class": "eval",
    #   "from": [log_probs, labels],
    #   # Pickling/serialization of the func ref should work when this is a global function of this module.
    #   # But depending on your setup, there might anyway not be any serialization.
    #   # TODO this should be serializable / pickleable...
    #   #  for that, it would need to be a global function in a module. or some other way, which needs to be defined...
    #   "eval": _transducer_full_sum_log_prob_eval_layer_func,
    #   "eval_locals": {
    #     "blank_index": blank_index,
    #     "input_spatial_dim": input_spatial_dim,
    #     "labels_spatial_dim": labels_spatial_dim,
    #   },
    #   "out_type": _transducer_full_sum_log_prob_eval_layer_out,
    # }, name=func.__name__)
