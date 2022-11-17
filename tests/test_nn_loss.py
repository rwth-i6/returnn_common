"""
Test losses
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, config_net_dict_via_serialized
import typing
from typing import Tuple

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def _make_dummy_model_with_ce_out() -> Tuple[nn.Module, nn.Tensor]:
  nn.auto_setup_name_ctx_ignore_func(_make_dummy_model_with_ce_out)
  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("input", 3)
  out_dim = nn.FeatureDim("out", 5)
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
  targets = nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, time_dim], sparse_dim=out_dim))

  linear = nn.Linear(in_dim, out_dim)
  out = linear(data)
  loss = nn.cross_entropy(target=targets, estimated=out, estimated_type="logits")
  return linear, loss


def test_cross_entropy():
  nn.reset_default_root_name_ctx()
  mod, loss = _make_dummy_model_with_ce_out()
  loss.mark_as_default_output()

  config_code = nn.get_returnn_config().get_complete_py_code_str(mod)
  assert "sparse_softmax_cross_entropy_with_logits" in config_code
  config, net_dict = config_net_dict_via_serialized(config_code)
  dummy_run_net(config)


def test_mark_as_loss():
  nn.reset_default_root_name_ctx()
  mod, loss = _make_dummy_model_with_ce_out()
  loss.mark_as_loss("ce")

  config_code = nn.get_returnn_config().get_complete_py_code_str(mod)
  config, net_dict = config_net_dict_via_serialized(config_code)
  dummy_run_net(config, train=True)


def _functional_mark_as_loss(x: nn.Tensor):
  x *= 2.
  x = nn.minimum(x, 10.)
  x.mark_as_loss("x")


def test_mark_as_loss_in_subnet():
  nn.reset_default_root_name_ctx()
  mod, loss = _make_dummy_model_with_ce_out()
  _functional_mark_as_loss(loss)

  config_code = nn.get_returnn_config().get_complete_py_code_str(mod)
  config, net_dict = config_net_dict_via_serialized(config_code)
  assert net_dict["_functional_mark_as_loss"]["class"] == "subnetwork"
  dummy_run_net(config, train=True)


def test_transducer_time_sync_full_sum_neg_log_prob():
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("input", 2)
  hidden_dim = nn.FeatureDim("hidden", 7)
  classes_dim = nn.FeatureDim("classes", 3)
  out_spatial_dim = nn.SpatialDim("out_spatial")
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim], sparse_dim=in_dim))
  targets = nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, out_spatial_dim], sparse_dim=classes_dim))

  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.am = nn.Linear(in_dim, hidden_dim)
      self.lm = nn.Linear(classes_dim, hidden_dim)
      self.logits = nn.Linear(hidden_dim, classes_dim + 1)

  def _log_fwdbwd(source, **_kwargs):
    import tensorflow as tf
    from returnn.tf.util import basic as tf_util

    x = source(0)
    assert isinstance(x, tf.Tensor)

    # noinspection PyUnusedLocal
    def _custom_grad(op, grad):
      with tf.control_dependencies([tf.print("log_probs grad", grad)]):
        return tf.identity(grad)

    grad_name = "%s_with_grad" % x.op.name.replace("/", "__")
    tf_util.opt_register_grad_func(
      op_type=grad_name,
      grad_func=_custom_grad,
      assert_is_same=False)

    g = x.graph
    with g.gradient_override_map({"Identity": grad_name}):
      with tf.control_dependencies([tf.print("log_probs", x)]):
        y = tf.identity(x)

    return y

  net = _Net()
  am = net.am(data)
  prev_targets, prev_targets_spatial_dim = nn.prev_target_seq(
    targets, spatial_dim=out_spatial_dim, bos_idx=0, out_one_longer=True)
  lm = net.lm(prev_targets)
  logits = net.logits(nn.combine_bc(am, "+", lm))
  log_probs = nn.log_softmax(logits, axis=logits.feature_dim)
  log_probs = nn.make_layer({"class": "eval", "eval": _log_fwdbwd, "from": log_probs}, name="log_probs")
  log_probs = nn.label_smoothed_log_prob_gradient(log_probs, 0.1)
  loss = nn.transducer_time_sync_full_sum_neg_log_prob(
    log_probs=log_probs, labels=targets, input_spatial_dim=time_dim, labels_spatial_dim=out_spatial_dim)
  loss.mark_as_loss("full_sum")

  config = nn.get_returnn_config().get_config_raw_dict(net)
  config.update(dict(
    optimizer={"class": "adam"},
    learning_rate=0.001,
    train={
      "class": "TaskNumberBaseConvertDataset",
      "input_base": in_dim.dimension, "output_base": classes_dim.dimension,
      "num_seqs": 2}
  ))
  dummy_run_net(config, net=net, train=True)
