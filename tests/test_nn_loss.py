"""
Test losses
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, config_net_dict_via_serialized
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def _make_dummy_model_with_ce_out(name_ctx: nn.NameCtx) -> nn.Layer:
  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("input", 3)
  out_dim = nn.FeatureDim("out", 5)
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
  targets = nn.get_extern_data(nn.Data("classes", dim_tags=[nn.batch_dim, time_dim], sparse_dim=out_dim))

  linear = nn.Linear(out_dim)
  out = linear(data, name=name_ctx)
  loss = nn.cross_entropy(target=targets, estimated=out, estimated_type="logits")
  return loss


def test_cross_entropy():
  with nn.NameCtx.new_root() as name_ctx:
    loss = _make_dummy_model_with_ce_out(name_ctx)
    loss.mark_as_default_output()

  config_code = name_ctx.get_returnn_config_serialized()
  assert "sparse_softmax_cross_entropy_with_logits" in config_code
  config, net_dict = config_net_dict_via_serialized(config_code)
  dummy_run_net(config)


def test_mark_as_loss():
  with nn.NameCtx.new_root() as name_ctx:
    loss = _make_dummy_model_with_ce_out(name_ctx)
    loss.mark_as_loss()

  config_code = name_ctx.get_returnn_config_serialized()
  config, net_dict = config_net_dict_via_serialized(config_code)
  dummy_run_net(config, train=True)
