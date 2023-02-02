"""
Test nn.utils.
"""

from __future__ import annotations
from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net_single_custom
import typing

if typing.TYPE_CHECKING:
    from .. import nn
else:
    from returnn_common import nn  # noqa


def test_prev_target_seq():
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim], sparse_dim=in_dim, available_for_inference=True)
    x = nn.get_extern_data(x)
    out, dim = nn.prev_target_seq(x, spatial_dim=time_dim, bos_idx=0, out_one_longer=True)
    assert dim == 1 + time_dim
    out.mark_as_default_output()
    config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
    dummy_run_net_single_custom(config_str, eval_flag=True)


def test_register_call_post_hook():
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim])
    x = nn.get_extern_data(x)
    net = nn.Linear(in_dim, nn.FeatureDim("out", 5))

    _marker_attrib = "returnn_common_test_i_was_here"

    # noinspection PyShadowingBuiltins
    def _hook(func, input, output):
        print("hook called:", func, input, output)
        output = output * 2.0
        setattr(output, _marker_attrib, True)
        return output

    nn.register_call_post_hook(net, _hook)

    y = net(x)
    assert getattr(y, _marker_attrib) is True
    y.mark_as_default_output()
    config_str = nn.get_returnn_config().get_complete_py_code_str(net)
    dummy_run_net_single_custom(config_str, eval_flag=True)


def test_weight_norm():
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim])
    x = nn.get_extern_data(x)
    net = nn.Linear(in_dim, nn.FeatureDim("out", 5))
    assert isinstance(net.weight, nn.Parameter)
    nn.weight_norm(net, "weight", net.out_dim)
    assert not isinstance(net.weight, nn.Parameter) and isinstance(net.weight, nn.Tensor)
    y = net(x)
    y.mark_as_default_output()
    config_str = nn.get_returnn_config().get_complete_py_code_str(net)
    dummy_run_net_single_custom(config_str, eval_flag=True)


def test_variational_weight_noise():
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim])
    x = nn.get_extern_data(x)
    net = nn.Linear(in_dim, nn.FeatureDim("out", 5))
    assert isinstance(net.weight, nn.Parameter)
    nn.variational_weight_noise(net, "weight", 0.075)
    assert not isinstance(net.weight, nn.Parameter) and isinstance(net.weight, nn.Tensor)
    y = net(x)
    y.mark_as_default_output()
    config_str = nn.get_returnn_config().get_complete_py_code_str(net)
    dummy_run_net_single_custom(config_str, train_flag=True)


def test_weight_dropout():
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim])
    x = nn.get_extern_data(x)
    net = nn.Linear(in_dim, nn.FeatureDim("out", 5))
    assert isinstance(net.weight, nn.Parameter)
    nn.weight_dropout(net, "weight", 0.3)
    assert not isinstance(net.weight, nn.Parameter) and isinstance(net.weight, nn.Tensor)
    y = net(x)
    y.mark_as_default_output()
    config_str = nn.get_returnn_config().get_complete_py_code_str(net)
    dummy_run_net_single_custom(config_str, train_flag=True)


def test_random_frame_drop():
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)
    x = nn.get_extern_data(x)
    from returnn_common.nn.utils.augmentation import random_frame_drop

    out, dim = random_frame_drop(x, in_spatial_dim=time_dim, drop_prob=0.1)
    out.mark_as_default_output()
    config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
    dummy_run_net_single_custom(config_str, eval_flag=True)
