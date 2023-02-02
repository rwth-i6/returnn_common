"""
Test nn._generated_layers
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_run_net_single_custom
import typing

if typing.TYPE_CHECKING:
    from .. import nn
else:
    from returnn_common import nn  # noqa


def test_merge_dims():
    nn.reset_default_root_name_ctx()
    time = nn.SpatialDim("time")
    spatial_static = nn.SpatialDim("extra-spatial", 10)
    feat = nn.FeatureDim("feat", 3)
    data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time, feat, spatial_static]))
    out, _ = nn.merge_dims(data, axes=[spatial_static, feat], out_dim=nn.FeatureDim("out", None))
    out.verify_out_shape({nn.batch_dim, time, out.feature_dim})
    out.mark_as_default_output()
    config = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
    dummy_run_net_single_custom(config)


def test_range_from_length():
    # https://github.com/rwth-i6/returnn_common/issues/134
    class _Net(nn.Module):
        def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
            durations = nn.cast(nn.reduce(x, axis=x.feature_dim, mode="sum"), dtype="int32")
            durations.verify_out_shape({nn.batch_dim, axis})
            t, _ = nn.range_from_length(durations)
            return t

    config, net_dict, net = dummy_config_net_dict(_Net, with_axis=True)
    dummy_run_net(config, net=net)


def test_repeat_int():
    # https://github.com/rwth-i6/returnn_common/issues/162
    nn.reset_default_root_name_ctx()
    time = nn.SpatialDim("time")
    data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time]))
    rep, rep_dim = nn.repeat(data, repetitions=5, axis=time)
    rep.mark_as_default_output()
    config = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
    dummy_run_net_single_custom(config)


def test_repeat_layer():
    # https://github.com/rwth-i6/returnn_common/issues/163
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, nn.FeatureDim("F", 3)]))
    const = nn.constant(value=5, shape=[nn.batch_dim, time_dim])
    nn.repeat(data, repetitions=const, axis=time_dim)[0].mark_as_default_output()
    config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
    dummy_run_net_single_custom(config_str)


def test_repeat_without_out_dim():
    # https://github.com/rwth-i6/returnn_common/issues/200
    from returnn.config import Config
    from returnn.tf.engine import Engine
    from returnn.datasets import init_dataset

    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    out_dim = nn.FeatureDim("out", 5)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)

    def _config_get_network(epoch: int, **_kwargs) -> dict:
        # noinspection PyStatementEffect
        epoch  # unused
        nn.reset_default_root_name_ctx()
        net = nn.Linear(in_dim, out_dim)
        y = net(nn.get_extern_data(x))
        out, dim = nn.repeat(y, repetitions=2, axis=time_dim)
        out.mark_as_default_output()
        y.mark_as_loss("y")
        net_dict = nn.get_returnn_config().get_net_dict_raw_dict(nn.Module())
        return net_dict

    config = Config(
        {
            "task": "train",
            "num_epochs": 1,
            "start_epoch": 1,
            "get_network": _config_get_network,
            "extern_data": {x.name: {"dim_tags": [nn.batch_dim, time_dim, in_dim], "available_for_inference": True}},
        }
    )
    train_dataset = init_dataset(
        {"class": "DummyDataset", "input_dim": in_dim.dimension, "output_dim": 5, "num_seqs": 3}
    )
    engine = Engine(config)
    engine.init_train_from_config(config, train_data=train_dataset)
    engine.train()


def test_repeat_without_out_dim_same_as_input():
    # https://github.com/rwth-i6/returnn_common/issues/200
    from returnn.config import Config
    from returnn.tf.engine import Engine
    from returnn.datasets import init_dataset

    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    out_dim = nn.FeatureDim("out", 5)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)

    def _config_get_network(epoch: int, **_kwargs) -> dict:
        # noinspection PyStatementEffect
        epoch  # unused
        nn.reset_default_root_name_ctx()
        net = nn.Linear(in_dim, out_dim)
        y = net(nn.get_extern_data(x))
        out, dim = nn.repeat(y, repetitions=1, axis=time_dim)
        out.mark_as_default_output()
        y.mark_as_loss("y")
        net_dict = nn.get_returnn_config().get_net_dict_raw_dict(nn.Module())
        return net_dict

    config = Config(
        {
            "task": "train",
            "num_epochs": 1,
            "start_epoch": 1,
            "get_network": _config_get_network,
            "extern_data": {x.name: {"dim_tags": [nn.batch_dim, time_dim, in_dim], "available_for_inference": True}},
        }
    )
    train_dataset = init_dataset(
        {"class": "DummyDataset", "input_dim": in_dim.dimension, "output_dim": 5, "num_seqs": 3}
    )
    engine = Engine(config)
    engine.init_train_from_config(config, train_data=train_dataset)
    net_out = engine.network.get_default_output_layer().output
    assert net_out.dim_tags == (nn.batch_dim, time_dim, out_dim)
    assert "time" in repr(net_out.dim_tags[1])
    assert "time" in repr(engine.network.extern_data.get_default_input_data().get_time_dim_tag())
    engine.train()


def test_repeat_as_loss():
    # https://github.com/rwth-i6/returnn_common/issues/201
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)
    out, dim = nn.repeat(nn.get_extern_data(x), repetitions=1, axis=time_dim, out_dim=time_dim)
    out.mark_as_default_output()
    out.mark_as_loss("y")
    config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
    dummy_run_net_single_custom(config_str, eval_flag=True)


def test_resize_dim_length():
    from returnn.config import Config
    from returnn.tf.engine import Engine
    from returnn.datasets import init_dataset

    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    data = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim])

    # get_network here is relevant for the test,
    # the exception was not raised otherwise.
    def _config_get_network(epoch: int, **_kwargs) -> dict:
        # noinspection PyStatementEffect
        epoch  # unused
        nn.reset_default_root_name_ctx()
        x = nn.get_extern_data(data)
        rnd_scale = nn.constant(1.1)
        out, dim = nn.resize(x, axis=time_dim, factor=rnd_scale, kind="linear")
        size = nn.length(dim)
        size.mark_as_default_output()
        net = nn.Linear(in_dim, in_dim)  # just to have some trainable params, for the test
        out = net(out)
        out.mark_as_loss("dummy")
        net_dict = nn.get_returnn_config().get_net_dict_raw_dict(net)
        return net_dict

    config = Config(
        {
            "task": "train",
            "num_epochs": 1,
            "start_epoch": 1,
            "get_network": _config_get_network,
            "extern_data": {data.name: {"dim_tags": [nn.batch_dim, time_dim, in_dim], "available_for_inference": True}},
        }
    )
    train_dataset = init_dataset(
        {"class": "DummyDataset", "input_dim": in_dim.dimension, "output_dim": 5, "num_seqs": 3}
    )
    engine = Engine(config)
    engine.init_train_from_config(config, train_data=train_dataset)
    engine.train()
