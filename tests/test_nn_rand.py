"""
Test nn.rand
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_run_net_single_custom
from pprint import pprint
import typing

if typing.TYPE_CHECKING:
    from .. import nn
else:
    from returnn_common import nn  # noqa


def test_random_normal():
    nn.reset_default_root_name_ctx()

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnd = nn.Random()

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            return x + self.rnd.normal(x.shape_ordered)

    config, net_dict, net = dummy_config_net_dict(_Net, reset_name_ctx=False)
    pprint(net_dict)
    dummy_run_net(config, net=net)


def test_random_multi_call():
    # https://github.com/rwth-i6/returnn_common/issues/148
    # Actually we will not really test the non-determinism as this is difficult to test.
    # We just test whether we can run it multiple times without error.
    nn.reset_default_root_name_ctx()

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnd = nn.Random()

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            return x + self.rnd.normal(x.shape_ordered) - self.rnd.normal(x.shape_ordered)

    config, net_dict, net = dummy_config_net_dict(_Net, reset_name_ctx=False)
    pprint(net_dict)
    dummy_run_net(config, net=net)


def test_random_in_loop():
    # https://github.com/rwth-i6/returnn_common/issues/184
    nn.reset_default_root_name_ctx()

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnd = nn.Random()

        def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
            loop = nn.Loop(axis=axis)
            with loop:
                x_ = loop.unstack(x)
                r_ = self.rnd.normal([nn.batch_dim, in_dim])
                r = loop.stack(x_ * 0.0 + r_)
            return r

    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("input", 3)
    data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
    net = _Net()
    out = net(data, axis=time_dim)
    out.mark_as_default_output()

    config_code = nn.get_returnn_config().get_complete_py_code_str(net)
    res = dummy_run_net_single_custom(config_code, default_out_dim_tag_order=("T", "B", "F"))
    out_np = res["layer:output"]
    print(out_np)
    out0_np = out_np[:1]  # [1,B,D]
    print((out0_np == out_np))
    assert not (out0_np == out_np).all()  # not all the same


def test_random_normal_shape_get_network():
    # https://github.com/rwth-i6/returnn_common/issues/197
    class _Net(nn.Module):
        def __call__(self, x_: nn.Tensor) -> nn.Tensor:
            return x_ + nn.random_normal(x_.shape_ordered)

    from returnn.config import Config
    from returnn.tf.engine import Engine

    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, in_dim], available_for_inference=True)

    def _config_get_network(epoch: int, **_kwargs) -> dict:
        # noinspection PyStatementEffect
        epoch  # unused
        nn.reset_default_root_name_ctx()
        net = _Net()
        # net = nn.Linear(out_dim)
        out = net(nn.get_extern_data(x))
        out.mark_as_default_output()
        out.mark_as_loss("dummy")
        net_dict = nn.get_returnn_config().get_net_dict_raw_dict(net)
        pprint(net_dict)
        return net_dict

    config = Config(
        {
            "task": "train",
            "num_epochs": 1,
            "start_epoch": 1,
            "get_network": _config_get_network,
            "extern_data": {x.name: {"dim_tags": [nn.batch_dim, in_dim], "available_for_inference": True}},
        }
    )
    engine = Engine(config)
    engine.init_train_from_config(config)


def test_random_normal_shape_get_network_with_time():
    # https://github.com/rwth-i6/returnn_common/issues/197
    class _Net(nn.Module):
        def __call__(self, x_: nn.Tensor) -> nn.Tensor:
            return x_ + nn.random_normal(x_.shape_ordered)

    from returnn.config import Config
    from returnn.tf.engine import Engine
    from returnn.datasets import init_dataset

    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)

    def _config_get_network(epoch: int, **_kwargs) -> dict:
        # noinspection PyStatementEffect
        epoch  # unused
        nn.reset_default_root_name_ctx()
        net = _Net()
        # net = nn.Linear(out_dim)
        out = net(nn.get_extern_data(x))
        out.mark_as_default_output()
        out.mark_as_loss("dummy")
        net_dict = nn.get_returnn_config().get_net_dict_raw_dict(net)
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
    res = engine.forward_single(dataset=train_dataset, seq_idx=0)
    print(res)


def test_random_normal_train_epoch():
    # https://github.com/rwth-i6/returnn_common/issues/198
    class _Net(nn.Module):
        def __init__(self):
            super(_Net, self).__init__()
            self.linear = nn.Linear(in_dim, in_dim)

        def __call__(self, x_: nn.Tensor) -> nn.Tensor:
            return self.linear(x_ + nn.random_normal(x_.shape_ordered)) + x_

    from returnn.config import Config
    from returnn.tf.engine import Engine
    from returnn.datasets import init_dataset

    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)

    def _config_get_network(epoch: int, **_kwargs) -> dict:
        # noinspection PyStatementEffect
        epoch  # unused
        nn.reset_default_root_name_ctx()
        net = _Net()
        # net = nn.Linear(out_dim)
        out = net(nn.get_extern_data(x))
        out.mark_as_default_output()
        out.mark_as_loss("dummy")
        net_dict = nn.get_returnn_config().get_net_dict_raw_dict(net)
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
    engine.train_epoch()
