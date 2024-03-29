"""
Test the base nn functionality
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import (
    dummy_run_net,
    dummy_run_net_single_custom,
    dummy_config_net_dict,
    dummy_default_in_dim,
    config_net_dict_via_serialized,
)
from pprint import pprint
from .utils import assert_equal
import numpy
import typing
from typing import Tuple

if typing.TYPE_CHECKING:
    from .. import nn
else:
    from returnn_common import nn  # noqa


def test_simple_linear():
    config, net_dict, net = dummy_config_net_dict(
        lambda: nn.Linear(dummy_default_in_dim, nn.FeatureDim("linear-out", 13))
    )
    pprint(config)
    dummy_run_net(config, net=net)


def test_simple_linear_ex():
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim])
    x = nn.get_extern_data(x)
    net = nn.Linear(in_dim, nn.FeatureDim("out", 5))
    y = net(x)
    y.mark_as_default_output()
    config_str = nn.get_returnn_config().get_complete_py_code_str(net)
    dummy_run_net_single_custom(config_str, eval_flag=True)


def test_simple_net_linear():
    class _Net(nn.Module):
        def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            """
            Forward
            """
            return self.linear(x)

    config, net_dict, net = dummy_config_net_dict(lambda: _Net(dummy_default_in_dim, nn.FeatureDim("linear-out", 13)))
    assert "linear" in net_dict
    pprint(config)
    dummy_run_net(config, net=net)


def test_simple_net_linear_square_matrix():
    # https://github.com/rwth-i6/returnn_common/issues/17
    # https://github.com/rwth-i6/returnn/pull/871
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            out_dim = nn.FeatureDim("linear-out", 13)
            self.linear = nn.Linear(dummy_default_in_dim, out_dim)
            self.linear2 = nn.Linear(out_dim, out_dim)

        def __call__(self, x) -> nn.Tensor:
            """
            Forward
            """
            x = self.linear(x)
            x = self.linear2(x)
            return x

    config, net_dict, net = dummy_config_net_dict(_Net)
    dummy_run_net(config, net=net)


def test_simple_net_arithmetic():
    class _Net(nn.Module):
        def __call__(self, x) -> nn.Tensor:
            """
            Forward
            """
            x = 1.0 / x + x * 2.0
            return x

    config, net_dict, net = dummy_config_net_dict(_Net)
    dummy_run_net(config)


def _functional_example(x: nn.Tensor) -> nn.Tensor:
    return nn.tanh((x + 1.0) * 0.5)


def test_functional_auto_name_ctx():
    class _Net(nn.Module):
        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            x -= 1.0
            x = _functional_example(x)
            x += 1.0
            x = _functional_example(x)
            return x

    config, net_dict, net = dummy_config_net_dict(_Net)
    assert "_functional_example" in net_dict
    assert_equal(net_dict["_functional_example"]["class"], "subnetwork")
    assert_equal(net_dict["_functional_example_0"]["class"], "subnetwork")
    assert "_functional_example_1" not in net_dict
    dummy_run_net(config)


def test_simple_net_share_params():
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = nn.FeatureDim("linear-out", 13)
            self.linear = nn.Linear(dummy_default_in_dim, self.dim)
            self.linear2 = nn.Linear(self.dim, self.dim)

        def __call__(self, x) -> nn.Tensor:
            """
            Forward
            """
            x = self.linear(x)
            x = self.linear2(x)
            x = self.linear2(x)
            return x

    config, net_dict, net = dummy_config_net_dict(_Net)
    assert "linear2" in net_dict
    dummy_run_net(config, net=net)


def test_explicit_root_ctx_sub():
    class _Net(nn.Module):
        # noinspection PyShadowingNames
        def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim, dropout=0.1):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.dropout = dropout

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            """
            forward
            """
            x = nn.dropout(x, self.dropout, axis=x.feature_dim, name="pre")
            x = self.linear(x)
            return x

    with nn.NameCtx.new_root() as name_ctx:
        net = _Net(dummy_default_in_dim, nn.FeatureDim("linear-out", 13))
        out = net(
            nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, nn.SpatialDim("time"), dummy_default_in_dim]))
        )
        assert isinstance(out, nn.Tensor)
        out.mark_as_default_output()

    config = name_ctx.get_returnn_config().get_config_raw_dict(net)
    net_dict = config["network"]
    pprint(net_dict)

    assert "linear" in net_dict
    assert "pre" in net_dict
    lin_layer_dict = net_dict["pre"]
    assert_equal(lin_layer_dict["class"], "dropout")
    assert_equal(lin_layer_dict["from"], "data:data")
    dummy_run_net(config)


def test_root_mod_call_twice():
    class TestBlock(nn.Module):
        """
        Test block
        """

        # noinspection PyShadowingNames
        def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim, dropout=0.1):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)
            self.dropout = dropout

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            """
            forward
            """
            x = nn.dropout(x, self.dropout, axis=x.feature_dim)
            x = self.linear(x)
            return x

    with nn.NameCtx.new_root() as name_ctx:
        in_dim = nn.FeatureDim("input", 5)
        test_block = TestBlock(in_dim, nn.FeatureDim("linear-out", 13))
        time_dim = nn.SpatialDim("time")
        y = test_block(nn.get_extern_data(nn.Data("input1", dim_tags=[nn.batch_dim, time_dim, in_dim])))
        z = test_block(nn.get_extern_data(nn.Data("input2", dim_tags=[nn.batch_dim, time_dim, in_dim])))

        print(y)
        assert isinstance(y, nn.Tensor)
        print(z)
        assert isinstance(z, nn.Tensor)
        y.mark_as_output()
        z.mark_as_output()

    config = name_ctx.get_returnn_config().get_config_raw_dict(test_block)
    net_dict = config["network"]
    pprint(net_dict)

    assert "linear" in net_dict


def test_multiple_returns_depth_1():
    class _SubNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dummy_default_in_dim, nn.FeatureDim("linear-out", 13))

        def __call__(self, x: nn.Tensor) -> Tuple[nn.Tensor, nn.Tensor]:
            """
            Forward
            """
            x = self.linear(x)
            return x, x

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _SubNet()

        def __call__(self, x) -> nn.Tensor:
            """
            Forward
            """
            out, add_out = self.sub(x)
            return out

    config, net_dict, net = dummy_config_net_dict(_Net)
    pprint(net_dict)
    assert net_dict["output"]["from"] == "sub"
    dummy_run_net(config)


def test_multiple_returns_depth_2():
    class _SubSubNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dummy_default_in_dim, nn.FeatureDim("linear-out", 13))

        def __call__(self, x: nn.Tensor) -> Tuple[nn.Tensor, nn.Tensor]:
            """
            Forward
            """
            x = self.linear(x)
            return x, x

    class _SubNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _SubSubNet()

        def __call__(self, x: nn.Tensor) -> Tuple[nn.Tensor, nn.Tensor]:
            """
            Forward
            """
            x, x_ = self.sub(x)
            return x, x_

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _SubNet()

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            """
            Forward
            """
            out, add_out = self.sub(x)
            return out

    config, net_dict, net = dummy_config_net_dict(_Net)
    pprint(net_dict)
    assert net_dict["output"]["from"] == "sub"
    assert net_dict["sub"]["subnetwork"]["output"]["from"] == "sub"
    assert (
        net_dict["sub"]["subnetwork"]["sub"]["subnetwork"]["linear"]["subnetwork"]["dot"]["from"][0]
        == "base:base:base:data:data"
    )
    dummy_run_net(config)


def test_from_call_variations():
    class _SubNet(nn.Module):
        def __init__(self, in_dim: nn.Dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, nn.FeatureDim("linear-out", 13))
            self.linear2 = nn.Linear(self.linear.out_dim, nn.FeatureDim("linear-out", 13))
            self.out_dim = self.linear2.out_dim

        def __call__(self, x: nn.Tensor) -> Tuple[nn.Tensor, nn.Tensor]:
            """
            Forward
            """
            x = self.linear(x)
            x = self.linear2(x)
            return x, x

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _SubNet(dummy_default_in_dim)
            self.sub2 = _SubNet(self.sub.out_dim)

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            """
            Forward
            """
            out, add_out = self.sub(x)
            out2, add_out2 = self.sub2(add_out)
            return out2

    config, net_dict, net = dummy_config_net_dict(_Net)
    pprint(net_dict)
    assert net_dict["output"]["from"] == "sub2"
    assert net_dict["sub"]["subnetwork"]["linear"]["subnetwork"]["dot"]["from"][0] == "base:base:data:data"
    assert net_dict["sub"]["subnetwork"]["linear2"]["subnetwork"]["dot"]["from"][0] == "base:linear"
    assert net_dict["sub2"]["subnetwork"]["linear"]["subnetwork"]["dot"]["from"][0] == "base:base:sub"
    assert net_dict["sub2"]["subnetwork"]["linear2"]["subnetwork"]["dot"]["from"][0] == "base:linear"
    dummy_run_net(config)


def test_from_call_variations2():
    class _SubNet(nn.Module):
        def __init__(self, in_dim: nn.Dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, nn.FeatureDim("linear-out", 13))
            self.linear2 = nn.Linear(self.linear.out_dim, nn.FeatureDim("linear-out", 13))
            self.out_dim = self.linear2.out_dim

        def __call__(self, x: nn.Tensor) -> Tuple[nn.Tensor, nn.Tensor]:
            """
            Forward
            """
            x_ = self.linear(x)
            x = self.linear2(x_)
            return x, x_

    class _SubNet2(nn.Module):
        def __init__(self, in_dim: nn.Dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, nn.FeatureDim("linear-out", 13))
            self.linear2 = nn.Linear(self.linear.out_dim, nn.FeatureDim("linear-out", 13))
            self.out_dim = self.linear2.out_dim

        def __call__(self, x: nn.Tensor, y: nn.Tensor) -> Tuple[nn.Tensor, nn.Tensor]:
            """
            Forward
            """
            x_ = self.linear(x)
            x = self.linear2(x_)
            return x, x_

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.sub = _SubNet(dummy_default_in_dim)
            self.linear = nn.Linear(self.sub.out_dim, nn.FeatureDim("linear-out", 13))
            self.sub2 = _SubNet2(self.sub.linear.out_dim)

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            """
            Forward
            """
            out, add_out = self.sub(x)
            lin = self.linear(out)
            out2, add_out2 = self.sub2(add_out, lin)
            return out2

    config, net_dict, net = dummy_config_net_dict(_Net)
    pprint(net_dict)


def test_get_name_in_current_ctx():
    def make_ctx(parent: nn.NameCtx = None, name: str = "", subnet=False):
        """
        helper that builds the different NameCtxs with correct attributes
        """
        if not parent:
            return nn.NameCtx.new_root()
        ctx = nn.NameCtx(parent=parent, name=name)
        if subnet:
            ctx.is_subnet = True
        return ctx

    root = make_ctx(name="root")
    sub_1 = make_ctx(parent=root, name="sub_1", subnet=True)
    same = make_ctx(parent=sub_1, name="same", subnet=True)
    child_1 = make_ctx(parent=same, name="child_1")
    sub_2 = make_ctx(parent=root, name="sub_2", subnet=True)
    child_2 = make_ctx(parent=sub_2, name="child_2")

    assert_equal(same.get_name_in_ctx(sub_1), "same")
    assert_equal(child_1.get_name_in_ctx(sub_1), "same/child_1")
    assert_equal(sub_2.get_name_in_ctx(sub_1), "base:sub_2")
    assert_equal(child_2.get_name_in_ctx(sub_1), "base:sub_2/child_2")


def test_deepcopy():
    import copy

    dims = [nn.FeatureDim(f"linear{i}-out", i + 3) for i in range(3)]
    config, net_dict, net = dummy_config_net_dict(
        lambda: nn.Sequential(
            copy.deepcopy(nn.Linear(in_dim, out_dim)) for in_dim, out_dim in zip([dummy_default_in_dim] + dims, dims)
        )
    )
    assert isinstance(net, nn.Sequential)
    lin1, lin2, lin3 = net
    assert isinstance(lin1, nn.Linear)
    assert (net, "0") in lin1._parents
    pprint(net_dict)
    assert "0" in net_dict
    assert_equal(net_dict["0"]["subnetwork"]["dot"]["from"][0], "base:data:data")
    dummy_run_net(config, net=net)


def test_deepcopy_same_linear():
    def _make_net():
        import copy

        lin = nn.Linear(dummy_default_in_dim, dummy_default_in_dim)
        return nn.Sequential(copy.deepcopy(lin) for _ in range(3))

    config, net_dict, net = dummy_config_net_dict(_make_net)
    assert isinstance(net, nn.Sequential)
    lin1, lin2, lin3 = net
    assert isinstance(lin1, nn.Linear)
    assert (net, "0") in lin1._parents
    pprint(net_dict)
    assert "0" in net_dict
    assert_equal(net_dict["0"]["subnetwork"]["dot"]["from"][0], "base:data:data")
    collected_var_names = set()
    for name, p in net.named_parameters():
        print(name, ":", p)
        collected_var_names.add(name.replace(".", "/"))
    assert_equal(collected_var_names, {"0/bias", "0/weight", "1/bias", "1/weight", "2/bias", "2/weight"})
    dummy_run_net(config, net=net)


def test_deepcopy_same_linear_diff_init():
    # https://github.com/rwth-i6/returnn_common/issues/216
    def _make_net():
        import copy

        lin = nn.Linear(dummy_default_in_dim, dummy_default_in_dim)
        return nn.Sequential(copy.deepcopy(lin) for _ in range(3))

    config, net_dict, net = dummy_config_net_dict(_make_net)
    assert isinstance(net, nn.Sequential)
    lin1, lin2, lin3 = net
    assert isinstance(lin1, nn.Linear)
    assert (net, "0") in lin1._parents
    pprint(net_dict)
    assert "0" in net_dict
    assert_equal(net_dict["0"]["subnetwork"]["dot"]["from"][0], "base:data:data")
    engine = dummy_run_net(config, net=net)
    w0 = engine.network.get_layer("0/weight").params["param"].eval(engine.tf_session)
    w1 = engine.network.get_layer("1/weight").params["param"].eval(engine.tf_session)
    assert isinstance(w0, numpy.ndarray) and isinstance(w1, numpy.ndarray)
    assert (w0 != 0.0).any()
    assert (w0 != w1).any()


def test_deepcopy_deep():
    class _Sub(nn.Module):
        def __init__(self):
            super(_Sub, self).__init__()
            self.lin = nn.Linear(dummy_default_in_dim, dummy_default_in_dim)

        def __call__(self, x):
            return self.lin(x)

    def _make_net():
        import copy

        sub = _Sub()
        assert (sub, "lin") in sub.lin._parents
        subs = [copy.deepcopy(sub) for _ in range(3)]
        sub0 = subs[0]
        assert (sub0, "lin") in sub0.lin._parents
        return nn.Sequential(subs)

    config, net_dict, net = dummy_config_net_dict(_make_net)
    assert isinstance(net, nn.Sequential)
    sub1, sub2, sub3 = net
    assert isinstance(sub1, _Sub)
    assert (net, "0") in sub1._parents
    assert (sub1, "lin") in sub1.lin._parents
    pprint(net_dict)
    assert "0" in net_dict
    dummy_run_net(config, net=net)


def test_variable():
    # https://github.com/rwth-i6/returnn_common/issues/141
    class _Net(nn.Module):
        def __init__(self, dim: nn.Dim):
            super().__init__()
            self.variable = nn.Parameter(shape=[dim])

        def __call__(self, x: nn.Tensor):
            return x + self.variable

    nn.reset_default_root_name_ctx()
    feat_dim = nn.FeatureDim("feature", 5)
    time_dim = nn.SpatialDim("time")
    inputs = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, feat_dim]))
    net = _Net(feat_dim)
    out = net(inputs)
    out.mark_as_default_output()
    config_code = nn.get_returnn_config().get_complete_py_code_str(net)
    config, net_dict = config_net_dict_via_serialized(config_code)
    dummy_run_net(config)


def test_parameter_weight_decay():
    nn.reset_default_root_name_ctx()
    feat_dim = nn.FeatureDim("feature", 5)
    time_dim = nn.SpatialDim("time")
    inputs = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, feat_dim]))

    net = nn.Linear(feat_dim, nn.FeatureDim("out", 4))
    out = net(inputs)
    out.mark_as_default_output()

    params = []
    for param in net.parameters():
        params.append(param)
        print("param:", param)
        param.weight_decay = 1.1
    assert len(params) == 2  # weight + bias

    config = nn.get_returnn_config().get_complete_py_code_str(net)
    config, net_dict = config_net_dict_via_serialized(config)
    assert "L2" in net_dict["weight"] and net_dict["weight"]["L2"] == 1.1
    dummy_run_net(config)


def test_const_array_serialization():
    class _Net(nn.Module):
        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            import numpy

            c = nn.constant(
                value=numpy.linspace(0.0, 10.0, x.feature_dim.dimension, dtype=numpy.float32), shape=[x.feature_dim]
            )
            return x + c

    config, net_dict, net = dummy_config_net_dict(_Net)
    dummy_run_net(config)


def test_asr_specaug_v1_eval_func_serialization():
    class _Net(nn.Module):
        def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
            # specaugment_v1 uses a custom eval layer with a ref to a Python function,
            # which is defined in that module.
            # Thus, extra care needs to be taken when serializing the config.
            from ..asr.specaugment import specaugment_v1

            return specaugment_v1(x)

    config, net_dict, net = dummy_config_net_dict(_Net, with_axis=True)
    dummy_run_net(config)


def test_returnn_config_direct_construction():
    # https://github.com/rwth-i6/returnn/issues/1069
    from returnn.config import Config
    from returnn.tf.engine import Engine
    from returnn.datasets import init_dataset

    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    out_dim = nn.FeatureDim("out", 5)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)

    def _config_get_network(epoch: int, **_kwargs) -> dict:
        print("_config_get_network called")  # it's called multiple times
        # noinspection PyStatementEffect
        epoch  # unused
        nn.reset_default_root_name_ctx()
        net = nn.Linear(in_dim, out_dim)
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
    engine.train()


def test_param_name_deep():
    # Access params from another subnet. Do this here by sharing the params in _SubNet.
    # This caused NameCtx.get_name_in_ctx to break before.
    class _SubNet(nn.Module):
        def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def __call__(self, x) -> nn.Tensor:
            return self.linear(x)

    class _Net(nn.Module):
        def __init__(self, in_dim: nn.Dim):
            super().__init__()
            self.in_dim = in_dim
            self.out_dim = nn.FeatureDim("linear-out", 13)
            self.sub = _SubNet(self.in_dim, self.out_dim)
            self.sub2 = _SubNet(self.in_dim, self.out_dim)
            self.sub2.linear.weight = self.sub.linear.weight
            self.sub2.linear.bias = self.sub.linear.bias

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            return self.sub(x) + self.sub2(x)

    config, net_dict, net = dummy_config_net_dict(lambda: _Net(dummy_default_in_dim))
    pprint(config)
    dummy_run_net(config, net=net)


def test_mod_early_setattr():
    class _Net(nn.Module):
        def __init__(self):
            # before super().__init__()
            self.linear = nn.Linear(dummy_default_in_dim, nn.FeatureDim("linear-out", 13))
            super().__init__()

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            return self.linear(x)

    config, net_dict, net = dummy_config_net_dict(_Net)
    dummy_run_net(config, net=net)


def test_make_layer_subnet_deep():
    class _Net(nn.Module):
        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            # There used to be a bug in make_layer / _data_from_layer_dict,
            # where such construction with multiple sources to the input
            # would cause an exponential grow of construction calls by the number of layers,
            # so the following code would almost loop infinitely.
            num_layers = 100
            subnet = {
                "output": {"class": "copy", "from": f"deep{num_layers}"},
                "deep0": {"class": "copy", "from": "data"},
                "deep1": {"class": "copy", "from": "data"},
            }
            for i in range(1, num_layers):
                subnet[f"deep{i + 1}"] = {"class": "combine", "kind": "add", "from": [f"deep{i}", f"deep{i - 1}"]}
            return nn.make_layer({"class": "subnetwork", "from": x, "subnetwork": subnet}, name="subnet")

    config, net_dict, net = dummy_config_net_dict(_Net)
    dummy_run_net(config, net=net)


def test_linear_wrong_type():
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim])
    x = nn.get_extern_data(x)
    net = nn.Linear(in_dim, nn.FeatureDim("out", 5))
    # This is deliberately wrong. We test that we get an error.
    # If we would not get an error, RETURNN would later throw an error,
    # but the RETURNN error somewhat confusing without further inspection.
    try:
        y = net({"foo": x})  # noqa
    except TypeError as exc:
        print("Got expected TypeError:", exc)
    else:
        raise Exception("did not get expected TypeError")
