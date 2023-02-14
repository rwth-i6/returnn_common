"""
Test nn.cond
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import (
    dummy_run_net,
    dummy_config_net_dict,
    dummy_run_net_single_custom,
    config_net_dict_via_serialized,
    dummy_default_in_dim,
    make_feed_dict,
)

import typing
import functools

if typing.TYPE_CHECKING:
    from .. import nn
else:
    from returnn_common import nn  # noqa


def test_cond():
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            out_dim = nn.FeatureDim("linear-out", 13)
            self.linear_true = nn.Linear(dummy_default_in_dim, out_dim)
            self.linear_false = nn.Linear(dummy_default_in_dim, out_dim)

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
                cond.true = self.linear_true(x)
                cond.false = self.linear_false(x)
                x = cond.result
            return x

    config, net_dict, net = dummy_config_net_dict(_Net)
    dummy_run_net(config, net=net)


def test_cond_shared_params():
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dummy_default_in_dim, nn.FeatureDim("linear-out", 13))

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
                cond.true = self.linear(x)
                cond.false = self.linear(x * 2.0)
                x = cond.result
            return x

    config, net_dict, net = dummy_config_net_dict(_Net)
    engine = dummy_run_net(config, net=net)
    params = engine.network.get_params_list()
    print(params)
    assert len(params) == 2
    assert params[0].name == "linear/bias/param:0"


def test_cond_twice_shared_params():
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            out_dim = nn.FeatureDim("linear-out", 13)
            self.pre_linear = nn.Linear(dummy_default_in_dim, out_dim)
            self.linear_true = nn.Linear(out_dim, out_dim)
            self.linear_false = nn.Linear(out_dim, out_dim)

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            x = self.pre_linear(x)
            with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
                cond.true = self.linear_true(x)
                cond.false = self.linear_false(x)
                x = cond.result
            with nn.Cond(nn.length(nn.batch_dim) % 2 == 1) as cond:
                cond.true = self.linear_true(x)
                cond.false = self.linear_false(x)
                x = cond.result
            return x

    config, net_dict, net = dummy_config_net_dict(_Net)
    dummy_run_net(config, net=net)


def test_cond_random():
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnd = nn.Random()

        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
                cond.true = x + self.rnd.normal(x.shape_ordered)
                cond.false = x
                x = cond.result
            return x

    config, net_dict, net = dummy_config_net_dict(_Net)
    dummy_run_net(config, net=net)


def test_cond_new_axis():
    # Like in SelfAttention.
    nn.reset_default_root_name_ctx()
    in_dim = nn.FeatureDim("in", 12)
    time_dim = nn.SpatialDim("time")
    x = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
    net = nn.Linear(in_dim, in_dim)
    axis = time_dim

    with nn.Cond(nn.dim_value(nn.batch_dim) % 2 == 0) as cond:
        x_ = x
        x_ = net(x_)
        new_dim = nn.SpatialDim(f"{axis.description}:new-dim")
        x_, _ = nn.replace_dim(x_, in_dim=axis, out_dim=new_dim)
        x_ = net(x_)
        cond.true = nn.reduce(x_, axis=new_dim, mode="max")
        cond.false = nn.reduce(x, axis=axis, mode="max")
    y = cond.result
    y.mark_as_default_output()

    config_str = nn.get_returnn_config().get_complete_py_code_str(net)
    dummy_run_net_single_custom(config_str)


def test_cond_dim():
    nn.reset_default_root_name_ctx()
    in_dim = nn.FeatureDim("in", 12)
    time_dim = nn.SpatialDim("time")
    spatial_dim = time_dim
    x = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
    out_spatial_dim = spatial_dim - 1 + spatial_dim
    clipping = 2
    clipped_spatial_dim = nn.SpatialDim(f"learned-rel-pos", dimension=2 * clipping + 1)
    mat_spatial_size = clipping + 1
    pos_emb = nn.random_uniform((clipped_spatial_dim, in_dim), maxval=1.0)

    # Example via LearnedRelativePositionalEncoding.
    with nn.Cond(nn.dim_value(spatial_dim) > mat_spatial_size) as cond:
        # True branch
        left = nn.gather(pos_emb, axis=clipped_spatial_dim, position=0)
        right = nn.gather(pos_emb, axis=clipped_spatial_dim, position=clipped_spatial_dim.dimension - 1)
        remaining_dim = spatial_dim - mat_spatial_size
        left = nn.expand_dim(left, dim=remaining_dim)
        right = nn.expand_dim(right, dim=remaining_dim)
        concat, out_spatial_dim_ = nn.concat(
            (left, remaining_dim), (pos_emb, clipped_spatial_dim), (right, remaining_dim)
        )
        concat, out_spatial_dim_ = nn.replace_dim(concat, in_dim=out_spatial_dim_, out_dim=out_spatial_dim)
        cond.true = concat

        # False branch, spatial_dim <= self.clipping
        cond.false, _ = nn.slice_nd(
            pos_emb,
            axis=clipped_spatial_dim,
            start=mat_spatial_size - nn.dim_value(spatial_dim),
            size=out_spatial_dim,
        )

    y = cond.result
    y = y + nn.reduce(x, axis=(time_dim, nn.batch_dim), mode="mean")
    y.mark_as_default_output()

    config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
    dummy_run_net_single_custom(config_str, make_feed_dict=functools.partial(make_feed_dict, n_time=1))


def test_cond_multiple_outputs():
    class _Net(nn.Module):
        def __call__(self, x: nn.Tensor) -> nn.Tensor:
            y1 = x * 0.5
            y2 = x * 0.7 - 0.2
            with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
                cond.true = (y1, y2)
                cond.false = (y2, y1)
                x1, x2 = cond.result
            return x1 - x2

    config, net_dict, net = dummy_config_net_dict(_Net)
    dummy_run_net(config, net=net)


def test_cond_chunking_conformer():
    # The test says "chunking conformer" but we reduced it as much as possible
    # while still reproducing the issue, and there is no chunking and also no conformer anymore.
    # However, the nn.Cond still seems to be very relevant for the issue.
    # But also, the nn.Loop later seems relevant.
    from typing import Tuple

    class Encoder(nn.Module):
        """encoder"""

        def __init__(self, in_dim: nn.Dim):
            super(Encoder, self).__init__()
            self.self_att = nn.RelPosSelfAttention(
                in_dim, proj_dim=None, key_dim_total=nn.FeatureDim("keys", 2), value_dim_total=in_dim, num_heads=1
            )
            self.out_dim = in_dim

        def __call__(self, x: nn.Tensor, *, in_spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
            x = self.self_att(x, axis=in_spatial_dim)
            return x, in_spatial_dim

    class Model(nn.Module):
        """Model definition"""

        def __init__(
            self,
            in_dim: nn.Dim,
            *,
            nb_target_dim: nn.Dim,
            wb_target_dim: nn.Dim,
            blank_idx: int,
        ):
            super(Model, self).__init__()
            self.in_dim = in_dim
            self.encoder = Encoder(in_dim)

            self.nb_target_dim = nb_target_dim
            self.wb_target_dim = wb_target_dim
            self.blank_idx = blank_idx

            self.out_label_logits = nn.Linear(self.encoder.out_dim, wb_target_dim)

        def encode(
            self,
            source: nn.Tensor,
            *,
            in_spatial_dim: nn.Dim,
        ) -> Tuple[nn.Tensor, nn.Dim]:
            """encode, and extend the encoder output for things we need in the decoder"""
            enc, enc_spatial_dim = source, in_spatial_dim
            with nn.Cond(nn.train_flag()) as cond:
                enc_, _ = self.encoder(enc, in_spatial_dim=enc_spatial_dim)
                cond.true = enc_
                enc_, _ = self.encoder(enc, in_spatial_dim=enc_spatial_dim)
                cond.false = enc_
            enc = cond.result
            return enc, enc_spatial_dim

        def decode(
            self,
            enc: nn.Tensor,  # single frame if axis is single step, or sequence otherwise ("am" before)
        ) -> nn.Tensor:
            """decoder step, or operating on full seq"""
            logits = self.out_label_logits(enc)
            return logits

    # noinspection PyShadowingNames
    def model_recog(
        *,
        model: Model,
        data: nn.Tensor,
        data_spatial_dim: nn.Dim,
        targets_dim: nn.Dim,  # noqa
    ) -> nn.Tensor:
        """
        Function is run within RETURNN.

        Earlier we used the generic beam_search function,
        but now we just directly perform the search here,
        as this is overall simpler and shorter.

        :return: recog results including beam
        """
        batch_dims = data.batch_dims_ordered((data_spatial_dim, data.feature_dim))
        enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
        beam_size = 3

        loop = nn.Loop(axis=enc_spatial_dim)  # time-sync transducer
        loop.max_seq_len = nn.dim_value(enc_spatial_dim) * 2  # WHAT? this triggers it?
        loop.state.target = nn.constant(model.blank_idx, shape=batch_dims, sparse_dim=model.wb_target_dim)
        with loop:
            enc = loop.unstack(enc)
            logits = model.decode(enc)
            log_prob = nn.log_softmax(logits, axis=model.wb_target_dim)
            loop.state.target = nn.choice(
                log_prob,
                input_type="log_prob",
                target=None,
                search=True,
                beam_size=beam_size,
                length_normalization=False,
            )
            res = loop.stack(loop.state.target)
        return res

    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    input_dim = nn.FeatureDim("input", 10)
    label_dim = nn.FeatureDim("labels", 5)
    data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))

    model = Model(
        input_dim,
        nb_target_dim=label_dim,
        wb_target_dim=label_dim + 1,
        blank_idx=label_dim.dimension,
    )
    model_recog(model=model, data=data, data_spatial_dim=time_dim, targets_dim=label_dim).mark_as_default_output()

    # config = nn.get_returnn_config().get_config_raw_dict(root_module=model)
    config_code = nn.get_returnn_config().get_complete_py_code_str(model)
    config, net_dict = config_net_dict_via_serialized(config_code)
    dummy_run_net(config, net=model)
