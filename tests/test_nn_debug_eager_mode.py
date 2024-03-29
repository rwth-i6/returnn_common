"""
debug eager mode test
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
import typing

if typing.TYPE_CHECKING:
    from .. import nn
else:
    from returnn_common import nn  # noqa


# Enables it globally now.
nn.enable_debug_eager_mode()


def test_simple_linear():
    """nn.Linear"""
    data = nn.get_extern_data(
        nn.Data("data_lin", dim_tags=[nn.batch_dim, nn.SpatialDim("time"), nn.FeatureDim("in", 5)])
    )
    assert data.data.placeholder is not None
    lin = nn.Linear(data.feature_dim, nn.FeatureDim("lin", 10))
    out = lin(data)
    assert lin.weight.data.placeholder is not None
    assert lin.bias.data.placeholder is not None
    assert out.data.placeholder is not None
    assert out.data.placeholder.numpy().size > 0
    assert (out.data.placeholder.numpy() != 0).any()


def test_conformer():
    """nn.Conformer"""
    time_dim = nn.SpatialDim("time")
    data = nn.get_extern_data(nn.Data("data_conformer", dim_tags=[nn.batch_dim, time_dim, nn.FeatureDim("in", 5)]))
    conformer = nn.ConformerEncoder(
        data.feature_dim,
        nn.FeatureDim("conformer", 10),
        num_layers=2,
        num_heads=2,
        input_layer=nn.ConformerConvSubsample(
            data.feature_dim,
            out_dims=[nn.FeatureDim("conv1", 32), nn.FeatureDim("conv2", 64)],
            filter_sizes=[(3, 3), (3, 3)],
            pool_sizes=[(2, 1), (2, 1)],
        ),
    )
    out, out_spatial_dim = conformer(data, in_spatial_dim=time_dim)
    assert out.data.placeholder is not None


def test_constant():
    res = nn.constant(value=42.0, shape=[nn.SpatialDim("custom-batch", 3)])
    assert res.data.placeholder is not None
    assert (res.data.placeholder.numpy() == 42.0).all()


def test_example_data_audio_get_sample_batch():
    from ..example_data import audio

    out, _ = audio.get_sample_batch()
    print("out:", out)
    assert out.data.placeholder is not None


def test_demo_asr_gt_wrapped_old_style():
    from ..example_data import audio

    raw_audio, _ = audio.get_sample_batch()
    from ..asr import gt

    audio, _ = gt.gammatone_v1(raw_audio)
    assert audio.data.placeholder is not None
