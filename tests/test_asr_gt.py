"""
test asr.gt
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


def test_gt_gammatone_v2():
    from ..example_data import audio

    raw_audio, raw_audio_spatial_dim = audio.get_sample_batch()
    from ..asr import gt

    mod = gt.GammatoneV2()
    audio, audio_spatial_dim = mod(raw_audio, in_spatial_dim=raw_audio_spatial_dim)
    assert audio.data.placeholder is not None
    audio_, out_spatial_dim_ = gt.gammatone_v1(raw_audio, normalization=None)
    assert audio_.data.placeholder is not None
    assert (audio_.data.placeholder.numpy() == audio.data.placeholder.numpy()).all()  # noqa
