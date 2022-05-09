"""
test asr.specaugment
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
import typing
from .returnn_helpers import dummy_run_net_single_custom

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_specaugment_v2():
  nn.reset_default_root_name_ctx()

  feat_dim = nn.FeatureDim("feat", 50)
  time_dim = nn.SpatialDim("time")
  audio = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, feat_dim]))

  from ..asr import specaugment
  masked = specaugment.specaugment_v2(
    audio, spatial_dim=time_dim, global_train_step_dependent=False, only_on_train=False)
  print(masked)
  masked.mark_as_default_output()

  code_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  print(code_str)
  dummy_run_net_single_custom(code_str)


def test_specaugment_v2_real_example_audio():
  nn.reset_default_root_name_ctx()

  raw_audio_spatial_dim = nn.SpatialDim("time")
  raw_audio = nn.get_extern_data(nn.Data("raw_samples", dim_tags=[nn.batch_dim, raw_audio_spatial_dim]))

  from ..asr import gt
  gammatone = gt.GammatoneV2()
  audio, time_dim = gammatone(raw_audio, in_spatial_dim=raw_audio_spatial_dim)

  for param in gammatone.parameters():
    param.initial = 0.  # just to have shorter net config for this test  # TODO remove this

  from ..asr import specaugment
  masked = specaugment.specaugment_v2(
    audio, spatial_dim=time_dim, global_train_step_dependent=False, only_on_train=False)
  print(masked)
  masked.mark_as_default_output()

  code_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  print(code_str)

  from ..example_data import audio
  raw_audio_np, raw_audio_seq_lens = audio.get_sample_batch_np()

  def _make_feed_dict(extern_data):
    from returnn.tf.network import ExternData
    assert isinstance(extern_data, ExternData)
    data = extern_data.data["raw_samples"]
    return {
      data.placeholder: raw_audio_np,
      data.get_sequence_lengths(): raw_audio_seq_lens,
    }

  dummy_run_net_single_custom(code_str, make_feed_dict=_make_feed_dict)
