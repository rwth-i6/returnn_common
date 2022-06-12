"""
test asr.specaugment
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
import typing
import os
from .returnn_helpers import dummy_run_net_single_custom, config_net_dict_via_serialized, make_scope

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
  code_str += "debug_runtime_sanity_checks = True\n\n"
  dummy_run_net_single_custom(code_str)


def test_specaugment_v2_real_example_audio():
  nn.reset_default_root_name_ctx()

  raw_audio_spatial_dim = nn.SpatialDim("time")
  raw_audio = nn.get_extern_data(nn.Data("raw_samples", dim_tags=[nn.batch_dim, raw_audio_spatial_dim]))

  from ..asr import gt
  gammatone = gt.GammatoneV2()
  audio, time_dim = gammatone(raw_audio, in_spatial_dim=raw_audio_spatial_dim)
  audio = nn.normalize(audio, axis=time_dim)
  audio_name = audio.mark_as_output().get_abs_name()

  from ..asr import specaugment
  masked = specaugment.specaugment_v2(
    audio, spatial_dim=time_dim, global_train_step_dependent=False, only_on_train=False)
  print(masked)
  masked.mark_as_default_output()

  code_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  code_str += "debug_runtime_sanity_checks = True\n\n"

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

  config_dict, net_dict = config_net_dict_via_serialized(code_str)
  from returnn.config import Config
  from returnn.tf.network import TFNetwork
  config = Config(config_dict)
  with make_scope() as session:
    net = TFNetwork(config=config, train_flag=False)
    net.construct_from_dict(net_dict)
    net.initialize_params(session)
    feed_dict = _make_feed_dict(net.extern_data)
    fetches = net.get_fetches_dict()
    for layer in net.get_output_layers():
      fetches[f"layer:{layer.name}"] = layer.output.placeholder

    def _eval():
      fetches1_np = session.run(fetches, feed_dict=feed_dict)
      fetches2_np = session.run(fetches, feed_dict=feed_dict)

      audio_np = fetches1_np[f"layer:{audio_name}"]
      print("audio shape:", audio_np.shape)
      masked_np = fetches1_np["layer:output"]
      print("masked shape:", masked_np.shape)
      assert audio_np.shape == masked_np.shape
      masked2_np = fetches2_np["layer:output"]
      print("masked2 shape:", masked2_np.shape)
      assert audio_np.shape == masked2_np.shape
      return audio_np, masked_np, masked2_np

    _eval()

    if "PYTEST_CURRENT_TEST" not in os.environ:
      try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
      except ImportError as exc:
        print("No matplotlib:", exc)
      else:

        def _plt_onclick(event=None):
          if event:
            event.canvas.figure.clear()

          audio_np, masked_np, masked2_np = _eval()

          plt.subplot(3, 1, 1)
          plt.imshow(audio_np[0, :300, :].T, origin="lower", vmin=-2, vmax=2)
          plt.subplot(3, 1, 2)
          plt.imshow(masked_np[0, :300, :].T, origin="lower", vmin=-2, vmax=2)
          plt.subplot(3, 1, 3)
          plt.imshow(masked2_np[0, :300, :].T, origin="lower", vmin=-2, vmax=2)

          if event:
            event.canvas.draw()

        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', _plt_onclick)
        _plt_onclick()
        plt.show()
