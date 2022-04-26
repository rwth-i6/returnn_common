"""
Simple utility to load (and maybe resample) the audio.
"""

from ... import nn
from typing import Sequence, Tuple, List, Optional
import numpy


def get_sample_batch(files: Sequence[str], *,
                     batch_size: int = 2,
                     sample_rate: int = 16_000
                     ) -> Tuple[nn.Tensor, nn.Dim]:
  """
  sample batch.

  This is currently only intended and implemented for debug-eager-mode.

  :return: samples of shape [B,samples], and out spatial dim
  """
  assert nn.is_debug_eager_mode_enabled()
  import tensorflow as tf
  audio_np, seq_lens = get_sample_batch_np(files, batch_size=batch_size, sample_rate=sample_rate)
  batch_dim = nn.SpatialDim("B", batch_size)
  out_spatial_dim = nn.SpatialDim("samples")
  out_spatial_dim.dyn_size_ext = nn.Data(
    name="seq_lens", dim_tags=[batch_dim], dtype=nn.Data.size_dtype,
    placeholder=tf.convert_to_tensor(seq_lens))
  audio = nn.constant(value=0., shape=[batch_dim, out_spatial_dim], dtype="float32")
  audio.data.placeholder = tf.convert_to_tensor(audio_np)
  return audio, out_spatial_dim


def get_sample_batch_np(files: Sequence[str], *,
                        batch_size: int = 2,
                        sample_rate: int = 16_000
                        ) -> Tuple[numpy.ndarray, List[int]]:
  """
  sample batch

  :return: samples of shape [B,samples], and list of lengths [B]
  """
  assert files
  files = list(files)
  while batch_size > len(files):
    files *= 2
  if batch_size < len(files):
    files = files[:batch_size]
  audios = []
  for fn in files:
    audios.append(load_audio_np(fn, sample_rate=sample_rate)[0])
  seq_lens = [len(a) for a in audios]
  out = numpy.zeros((batch_size, max(seq_lens)), dtype=numpy.float32)
  for i, a in enumerate(audios):
    out[i, :len(a)] = a
  return out, seq_lens


def load_audio_np(filename: str, *, sample_rate: Optional[int] = None) -> Tuple[numpy.ndarray, int]:
  """
  load audio

  :param filename:
  :param sample_rate: if given, will resample using :func:`resample`
  :return: raw samples (1D numpy array) and sample rate
  """
  # Don't use librosa.load which internally uses audioread which would use Gstreamer as a backend,
  # which has multiple issues:
  # https://github.com/beetbox/audioread/issues/62
  # https://github.com/beetbox/audioread/issues/63
  # Instead, use PySoundFile, which is also faster. See here for discussions:
  # https://github.com/beetbox/audioread/issues/64
  # https://github.com/librosa/librosa/issues/681
  import soundfile  # noqa  # pip install pysoundfile
  audio, sample_rate_ = soundfile.read(filename)
  if sample_rate is not None:
    audio = resample_np(audio, sample_rate_, sample_rate)
    sample_rate_ = sample_rate
  return audio, sample_rate_


def resample_np(audio: numpy.ndarray, in_sample_rate: int, out_sample_rate: int) -> numpy.ndarray:
  """
  resample
  """
  if in_sample_rate == out_sample_rate:
    return audio
  # https://stackoverflow.com/questions/29085268/resample-a-numpy-array/52347385#52347385
  import samplerate  # noqa  # pip install samplerate
  return samplerate.resample(audio, out_sample_rate / in_sample_rate, 'sinc_best')
