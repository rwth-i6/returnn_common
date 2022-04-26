"""
Simple utility to load (and maybe resample) the audio.
"""

from typing import Sequence, Tuple, List, Optional
import numpy


def get_sample_batch(files: Sequence[str], *,
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
    audios.append(load_audio(fn, sample_rate=sample_rate)[0])
  seq_lens = [len(a) for a in audios]
  out = numpy.zeros((batch_size, max(seq_lens)), dtype=numpy.float32)
  for i, a in enumerate(audios):
    out[i, :len(a)] = a
  return out, seq_lens


def load_audio(filename: str, *, sample_rate: Optional[int] = None) -> Tuple[numpy.ndarray, int]:
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
    audio = resample(audio, sample_rate_, sample_rate)
    sample_rate_ = sample_rate
  return audio, sample_rate_


def resample(audio: numpy.ndarray, in_sample_rate: int, out_sample_rate: int) -> numpy.ndarray:
  """
  resample
  """
  if in_sample_rate == out_sample_rate:
    return audio
  # https://stackoverflow.com/questions/29085268/resample-a-numpy-array/52347385#52347385
  import samplerate
  return samplerate.resample(audio, out_sample_rate / in_sample_rate, 'sinc_best')
