"""
Librispeech examples
"""

from .... import nn
from typing import Tuple, List
import numpy
import os

__my_dir__ = os.path.dirname(os.path.abspath(__file__))

files = [
  __my_dir__ + "/train-clean-100--6476-57446-0077.flac.ogg",
  __my_dir__ + "/train-other-500--7012-81370-0044.flac.ogg",
]


def get_sample_batch(batch_size: int = 2,
                     sample_rate: int = 16_000
                     ) -> Tuple[nn.Tensor, nn.Dim]:
  """
  load
  """
  from ..load_audio_file import get_sample_batch
  return get_sample_batch(files, batch_size=batch_size, sample_rate=sample_rate)


def get_sample_batch_np(batch_size: int = 2,
                        sample_rate: int = 16_000
                        ) -> Tuple[numpy.ndarray, List[int]]:
  """
  load
  """
  from ..load_audio_file import get_sample_batch_np
  return get_sample_batch_np(files, batch_size=batch_size, sample_rate=sample_rate)
