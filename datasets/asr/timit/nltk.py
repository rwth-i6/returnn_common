"""
NltkTimitDataset in RETURNN automatically downloads the data via `nltk`,
so no preparation is needed.
This is useful for demos/tests.
Note that this is only a subset of the official TIMIT corpus.
See :class:`NltkTimitDataset` for more details.
"""

from __future__ import annotations
from typing import Dict, Any
from returnn.config import get_global_config

from ...interface import DatasetConfig


config = get_global_config()

num_outputs = {'data': (40*2, 2), 'classes': (61, 1)}
num_inputs = num_outputs["data"][0]
_num_seqs = {'train': 144, 'dev': 16}


class NltkTimit(DatasetConfig):
  """
  NLTK TIMIT Dataset
  """
  def __init__(self, audio_dim=50, debug_mode=None):
    super(NltkTimit, self).__init__()
    if debug_mode is None:
      debug_mode = config.typed_dict.get("debug_mode", False)
    self.audio_dim = audio_dim
    self.debug_mode = debug_mode

  def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
    """
    Get extern data
    """
    return {
      "data": {"dim": self.audio_dim},
      "classes": {"sparse": True, "dim": 61},
    }

  def get_train_dataset(self) -> Dict[str, Any]:
    """
    Get train dataset
    """
    return self.get_dataset("train")

  def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
    """
    Get eval datasets
    """
    return {
      "dev": self.get_dataset("dev"),
      "devtrain": self.get_dataset("train")}

  def get_dataset(self, key, subset=None):
    """
    Get datasets
    """
    assert key in {"train", "dev"}
    assert not subset
    return {
      "class": "NltkTimitDataset",
      "train": (key == "train"),
      "seq_ordering": {
        "train": "default" if self.debug_mode else "laplace:.10",
        "dev": "sorted_reverse"}[key],
      "fixed_random_seed": {"train": None, "dev": 1}[key],
      "estimated_num_seqs": _num_seqs[key],
      "num_feature_filters": self.audio_dim,
    }
