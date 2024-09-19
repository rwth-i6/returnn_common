"""
NltkTimitDataset in RETURNN automatically downloads the data via `nltk`,
so no preparation is needed.
This is useful for demos/tests.
Note that this is only a subset of the official TIMIT corpus.
See :class:`NltkTimitDataset` for more details.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from returnn.config import get_global_config

from ...interface import DatasetConfig


config = get_global_config()

num_outputs = {"data": (40 * 2, 2), "classes": (61, 1)}
num_inputs = num_outputs["data"][0]
_num_seqs = {"train": 144, "dev": 16}


class NltkTimit(DatasetConfig):
    """
    NLTK TIMIT Dataset
    """

    def __init__(self, *, audio_dim=50, debug_mode=None, main_key: Optional[str] = None):
        super(NltkTimit, self).__init__()
        if debug_mode is None:
            debug_mode = config.typed_dict.get("debug_mode", False)
        self.audio_dim = audio_dim
        self.debug_mode = debug_mode
        self.main_key = main_key

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get extern data
        """
        from returnn.tensor import Dim, batch_dim

        feature_dim = Dim(self.audio_dim, name="audio", kind=Dim.Types.Feature)
        classes_dim = Dim(61, name="phonemes", kind=Dim.Types.Feature)
        time_dim = Dim(None, name="time", kind=Dim.Types.Spatial)
        return {
            "data": {"dim_tags": [batch_dim, time_dim, feature_dim]},
            "classes": {"dim_tags": [batch_dim, time_dim], "sparse_dim": classes_dim},
        }

    def get_train_dataset(self) -> Dict[str, Any]:
        """
        Get train dataset
        """
        return self.get_dataset("train", train=True)

    def get_train_dataset_for_forward(self) -> Dict[str, Any]:
        return self.get_dataset("train")

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get eval datasets_old_2022_10
        """
        return {"dev": self.get_dataset("dev"), "devtrain": self.get_dataset("train")}

    def get_main_name(self) -> str:
        """main name"""
        assert self.main_key, "main key not defined"
        return self.main_key

    def get_main_dataset(self) -> Dict[str, Any]:
        """main dataset"""
        assert self.main_key, "main key not defined"
        return self.get_dataset(self.main_key)

    def get_dataset(self, key, *, subset=None, train: bool = False):
        """
        Get datasets_old_2022_10
        """
        assert key in {"train", "dev"}
        assert not subset
        return {
            "class": "NltkTimitDataset",
            "train": (key == "train"),
            "seq_ordering": "default" if self.debug_mode and train else "laplace:.10" if train else "sorted_reverse",
            "fixed_random_seed": None if train else 1,
            "estimated_num_seqs": _num_seqs[key],
            "num_feature_filters": self.audio_dim,
        }
