"""
Datasets common interfaces
"""

from __future__ import annotations
from typing import Dict, Optional, Any


class DatasetConfig:
    """
    Base class to be used to define a dataset (`train`, `dev` (for cross-validation and learning rate scheduling) etc)
    and `extern_data` for RETURNN.
    For an example instance `dataset`,
    you might do this in your RETURNN config::

      globals().update(dataset.get_config_opts())
    """

    def __repr__(self):
        parts = []
        if self.get_main_name():
            parts.append(f"main_name={self.get_main_name()}")
            ds = self.get_main_dataset()
        else:
            ds = self.get_train_dataset()
        parts.insert(0, f"class={ds['class']}")
        return f"<{self.__class__.__name__} {' '.join(parts)}>"

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Get extern data
        """
        raise NotImplementedError

    def get_train_dataset(self) -> Dict[str, Any]:
        """
        Get train dataset
        """
        raise NotImplementedError

    def get_train_dataset_for_forward(self) -> Dict[str, Any]:
        """
        Get train dataset for forward/eval (usually disables perturbations, enables sorting, no partition_epoch).
        """
        raise NotImplementedError

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        """
        :return: e.g. {"dev": ..., "devtrain": ...}
        This is intended for eval_datasets in the RETURNN config,
        which is used for cross-validation and learning rate scheduling.
        """
        raise NotImplementedError

    def get_main_name(self) -> str:
        """name of main dataset"""
        raise NotImplementedError

    def get_main_dataset(self) -> Dict[str, Any]:
        """
        More generic function, when this API is used for other purpose than training,
        e.g. recognition, generating alignment, collecting some statistics, etc,
        on one specific dataset.
        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def get_default_input(self) -> Optional[str]:
        """
        What is the default input data key of the dataset.
        (If that is meaningful.) (Must be in extern data.)
        """
        return "data"

    # noinspection PyMethodMayBeStatic
    def get_default_target(self) -> Optional[str]:
        """
        What is the default target key of the dataset.
        (If that is meaningful.) (Must be in extern data.)
        """
        return "classes"

    def get_config_opts(self) -> Dict[str, Any]:
        """
        E.g. in your main config, you could do::

          globals().update(dataset.get_config_opts())
        """
        return {
            "extern_data": self.get_extern_data(),
            "train": self.get_train_dataset(),
            "eval_datasets": self.get_eval_datasets(),
            "default_input": self.get_default_input(),
            "target": self.get_default_target(),
        }

    def copy_as_static(self, *, with_main_dataset: bool = True) -> DatasetConfigStatic:
        """
        Static copy
        """
        return DatasetConfigStatic(
            main_name=self.get_main_name(),
            main_dataset=self.get_main_dataset() if with_main_dataset else None,
            extern_data=self.get_extern_data(),
            default_input=self.get_default_input(),
            default_target=self.get_default_target(),
            train_dataset=self.get_train_dataset(),
            eval_datasets=self.get_eval_datasets(),
        )

    def copy_train_as_static(self, name: str = "train") -> DatasetConfigStatic:
        """
        Static copy of train dataset (via :func:`get_train_dataset_for_forward`),
        putting train as the main dataset.
        This provides train for forward or other jobs.
        """
        return DatasetConfigStatic(
            main_name=name,
            main_dataset=self.get_train_dataset_for_forward(),
            extern_data=self.get_extern_data(),
            default_input=self.get_default_input(),
            default_target=self.get_default_target(),
        )


class DatasetConfigStatic(DatasetConfig):
    """
    Static copy
    """

    def __init__(
        self,
        *,
        main_name: Optional[str] = None,
        main_dataset: Optional[Dict[str, Any]] = None,
        extern_data: Dict[str, Dict[str, Any]],
        default_input: Optional[str] = None,
        default_target: Optional[str] = None,
        train_dataset: Optional[Dict[str, Any]] = None,
        eval_datasets: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.main_name = main_name
        self.main_dataset = main_dataset
        self.extern_data = extern_data
        self.default_input = default_input
        self.default_target = default_target
        self.train_dataset = train_dataset
        self.eval_datasets = eval_datasets

    def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
        return self.extern_data

    def get_default_input(self) -> Optional[str]:
        return self.default_input

    def get_default_target(self) -> Optional[str]:
        return self.default_target

    def get_train_dataset(self) -> Dict[str, Any]:
        assert self.train_dataset is not None, "train dataset not defined"
        return self.train_dataset.copy()

    def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
        assert self.eval_datasets is not None, "eval datasets not defined"
        return {k: v.copy() for k, v in self.eval_datasets.items()}

    def get_main_name(self) -> str:
        assert self.main_name is not None, "main name not defined"
        return self.main_name

    def get_main_dataset(self) -> Dict[str, Any]:
        assert self.main_dataset is not None, "main dataset not defined"
        return self.main_dataset.copy()


class VocabConfig:
    """
    Defines a vocabulary, and esp also number of classes.
    See :func:`VocabConfigStatic.from_global_config` for a reasonable default.
    """

    def __repr__(self):
        return f"<{self.__class__.__name__} num_classes={self.get_num_classes()} eos_idx={self.get_eos_idx()}>"

    def get_num_classes(self) -> int:
        """
        Get num classes
        """
        raise NotImplementedError

    def get_opts(self) -> Dict[str, Any]:
        """
        Options for RETURNN vocab,
        e.g. as defined in `Data`, `extern_data`, :func:`Vocabulary.create_vocab` (in RETURNN).
        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def get_eos_idx(self) -> Optional[int]:
        """end-of-sequence (EOS)"""
        return None

    def get_bos_idx(self) -> Optional[int]:
        """beginning-of-sequence (BOS)"""
        return self.get_eos_idx()


class VocabConfigStatic(VocabConfig):
    """
    Static vocab (predefined num classes, vocab opts).
    """

    def __init__(self, *, num_classes: int, opts: Dict[str, Any]):
        super(VocabConfigStatic, self).__init__()
        self.num_classes = num_classes
        self.opts = opts

    @classmethod
    def from_global_config(cls, data_key: str) -> VocabConfigStatic:
        """
        Init from global config
        """
        from returnn.config import get_global_config

        config = get_global_config()
        extern_data_opts = config.typed_dict["extern_data"]
        data_opts = extern_data_opts[data_key]
        return VocabConfigStatic(num_classes=data_opts["dim"], opts=data_opts.get("vocab", {}))

    def get_num_classes(self) -> int:
        """
        Get num classes
        """
        return self.num_classes

    def get_opts(self) -> Dict[str, Any]:
        """
        Get opts
        """
        return self.opts


class TargetConfig:
    """
    Describes what target (data key in dataset & extern_data) to use.
    Used for models.
    """

    def __init__(self, key: str = None, *, vocab: VocabConfig = None):
        """
        Defaults will be received from the global config
        (`target` for `key`, or `extern_data` for `vocab`).
        """
        if not key:
            from returnn.config import get_global_config

            config = get_global_config()
            key = config.typed_dict["target"]
        if not vocab:
            vocab = VocabConfigStatic.from_global_config(key)
        self.vocab = vocab
        self.key = key

    @classmethod
    def global_from_config(cls) -> TargetConfig:
        """
        Construct global from config
        """
        # The default constructor with empty args will just return that.
        return TargetConfig()

    def get_num_classes(self) -> int:
        """
        Get num classes
        """
        return self.vocab.get_num_classes()
