
"""
Datasets common interfaces
"""

from __future__ import annotations
from typing import Dict, Optional, Any
import dataclasses


@dataclasses.dataclass
class SupervisedTask:
  """
  Covers the training dataset and dev/eval etc. for recognition, including how to score it.
  This goes beyond :class:`DatasetConfig`, or rather covers multiple :class:`DatasetConfig`.

  It should be possible to replace Librispeech by Switchboard. Maybe even translation tasks later.

  Note that the dataset would also already include things like feature extraction details, output labels (BPE etc).
  """

  # for training
  train_dataset: DatasetConfig  # also includes cross-validation dataset for learning rate scheduling etc

  # for recognition
  dev_dataset: DatasetConfig  # used to select best epoch, maybe tune LM scale or so.
  eval_datasets: Dict[str, DatasetConfig]

  main_measure_type: MeasureType  # e.g. WER
  main_measure_name: str  # e.g. dataset name but arbitrary, just to describe the main measure value


@dataclasses.dataclass
class ScoreResult:
  """
  Intended to cover all relevant results.
  """
  main_number_description: str  # e.g. "test-other"
  main_number: MeasureValue  # e.g. the final best WER


@dataclasses.dataclass(frozen=True)
class MeasureType:
  short_name: str  # e.g. "WER"
  lower_is_better: bool = True


@dataclasses.dataclass(frozen=True)
class MeasureValue:
  type_: MeasureType
  value: float


class DatasetConfig:
  """
  Base class to be used to define a dataset (`train`, `dev` (for cross-validation and learning rate scheduling) etc)
  and `extern_data` for RETURNN.
  For an example instance `dataset`,
  you might do this in your RETURNN config::

    globals().update(dataset.get_config_opts())
  """

  def get_extern_data(self) -> Dict[str, Dict[str]]:
    """
    Get extern data
    """
    raise NotImplementedError

  def get_train_dataset(self) -> Dict[str]:
    """
    Get train dataset
    """
    raise NotImplementedError

  def get_eval_datasets(self) -> Dict[str, Dict[str]]:
    """
    :return: e.g. {"dev": ..., "devtrain": ...}
    This is intended for eval_datasets in the RETURNN config,
    which is used for cross-validation and learning rate scheduling.
    """
    raise NotImplementedError

  def get_main_dataset(self) -> Dict[str]:
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

  def get_config_opts(self) -> Dict[str]:
    """
    E.g. in your main config, you could do::

      globals().update(dataset.get_config_opts())
    """
    return {
      "extern_data": self.get_extern_data(),
      "train": self.get_train_dataset(),
      "eval_datasets": self.get_eval_datasets(),
      "target": self.get_default_target()}


class VocabConfig:
  """
  Defines a vocabulary, and esp also number of classes.
  See :func:`VocabConfigStatic.from_global_config` for a reasonable default.
  """

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
