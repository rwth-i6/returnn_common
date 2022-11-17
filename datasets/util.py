from typing import Optional, Union

_in_sisyphus_config = False

try:
  from sisyphus import tk
  from sisyphus.loader import config_manager
  FilePathType = Union[tk.Path, str]
  if config_manager.current_config is not None:
    _in_sisyphus_config = True
except ImportError:
  FilePathType = str


def is_in_sisyphus_config() -> bool:
  """
  :return: returns true if the module was imported from within a Sisyphus config
  """
  global _in_sisyphus_config
  return _in_sisyphus_config


def assert_path_type_sisyphus(var: Optional[FilePathType]):
  """
  checks if a path is a tk.Path during Sisyphus execution
  :param var: any file path variable
  """
  if var is not None and is_in_sisyphus_config():
    if not isinstance(var, tk.Path):
      raise TypeError("Please us tk.Path objects not strings as path definitions")
