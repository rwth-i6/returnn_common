"""
Extra functions
"""

from __future__ import annotations
from typing import Optional, Union

_in_sisyphus_config = False

try:
    from sisyphus import tk
    from sisyphus.loader import config_manager

    FilePathType = Union[tk.Path, str]
except ImportError:
    FilePathType = str
    tk = None
    config_manager = None


def is_in_sisyphus_config() -> bool:
    """
    :return: returns true if the module was imported from within a Sisyphus config
    """
    if config_manager and config_manager.current_config is not None:  # noqa
        return True
    return False


def assert_path_type_sisyphus(var: Optional[FilePathType]):
    """
    checks if a path is a tk.Path during Sisyphus execution
    :param var: any file path variable
    """
    if var is not None and is_in_sisyphus_config():
        if not isinstance(var, tk.Path):  # noqa
            raise TypeError("Please use tk.Path objects not strings as Dataset path definitions")
