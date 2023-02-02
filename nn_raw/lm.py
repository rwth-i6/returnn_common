"""
Language model functions.
"""

from ..datasets_old_2022_10.interface import VocabConfig
from typing import Dict, Any


class Lm:
    """
    Represents language model.
    """

    vocab: VocabConfig
    opts: Dict[str, Any]
    net_dict: Dict[str, Any]
    model_path: str
