"""
Some generic utils (which doesn't fit into math_, array_, etc)
"""

from .ctc import *
from .dims import *
from .dropout import *
from .gradient import *
from .hooks import register_call_post_hook
from .label_smoothing import *
from .stochastic_depth import *
from .switchout import *
from .targets import *
from .variational_weight_noise import *
from .weight_dropout import *
from .weight_norm import weight_norm, remove_weight_norm
