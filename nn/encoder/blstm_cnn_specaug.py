"""
Common BLSTM-based encoder often used for end-to-end (attention, transducer) models:

SpecAugment . PreCNN . BLSTM
"""


from typing import Tuple

from ...asr.specaugment import specaugment_v2
from ... import nn
from .blstm_cnn import BlstmCnnEncoder


class BlstmCnnSpecAugEncoder(BlstmCnnEncoder):
    """
    SpecAugment . PreCNN . BLSTM
    """

    def __call__(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
        source = specaugment_v2(source, spatial_dim=in_spatial_dim)
        source, in_spatial_dim = super(BlstmCnnSpecAugEncoder, self).__call__(source, in_spatial_dim=in_spatial_dim)
        return source, in_spatial_dim
