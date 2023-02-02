"""
Base interface for any kind of encoder.

This is basically any generic function x -> y.

Note that in practice, when designing some model,
this interface is even not needed,
because you only care about the final encoded vectors,
and not how you got there.
Automatic differentiation will automatically
also train the encoder.
So, for most purpose, e.g. for a decoder (see :mod:`..decoder.base`),
you only care about some encoded vector of type :class:`Tensor`.
"""

from typing import Tuple, Union
from abc import ABC
from ... import nn


class IEncoder(nn.Module, ABC):
    """
    Generic encoder interface

    The encoder is a function x -> y.
    The input can potentially be sparse or dense.
    The output is dense with feature dim `out_dim`.
    """

    out_dim: nn.Dim

    def __call__(self, source: nn.Tensor) -> nn.Tensor:
        """
        Encode the input
        """
        raise NotImplementedError


class ISeqFramewiseEncoder(nn.Module, ABC):
    """
    This specializes IEncoder that it operates on a sequence.
    The output sequence length here is the same as the input.
    """

    out_dim: nn.Dim

    def __call__(self, source: nn.Tensor, *, spatial_dim: nn.Dim) -> nn.Tensor:
        raise NotImplementedError


class ISeqDownsamplingEncoder(nn.Module, ABC):
    """
    This is more specific than IEncoder in that it operates on a sequence.
    The output sequence length here is shorter than the input.

    This is a common scenario for speech recognition
    where the input might be on 10ms/frame
    and the output might cover 30ms/frame or 60ms/frame or so.
    """

    out_dim: nn.Dim
    # In most cases (pooling, conv), the output sequence length will bei ceildiv(input_seq_len, factor)
    # and factor is an integer.
    # However, this is not a hard condition.
    # The downsampling factor only describes the linear factor in the limit.
    downsample_factor: Union[int, float]

    def __call__(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
        raise NotImplementedError
