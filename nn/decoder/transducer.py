"""
Transducer

TODO
  this is supposed to be more specific than what we currently have started to implement in base.
  i think having this more specific might actually be easier.
  this is just as incomplete and work-in-progress though.
  in the end, the question is, what interface do we really need, or what do we actually need the interface for.
  each config should define the search in some way, and the training.
  but it's actually up to the setup to put the pieces together,
  so we should just provide the building blocks here.

"""

from __future__ import annotations
from abc import ABC
from ... import nn


class ITransducerDecoderBase:
  """
  Generic transducer interface
  """
  def __init__(self):
    pass

  def next_align_frame_log_prob(self, *,
                                prev_align_label: nn.Tensor,
                                state: nn.LayerState
                                ) -> (nn.Tensor, nn.LayerState):
    """
    :return: probs over align label (incl blank), new state

    This is used for the search.
    """
    raise NotImplementedError


class ITransducerDecoderFullSum(ITransducerDecoderBase, ABC):
  """
  Covers the standard transducer, where full-sum over all possible alignments is feasible.
  """

  def full_sum_label_seq_score(self, *,
                               encoder: nn.Tensor,
                               encoder_spatial_dim: nn.Dim,
                               labels: nn.Tensor,
                               labels_spatial_dim: nn.Dim
                               ) -> nn.Tensor:
    """
    :param encoder: encoder sequence, shape like [B,T,D]
    :param encoder_spatial_dim:
    :param labels: non-blank labels, shape like [B,S]
    :param labels_spatial_dim:
    :return: label seq score, shape like [B]

    Internally, in the standard case, this would calculate a tensor [B,T,S,C] with C num classes,
    and then perform some efficient full-sum implementation on that.
    """
    raise NotImplementedError


def search_transducer(decoder: ITransducerDecoderBase, *,
                      encoder: nn.Tensor,
                      encoder_spatial_dim: nn.Dim
                      ) -> (nn.Tensor, nn.Tensor):
  """
  Perform search. Return
  """
  decoder, encoder, encoder_spatial_dim  # noqa  # TODO
  # TODO...
