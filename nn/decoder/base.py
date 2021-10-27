"""
Generic decoder interface.

This is supposed to cover the decoder of an attention-based encoder-decoder and of a transducer.

See :class:`Decoder`.

References:

  See ../transducer/transducer_fullsum.py.

TODO this is all work-in-progress. the transducer-fullsum was the base for this code,
  but then it got extended for the new base interfaces (Module, Rec),
  and generalized for attention-encoder-decoder models,
  and the interfaces and argument names were generalized.
"""

from __future__ import annotations
from typing import Optional
from enum import Enum
from ..base import Module, LayerRef


class LabelTopology(Enum):
  """
  Possible label topologies
  """
  LABEL_SYNC = 1
  TIME_SYNC_PEAKY = 2
  TIME_SYNC_LABEL_LOOP = 3
  WITH_VERTICAL = 4


class Decoder(Module):
  """
  Generic decoder, for attention-based encoder-decoder or transducer.
  Can use label-sync label topology, or time-sync (RNA/CTC), or with vertical transitions (RNN-T).
  The label emitted in the current step is referred to as alignment-label (or step-label),
  and can include blank in case this is not label-sync.

  None of this is really enforced here, and what mainly defines the interfaces
  is the dependency graph.
  The returned shapes and time axes could be anything,
  as long as it fits together.
  The step_sync_rnn could also return a 4D tensor with both time-axis and label-axis.

  Dependency graph:

    prev:output_nb_label, (encoder), (prev:step_sync_rnn) -> label_sync_rnn                 # runs label-sync
    (label_sync_rnn), encoder, (prev:align_label) -> step_sync_rnn
    step_sync_rnn -> log_prob_separate_nb                             # runs label-sync
    step_sync_rnn, log_prob_separate_nb -> log_prob_separate_wb       # runs step-sync

  In case of attention-encoder-decoder model (label-sync), or in general label-sync decoder:

  # TODO...
  """

  def __init__(self, *,
               label_topology: LabelTopology,
               label_sync_rnn: IDecoderLabelSyncRnn = None,  # earlier: slow_rnn
               step_sync_rnn: IDecoderStepSyncRnn = None,  # earlier: fast_rnn
               log_prob_separate_nb: IDecoderLogProbSeparateNb = None,
               log_prob_separate_wb: Optional[IDecoderLogProbSeparateWb] = None,
               ):
    """
    :param label_sync_rnn: runs label-sync. earlier slow_rnn
    :param step_sync_rnn: runs step-sync (align-sync, time-sync or label-sync). earlier fast_rnn
    :param log_prob_separate_nb:
    :param log_prob_separate_wb: runs step-sync. might output blank.
      combines log_prob_separate_nb with sigmoid decision whether to output a label or blank.
    """
    super().__init__()
    self.label_topology = label_topology
    self.label_sync_rnn = label_sync_rnn  # earlier: slow_rnn
    self.step_sync_rnn = step_sync_rnn  # earlier: fast_rnn
    self.log_prob_separate_nb = log_prob_separate_nb
    self.log_prob_separate_wb = log_prob_separate_wb

  def forward(self, encoder: LayerRef) -> LayerRef:
    """
    Make one decoder step (train and/or recognition).
    """
    # TODO ...


# TODO enc ctx module
# TODO make generic type for args (Generic, TypeVar) to support not just single layer?


class IDecoderLabelSyncRnn(Module):
  """
  Represents SlowRNN in Transducer.
  """
  def forward(self, *,
              prev_sparse_label_nb: LayerRef,
              prev_emit: LayerRef,
              unmasked_sparse_label_nb_seq: Optional[LayerRef] = None,
              prev_step_sync_rnn: LayerRef,
              encoder: LayerRef  # TODO enc ctx?
              ) -> LayerRef:
    """
    Make layer dict.
    """
    raise NotImplementedError


class IDecoderStepSyncRnn(Module):
  """
  Represents FastRNN in Transducer.
  Otherwise in general this runs step-synchronous,
  which is alignment-synchronous or time-synchronous for RNN-T/RNA/CTC,
  or label-synchronous for att-enc-dec.
  """
  def forward(self, *,
              prev_label_wb: LayerRef,
              encoder: LayerRef,  # TODO enc ctx. or not? need full encoder for full-sum case...
              label_sync_rnn: LayerRef) -> LayerRef:
    """
    prev_label_wb and encoder use the same time dim (T) (or none).
    label_sync_rnn can use the same (unmasked) (or none) or a different (U+1) (maybe in full-sum setting).
    When label_sync_rnn has a different time dim than prev_label_wb/encoder,
    we expect 2 time dims as output (Tx(U+1)).
    """
    raise NotImplementedError


class IDecoderLogProbSeparateNb(Module):
  """
  Log prob separate without blank.
  """
  def forward(self, step_sync_rnn: LayerRef) -> LayerRef:
    """
    Make log-prob distribution over labels (without blank).

    :param step_sync_rnn: might be the FastRNN in transducer, or SlowRNN in att-enc-dec model
    """
    raise NotImplementedError


class IDecoderLogProbSeparateWb(Module):
  """
  Log prob with blank.
  """
  def forward(self, step_sync_rnn: LayerRef, log_prob_nb: LayerRef) -> LayerRef:
    """
    Make layer dict.
    """
    raise NotImplementedError
