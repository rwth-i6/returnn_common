"""
Generic decoder interface.

This is supposed to cover the decoder of an attention-based encoder-decoder and of a transducer.

See :class:`Decoder`.

References:

  See ../transducer/transducer_fullsum.py.
  https://github.com/rwth-i6/returnn_common/issues/49

TODO this is all work-in-progress. the transducer-fullsum was the base for this code,
  but then it got extended for the new base interfaces (Module, Rec),
  and generalized for attention-encoder-decoder models,
  and the interfaces and argument names were generalized.
"""

from __future__ import annotations
from typing import Union, Optional, Sequence, Tuple
from enum import Enum
import dataclasses
from ... import nn


class LabelTopology(Enum):
  """
  Possible label topologies
  """
  LABEL_SYNC = 1
  TIME_SYNC_PEAKY = 2
  TIME_SYNC_LABEL_LOOP = 3
  WITH_VERTICAL = 4


class Decoder(nn.Module):
  """
  Generic decoder, for attention-based encoder-decoder or transducer.
  Can use label-sync label topology, or time-sync (RNA/CTC), or with vertical transitions (RNN-T).
  The label emitted in the current (align) step is referred to as alignment-label (or step-label),
  and can include blank in case this is not label-sync.

  None of this is really enforced here, and what mainly defines the interfaces
  is the dependency graph.
  The returned shapes and time axes could be anything,
  as long as it fits together.
  The predictor could also return a 4D tensor with both time-axis and label-axis.

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
               label_predict_enc: Optional[TDecoderLabelSync],
               predictor: TDecoderJointNetLogProb,
               target_dim: nn.Dim,
               target_bos_symbol: int = 0,
               target_eos_symbol: int = 0,
               ):
    super().__init__()
    self.label_topology = label_topology
    self.label_predict_enc = label_predict_enc  # earlier: slow_rnn. label-sync. incl (nb) label embedding
    self.predictor = predictor  # earlier: fast_rnn + readout. align-sync or matrix time * label. predicts align label
    self.target_dim = target_dim  # includes blank if not label-sync
    self.target_bos_symbol = target_bos_symbol
    self.target_eos_symbol = target_eos_symbol

  def __call__(self, *,
               encoder: nn.Tensor,
               encoder_spatial_axis: nn.Dim,
               target: Optional[Union[nn.Tensor, nn.SearchFuncInterface]] = None,
               axis: Optional[nn.Dim] = None,
               state: Optional[nn.LayerState] = None,
               ) -> Tuple[nn.Tensor, nn.Dim, nn.LayerState]:
    """
    Make one decoder step (train and/or recognition).
    """
    batch_dims = list(encoder.shape_ordered)
    batch_dims.remove(encoder_spatial_axis)
    batch_dims.remove(encoder.feature_dim)

    # TODO ...
    search = None
    if isinstance(target, nn.SearchFuncInterface):
      search = target
      target = None
    if target is not None:
      assert axis, f"{self}: Target spatial axis must be specified when target is given"

    loop = nn.Loop(axis=axis)
    loop.state = state if state else self.default_initial_state(batch_dims=batch_dims)
    with loop:

      if self.label_predict_enc is None:
        label_predict_enc = None
      elif isinstance(self.label_predict_enc, IDecoderLabelSyncRnn):
        label_predict_enc, loop.state.label_predict_enc = self.label_predict_enc(
          prev_label=loop.state.label_nb,
          encoder_seq=encoder,
          state=loop.state.label_predict_enc)
      elif isinstance(self.label_predict_enc, IDecoderLabelSyncLabelsOnlyRnn):
        label_predict_enc, loop.state.label_predict_enc = self.label_predict_enc(
          prev_label=loop.state.label_nb,
          state=loop.state.label_predict_enc)
      elif isinstance(self.label_predict_enc, IDecoderLabelSyncAlignDepRnn):
        encoder_frame = ...  # TODO align dep. or unstack if time-sync
        label_predict_enc, loop.state.label_predict_enc = self.label_predict_enc(
          prev_label=loop.state.label_nb,
          encoder_seq=encoder,
          encoder_frame=encoder_frame,
          encoder_frame_idx=...,  # TODO...
          state=loop.state.label_predict_enc)
      else:
        raise TypeError(f"{self}: Unsupported label_predict_enc type {type(self.label_predict_enc)}")

      # TODO join all align label cases
      #  we want to allow for LM fusion, ILM, and thus need to have the nb label prob separate

      if isinstance(self.predictor, IDecoderLabelSyncLogits):
        assert self.label_topology == LabelTopology.LABEL_SYNC, f"{self}: Label topology must be label-sync"
        assert label_predict_enc is not None, f"{self}: Label predict encoder must be specified"
        probs = self.predictor(label_sync_in=label_predict_enc)
        probs_type = "logits"
      elif isinstance(self.predictor, IDecoderJointNoStateLogProb):
        assert self.label_topology != LabelTopology.LABEL_SYNC, f"{self}: Label topology must not be label-sync"
        assert label_predict_enc is not None, f"{self}: Label predict encoder must be specified"
        encoder_frame = ...  # TODO share with above
        predictor_out = self.predictor(time_sync_in=encoder_frame, label_sync_in=label_predict_enc)
        probs = predictor_out.prob_like_wb
        probs_type = predictor_out.prob_like_type
      elif isinstance(self.predictor, IDecoderJointAlignStateLogProb):
        assert self.label_topology != LabelTopology.LABEL_SYNC, f"{self}: Label topology must not be label-sync"
        assert label_predict_enc is not None, f"{self}: Label predict encoder must be specified"
        encoder_frame = ...  # TODO share with above
        predictor_out, loop.state.predictor = self.predictor(
          time_sync_in=encoder_frame,
          label_sync_in=label_predict_enc,
          prev_align_label=loop.state.label_wb,
          state=loop.state.predictor)
        probs = predictor_out.prob_like_wb
        probs_type = predictor_out.prob_like_type
      elif isinstance(self.predictor, IDecoderJointNoCtxLogProb):
        assert self.label_topology != LabelTopology.LABEL_SYNC, f"{self}: Label topology must not be label-sync"
        assert label_predict_enc is None, f"{self}: Label predict encoder not used"
        encoder_frame = ...  # TODO share with above
        predictor_out = self.predictor(time_sync_in=encoder_frame)
        probs = predictor_out.prob_like_wb
        probs_type = predictor_out.prob_like_type
      elif isinstance(self.predictor, IDecoderAlignStateLogProb):
        assert self.label_topology != LabelTopology.LABEL_SYNC, f"{self}: Label topology must not be label-sync"
        assert label_predict_enc is None, f"{self}: Label predict encoder not used"
        encoder_frame = ...  # TODO share with above
        predictor_out, loop.state.predictor = self.predictor(
          time_sync_in=encoder_frame,
          prev_align_label=loop.state.label_wb,
          state=loop.state.predictor)
        probs = predictor_out.prob_like_wb
        probs_type = predictor_out.prob_like_type
      else:
        raise TypeError(f"{self}: Unsupported predictor type {type(self.predictor)}")

      # TODO loss handling here? in that case, cleverly do the most efficient?
      # TODO logits instead of log probs?
      # TODO see below, related is whether and we output
      # TODO or just return log prob (either framewise or fullsum)

      target = loop.unstack(target) if target is not None else None
      if search:
        search.apply_loop(loop)
        align_label = search.choice(probs=probs, probs_type=probs_type)
      else:
        assert target is not None
        align_label = target
      if self.label_topology == LabelTopology.LABEL_SYNC:
        loop.state.label_nb = align_label
        loop.end(loop.state.label_nb == self.target_eos_symbol, include_eos=False)
      else:
        loop.state.label_wb = align_label

      out_labels = loop.stack(align_label) if target is None else None
      # TODO? out_logits = loop.stack(logits)  # TODO not necessarily logits...

    # TODO dataclass out instead...
    return out_labels, loop.axis, loop.state

  def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
    """default init state"""


# TODO enc ctx module
# TODO make generic type for args (Generic, TypeVar) to support not just single layer?


class IDecoderJointLogProbOutput:
  """
  What :class:`IDecoderJointLogProb` returns.
  """

  @property
  def blank_idx(self) -> int:
    """
    :return: blank index
    """
    raise NotImplementedError

  @property
  def prob_like_wb(self) -> nn.Tensor:
    """
    :return: logits if possible, else log probs. see prob_like_type
    """
    return self.log_prob_wb

  @property
  def prob_like_type(self) -> str:
    """
    :return: type of prob_like_wb. "logits" or "log_prob"
    """
    return "log_prob"

  @property
  def log_prob_wb(self) -> nn.Tensor:
    """
    :return: shape (...,D) where is number of classes *with* blank, log probs
    """
    assert self.__class__.log_prob_nb is not IDecoderJointLogProbOutput.log_prob_nb
    log_prob_nb = self.log_prob_nb
    log_prob_blank = nn.expand_dim(self.log_prob_blank, dim=nn.FeatureDim("blank", 1))
    log_prob_nb += self.log_prob_not_blank
    if self.blank_idx == 0:
      log_prob_wb = nn.concat_features(log_prob_blank, log_prob_nb)
    elif self.blank_idx == log_prob_nb.feature_dim:
      log_prob_wb = nn.concat_features(log_prob_nb, log_prob_blank)
    else:
      raise NotImplementedError(f"blank idx {self.blank_idx}, dims (excl blank) {log_prob_nb.feature_dim}")
    return log_prob_wb

  @property
  def log_prob_nb(self) -> nn.Tensor:
    """
    :return: shape (...,D) where is number of classes *without* blank, log probs
    """
    assert self.__class__.log_prob_wb is not IDecoderJointLogProbOutput.log_prob_wb
    log_prob_wb = self.log_prob_wb
    if self.blank_idx == 0:
      log_prob_nb, _ = nn.slice(log_prob_wb, axis=log_prob_wb.feature_dim, slice_start=1)
    elif self.blank_idx == log_prob_wb.feature_dim - 1:
      log_prob_nb, _ = nn.slice(log_prob_wb, axis=log_prob_wb.feature_dim, slice_end=-1)
    else:
      raise NotImplementedError(f"blank idx {self.blank_idx}, dims (incl blank) {log_prob_wb.feature_dim}")
    log_prob_nb -= self.log_prob_not_blank
    return log_prob_nb

  @property
  def log_prob_blank(self) -> nn.Tensor:
    """
    :return: log prob of blank
    """
    assert self.__class__.log_prob_wb is not IDecoderJointLogProbOutput.log_prob_wb
    log_prob_wb = self.log_prob_wb
    return nn.gather(log_prob_wb, position=self.blank_idx, axis=log_prob_wb.feature_dim)

  @property
  def log_prob_not_blank(self) -> nn.Tensor:
    """
    :return: log prob of not blank -> log prob of emitting a non-blank label
    """
    log_prob_blank = self.log_prob_blank
    return nn.safe_log(-nn.expm1(log_prob_blank))


@dataclasses.dataclass
class DecoderJointLogProbSeparatedOutput(IDecoderJointLogProbOutput):
  """
  blank separated
  """
  blank_idx: int
  log_prob_nb: nn.Tensor
  log_prob_blank: nn.Tensor
  log_prob_not_blank: nn.Tensor  # log(-expm1(log_prob_blank)) but you maybe could calc it more directly


class IDecoderLabelSyncLogits(nn.Module):
  """
  For simple (maybe attention-based) encoder-decoder models,
  getting input from some label-sync encoding (TDecoderLabelSync).

  This will produce logits (non-normalized log probs) for non-blank labels.
  There is no blank in this concept.
  """

  def __call__(self, *, label_sync_in: nn.Tensor) -> nn.Tensor:
    raise NotImplementedError


class IDecoderJointNoStateLogProb(nn.Module):
  """
  Joint network for transducer-like models (e.g. the original RNN-T):

  Getting in time-sync inputs, label-sync inputs,
  producing probabilities for labels + blank.

  In case of full-sum training,
  for T time frames,
  N label frames,
  D classes (incl blank),
  it would produce a matrix (..., T, N, D),
  no matter if we use RNN-T (with vertical) or RNA (monotonic) label topology.
  """

  def __call__(self, *, time_sync_in: nn.Tensor, label_sync_in: nn.Tensor) -> IDecoderJointLogProbOutput:
    raise NotImplementedError


class IDecoderJointAlignStateLogProb(nn.Module):
  """
  Joint network for transducer-like models (specifically the extended transducer model):

  Getting in time-sync inputs, label-sync inputs,
  producing probabilities for labels + blank.
  """

  def __call__(self, *,
               time_sync_in: nn.Tensor,
               label_sync_in: nn.Tensor,
               prev_align_label: nn.Tensor,
               state: nn.LayerState,  # align-sync
               ) -> Tuple[IDecoderJointLogProbOutput, nn.LayerState]:
    raise NotImplementedError


class IDecoderJointNoCtxLogProb(nn.Module):
  """
  Joint network for CTC-like models, having no dependence on the label context:

  Getting in time-sync inputs,
  producing probabilities for labels + blank.
  """

  def __call__(self, *, time_sync_in: nn.Tensor) -> IDecoderJointLogProbOutput:
    raise NotImplementedError


class IDecoderAlignStateLogProb(nn.Module):
  """
  Joint network for transducer-like models, no explicit nb label dep, only align-label (like RNA):

  Getting in time-sync inputs,
  producing probabilities for labels + blank.
  """

  def __call__(self, *,
               time_sync_in: nn.Tensor,
               prev_align_label: nn.Tensor,
               state: nn.LayerState,  # align-sync
               ) -> Tuple[IDecoderJointLogProbOutput, nn.LayerState]:
    raise NotImplementedError


TDecoderJointNetLogProb = Union[
  IDecoderLabelSyncLogits, IDecoderJointNoStateLogProb, IDecoderJointAlignStateLogProb,
  IDecoderJointNoCtxLogProb, IDecoderAlignStateLogProb]


class IDecoderLabelSyncRnn(nn.Module):
  """
  Represents SlowRNN in Transducer.

  Inputs:
  - prev_label: last (non-blank) label (called prev_sparse_label_nb earlier)
  - encoder_seq: whole sequence
  - state: prev state

  Outputs:
  - some encoded tensor (no log probs or anything like that). The joint network gets this.
  - new state

  It is up to the implementation to ignore some inputs.
  Ignoring some inputs actually allows for optimizations.
  The standard transducer would actually *only* use the prev_label_nb and state.
  The standard attention-based encoder-decoder would use prev_label_nb, encoder_seq and state,
  and do attention on the encoder_seq.
  (The interface might be extended for segmental models which get the current segment of the encoder.)

  This is usually in a masked computation layer.
  In earlier designs, the interface was supposed to construct the masked computation layer itself,
  and thus required the following further inputs:
  - prev_emit: bool scalar, whether a non-blank label was emitted. used for the mask of the masked computation layer
  - unmasked_sparse_label_nb_seq: optional; like prev_sparse_label_nb but whole sequence.
      This allows to optimize the masked computation layer in training.
  Now, this is not part of the interface anymore, and we handle this outside.

  In earlier designs, there were also the additional inputs:
  - prev_step_sync_rnn: making it alignment-dependent when used; last step-sync-RNN output
  In practice, this was not really used much and thus removed to keep it simpler.

  In other related classes, there are additional inputs:
  - encoder_frame: making it alignment-dependent when used; current frame of encoder
  """

  def __call__(self, *,
               prev_label: nn.Tensor,
               encoder_seq: nn.Tensor,
               state: nn.LayerState,
               ) -> Tuple[nn.Tensor, nn.LayerState]:
    raise NotImplementedError


class IDecoderLabelSyncLabelsOnlyRnn(nn.Module):
  """
  Represents SlowRNN in Transducer.
  In this case, actually this is exactly like the language model interface
  (we might unify the interface).

  Inputs:
  - prev_label: last (non-blank) label (called prev_sparse_label_nb earlier)
  - state: prev state

  Outputs:
  - some encoded tensor (no log probs or anything like that). The joint network gets this.
  - new state
  """

  def __call__(self, *,
               prev_label: nn.Tensor,
               state: nn.LayerState,
               ) -> Tuple[nn.Tensor, nn.LayerState]:
    raise NotImplementedError


class IDecoderLabelSyncAlignDepRnn(nn.Module):
  """
  Represents SlowRNN in Transducer, i.e. label-sync. This variant is alignment-dependent.

  Inputs:
  - prev_label: last (non-blank) label (called prev_sparse_label_nb earlier)
  - encoder_seq: whole sequence
  - encoder_frame: making it alignment-dependent; current frame of encoder
  - encoder_frame_idx: encoder_seq[encoder_frame_idx] == encoder_frame
  - state: prev state

  Outputs:
  - some encoded tensor (no log probs or anything like that). The joint network gets this.
  - new state
  """

  def __call__(self, *,
               prev_label: nn.Tensor,
               encoder_seq: nn.Tensor,
               encoder_frame: nn.Tensor,
               encoder_frame_idx: nn.Tensor,
               state: nn.LayerState,
               ) -> Tuple[nn.Tensor, nn.LayerState]:
    raise NotImplementedError


TDecoderLabelSync = Union[IDecoderLabelSyncRnn, IDecoderLabelSyncLabelsOnlyRnn, IDecoderLabelSyncAlignDepRnn]


class IDecoderStepSyncRnn(nn.Module):
  """
  Represents FastRNN in Transducer.
  Otherwise, in general this runs step-synchronous,
  which is alignment-synchronous or time-synchronous for RNN-T/RNA/CTC,
  or label-synchronous for att-enc-dec.
  """
  def __call__(self, *,
               prev_label_wb: nn.Tensor,
               encoder: nn.Tensor,  # TODO enc ctx. or not? need full encoder for full-sum case...
               label_sync_rnn: nn.Tensor) -> nn.Tensor:
    """
    prev_label_wb and encoder use the same time dim (T) (or none).
    label_sync_rnn can use the same (unmasked) (or none) or a different (U+1) (maybe in full-sum setting).
    When label_sync_rnn has a different time dim than prev_label_wb/encoder,
    we expect 2 time dims as output (Tx(U+1)).
    """
    raise NotImplementedError


class IDecoderLogProbSeparateNb(nn.Module):
  """
  Log prob separate without blank.
  """
  def __call__(self, step_sync_rnn: nn.Tensor) -> nn.Tensor:
    """
    Make log-prob distribution over labels (without blank).

    :param step_sync_rnn: might be the FastRNN in transducer, or SlowRNN in att-enc-dec model
    """
    raise NotImplementedError


class IDecoderLogProbSeparateWb(nn.Module):
  """
  Log prob with blank.
  """
  def __call__(self, step_sync_rnn: nn.Tensor, log_prob_nb: nn.Tensor) -> nn.Tensor:
    """
    Make layer dict.
    """
    raise NotImplementedError
