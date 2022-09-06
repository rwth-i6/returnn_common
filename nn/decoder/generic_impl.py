"""
Generic decoder implementation.

work-in-progress!
(Maybe it will just stay as an example, guide, proof-of-concept, or so, and never really be used...)
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Union, Tuple, Sequence

from ... import nn
from .base import LabelTopology, TDecoderLabelSync, TDecoderJointNetLogProb, \
  IDecoderLabelSyncStateScaledLogProbs, IDecoderLabelSyncAlignDepRnn, IDecoderJointBaseLogProb, IDecoderLabelSyncRnn, \
  IDecoderLabelSyncLabelsOnlyRnn, IDecoderLabelSyncLogits, IDecoderJointNoStateLogProb, \
  IDecoderJointAlignStateLogProb, \
  IDecoderJointNoCtxLogProb, IDecoderAlignStateLogProb


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
               target_blank_symbol: Optional[int] = None,
               ):
    super().__init__()
    self.label_topology = label_topology
    self.label_predict_enc = label_predict_enc  # earlier: slow_rnn. label-sync. incl (nb) label embedding
    self.predictor = predictor  # earlier: fast_rnn + readout. align-sync or matrix time * label. predicts align label
    self.target_dim = target_dim  # includes blank if not label-sync
    self.target_bos_symbol = target_bos_symbol
    self.target_eos_symbol = target_eos_symbol
    if self.label_topology != LabelTopology.LABEL_SYNC:
      assert target_blank_symbol is not None, f"{self} need target_blank_symbol with non-label-sync topology"
    self.target_blank_symbol = target_blank_symbol

  def __call__(self, *,
               encoder: nn.Tensor,
               encoder_spatial_axis: nn.Dim,
               target: Union[nn.Tensor, nn.SearchFuncInterface],
               label_scores_ext: Optional[IDecoderLabelSyncStateScaledLogProbs] = None,
               axis: Optional[nn.Dim] = None,
               state: Optional[nn.LayerState] = None,
               ) -> Tuple[nn.Tensor, nn.Dim, nn.LayerState]:
    """
    Make one decoder step (train and/or recognition).
    """
    # TODO ...
    search = None
    if isinstance(target, nn.SearchFuncInterface):
      search = target
      target = None
    if target is not None:
      assert axis, f"{self}: Target spatial axis must be specified when target is given"

    loop = nn.Loop(axis=axis)
    loop.state = state if state else self.default_initial_state(
      batch_dims=encoder.batch_dims_ordered(remove=(encoder_spatial_axis, encoder.feature_dim)))
    with loop:

      encoder_frame_idx = None
      if self.label_topology != LabelTopology.LABEL_SYNC:
        encoder_frame_idx = loop.state.encoder_frame_idx

      encoder_frame = None
      if (  # only get it when we really need it
            isinstance(self.label_predict_enc, IDecoderLabelSyncAlignDepRnn) or
            isinstance(self.predictor, IDecoderJointBaseLogProb)):
        if LabelTopology.is_time_sync(self.label_topology):
          assert axis in (nn.single_step_dim, encoder_spatial_axis)
          encoder_frame = loop.unstack(encoder)
        else:
          assert encoder_frame_idx is not None, f"{self}: invalid label topology {self.label_topology}?"
          encoder_frame = nn.gather(encoder, axis=encoder_spatial_axis, position=encoder_frame_idx)

      if self.label_predict_enc is None:
        label_predict_enc = None
      else:
        if self.label_topology == LabelTopology.LABEL_SYNC:
          comp_context_mgr = nullcontext()
        else:
          if target is not None and not isinstance(self.label_predict_enc, IDecoderLabelSyncAlignDepRnn):
            # We are not align dep, i.e. don't need the current encoder frame,
            # and we also have the targets, thus we can directly operate in the target non-blank frames.
            # This allows to make the training criterion more efficient below.
            comp_mask = target != self.target_blank_symbol
          else:
            comp_mask = loop.state.label_wb != self.target_blank_symbol
          comp_context_mgr = nn.MaskedComputation(mask=comp_mask)
        with comp_context_mgr:
          if isinstance(self.label_predict_enc, IDecoderLabelSyncRnn):
            label_predict_enc, loop.state.label_predict_enc = self.label_predict_enc(
              prev_label=loop.state.label_nb,
              encoder_seq=encoder,
              state=loop.state.label_predict_enc)
          elif isinstance(self.label_predict_enc, IDecoderLabelSyncLabelsOnlyRnn):
            label_predict_enc, loop.state.label_predict_enc = self.label_predict_enc(
              prev_label=loop.state.label_nb,
              state=loop.state.label_predict_enc)
          elif isinstance(self.label_predict_enc, IDecoderLabelSyncAlignDepRnn):
            assert encoder_frame is not None
            label_predict_enc, loop.state.label_predict_enc = self.label_predict_enc(
              prev_label=loop.state.label_nb,
              encoder_seq=encoder,
              encoder_frame=encoder_frame,
              encoder_frame_idx=encoder_frame_idx,
              state=loop.state.label_predict_enc)
          else:
            raise TypeError(f"{self}: Unsupported label_predict_enc type {type(self.label_predict_enc)}")

      if isinstance(self.predictor, IDecoderLabelSyncLogits):
        assert self.label_topology == LabelTopology.LABEL_SYNC, f"{self}: Label topology must be label-sync"
        assert label_predict_enc is not None, f"{self}: Label predict encoder must be specified"
        probs = self.predictor(label_sync_in=label_predict_enc)
        probs_type = "logits"
      elif isinstance(self.predictor, IDecoderJointBaseLogProb):
        assert self.label_topology != LabelTopology.LABEL_SYNC, f"{self}: Label topology must not be label-sync"
        assert label_predict_enc is not None, f"{self}: Label predict encoder must be specified"
        assert encoder_frame is not None
        if isinstance(self.predictor, IDecoderJointNoStateLogProb):
          predictor_out = self.predictor(time_sync_in=encoder_frame, label_sync_in=label_predict_enc)
        elif isinstance(self.predictor, IDecoderJointAlignStateLogProb):
          predictor_out, loop.state.predictor = self.predictor(
            time_sync_in=encoder_frame,
            label_sync_in=label_predict_enc,
            prev_align_label=loop.state.label_wb,
            state=loop.state.predictor)
        elif isinstance(self.predictor, IDecoderJointNoCtxLogProb):
          predictor_out = self.predictor(time_sync_in=encoder_frame)
        elif isinstance(self.predictor, IDecoderAlignStateLogProb):
          predictor_out, loop.state.predictor = self.predictor(
            time_sync_in=encoder_frame,
            prev_align_label=loop.state.label_wb,
            state=loop.state.predictor)
        else:
          raise TypeError(f"{self}: Unsupported predictor joint type {type(self.predictor)}")
        probs = predictor_out.prob_like_wb
        probs_type = predictor_out.prob_like_type
        assert predictor_out.blank_idx == self.target_blank_symbol
        if predictor_out.direct_logits_nb is not None:
          # TODO ...
          # nn.cross_entropy(...)
          # nn.binary_cross_entropy()
          pass
      else:
        raise TypeError(f"{self}: Unsupported predictor type {type(self.predictor)}")

      # TODO use label_scores_ext
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

      if encoder_frame_idx is not None:
        if LabelTopology.is_time_sync(self.label_topology):
          loop.state.encoder_frame_idx = encoder_frame_idx + 1
        elif self.label_topology == LabelTopology.WITH_VERTICAL:
          loop.state.encoder_frame_idx = (
            encoder_frame_idx + nn.cast(align_label == self.target_blank_symbol, dtype="int32"))
        else:
          raise ValueError(f"{self}: Unexpected label topology {self.label_topology}")

      out_labels = loop.stack(align_label) if target is None else None
      # TODO? out_logits = loop.stack(logits)  # TODO not necessarily logits...

    # TODO dataclass out instead...
    return out_labels, loop.axis, loop.state

  def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
    """default init state"""
    # TODO ...
