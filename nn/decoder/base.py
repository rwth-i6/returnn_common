"""
Generic decoder interface and implementation.

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
from typing import Union, Optional, Tuple
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

    @classmethod
    def is_time_sync(cls, topo: LabelTopology) -> bool:
        """
        :return: whether the given topology is time-synchronous
        """
        return topo in (cls.TIME_SYNC_PEAKY, cls.TIME_SYNC_LABEL_LOOP)


class IDecoder:
    """
    Generic decoder interface
    """

    @property
    def label_topology(self) -> LabelTopology:
        """
        label topology
        """
        raise NotImplementedError

    def next_log_prob(self, *, prev_label: nn.Tensor, state: nn.LayerState) -> (nn.Tensor, nn.LayerState):
        """
        :return: probs over labels, new state

        The labels are alignment labels (incl blank) in case the label topology is not label-sync.

        This is used for the search.
        """
        raise NotImplementedError


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
        # Default implementation uses log_prob_nb and log_prob_not_blank to derive it.
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
        # Default implementation uses direct_logits_nb if possible.
        if self.direct_logits_nb is not None:
            return nn.log_softmax(self.direct_logits_nb, axis=self.direct_logits_nb.feature_dim)
        # Otherwise use log_prob_wb and log_prob_not_blank to derive it.
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
    def direct_logits_nb(self) -> Optional[nn.Tensor]:
        """
        :return: logits for non-blank labels
        """
        return None

    @property
    def log_prob_blank(self) -> nn.Tensor:
        """
        :return: log prob of blank
        """
        # Default implementation uses log_prob_wb to derive it.
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


class IGenericLanguageModelLogits(nn.Module):
    """
    Generic language model.
    This interface should be usable both for training (operating on the whole sequence)
    and recognition (operating per step).
    It returns logits.
    """

    def __call__(
        self, *, prev_label: nn.Tensor, axis: nn.Dim = nn.single_step_dim, state: Optional[nn.LayerState]
    ) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
        raise NotImplementedError

    def as_decoder_scaled_log_probs(
        self, *, scale: Union[nn.Tensor, float] = 1.0
    ) -> IDecoderLabelSyncStateScaledLogProbs:
        """
        :return: interface
        """
        return _IDecoderScaledLogProbsFromLm(language_model=self, scale=scale)


class IDecoderLabelSyncStateScaledLogProbs:
    """
    For example, a language model.
    This interface is used for both the external LM fusion,
    and also internal LM subtraction.
    The internal LM estimation can fit into this interface.
    The score returned here could be:
      log P_extLM(a_s | ...) - log P_ILM(a_s | ...).
    This interface is not a nn.Module because this would dynamically be added to Decoder.__call__.
    """

    def __call__(
        self, *, prev_label: nn.Tensor, state: Optional[nn.LayerState]
    ) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
        raise NotImplementedError


class _IDecoderScaledLogProbsFromLm(IDecoderLabelSyncStateScaledLogProbs):
    def __init__(
        self,
        *,
        language_model: IGenericLanguageModelLogits,
        scale: Union[nn.Tensor, float] = 1.0,
    ):
        super(_IDecoderScaledLogProbsFromLm, self).__init__()
        self.language_model = language_model
        self.scale = scale

    def __call__(
        self, *, prev_label: nn.Tensor, state: Optional[nn.LayerState]
    ) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
        logits, state = self.language_model(prev_label=prev_label, axis=nn.single_step_dim, state=state)
        res = nn.log_softmax(logits, axis=logits.feature_dim)
        if not isinstance(self.scale, (float, int)) or self.scale != 1:
            res *= self.scale
        return res, state


class IDecoderLabelSyncLogits(nn.Module):
    """
    For simple (maybe attention-based) encoder-decoder models,
    getting input from some label-sync encoding (TDecoderLabelSync).

    This will produce logits (non-normalized log probs) for non-blank labels.
    There is no blank in this concept.
    """

    def __call__(self, *, label_sync_in: nn.Tensor) -> nn.Tensor:
        raise NotImplementedError


class IDecoderJointBaseLogProb(nn.Module):
    """
    Joint network for transducer-like models (e.g. the original RNN-T).
    Base interface.
    It shares that it always outputs a :class:`IDecoderJointLogProbOutput`,
    and always gets in time_sync_in.
    """

    def __call__(self, **kwargs) -> IDecoderJointLogProbOutput:
        raise NotImplementedError


class IDecoderJointNoStateLogProb(IDecoderJointBaseLogProb):
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


class IDecoderJointAlignStateLogProb(IDecoderJointBaseLogProb):
    """
    Joint network for transducer-like models (specifically the extended transducer model):

    Getting in time-sync inputs, label-sync inputs,
    producing probabilities for labels + blank.
    """

    def __call__(
        self,
        *,
        time_sync_in: nn.Tensor,
        label_sync_in: nn.Tensor,
        prev_align_label: nn.Tensor,
        state: nn.LayerState,  # align-sync
    ) -> Tuple[IDecoderJointLogProbOutput, nn.LayerState]:
        raise NotImplementedError


class IDecoderJointNoCtxLogProb(IDecoderJointBaseLogProb):
    """
    Joint network for CTC-like models, having no dependence on the label context:

    Getting in time-sync inputs,
    producing probabilities for labels + blank.
    """

    def __call__(self, *, time_sync_in: nn.Tensor) -> IDecoderJointLogProbOutput:
        raise NotImplementedError


class IDecoderAlignStateLogProb(IDecoderJointBaseLogProb):
    """
    Joint network for transducer-like models, no explicit nb label dep, only align-label (like RNA):

    Getting in time-sync inputs,
    producing probabilities for labels + blank.
    """

    def __call__(
        self,
        *,
        time_sync_in: nn.Tensor,
        prev_align_label: nn.Tensor,
        state: nn.LayerState,  # align-sync
    ) -> Tuple[IDecoderJointLogProbOutput, nn.LayerState]:
        raise NotImplementedError


TDecoderJointNetLogProb = Union[
    IDecoderLabelSyncLogits,
    IDecoderJointNoStateLogProb,
    IDecoderJointAlignStateLogProb,
    IDecoderJointNoCtxLogProb,
    IDecoderAlignStateLogProb,
]


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

    def __call__(
        self,
        *,
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

    def __call__(
        self,
        *,
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

    def __call__(
        self,
        *,
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

    def __call__(
        self,
        *,
        prev_label_wb: nn.Tensor,
        encoder: nn.Tensor,  # TODO enc ctx. or not? need full encoder for full-sum case...
        label_sync_rnn: nn.Tensor,
    ) -> nn.Tensor:
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
