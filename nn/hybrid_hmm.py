"""
The base model and interface of a hybrid NN-HMM.
"""

from typing import Optional, Union, Tuple
from .. import nn
from .encoder import ISeqFramewiseEncoder, ISeqDownsamplingEncoder


EncoderType = Union[ISeqFramewiseEncoder, ISeqDownsamplingEncoder]


class IHybridHMM(nn.Module):
  """
  Hybrid NN-HMM interface
  """

  encoder: EncoderType
  out_dim: nn.Dim

  def __call__(self, source: nn.Tensor, *,
               state: Optional[nn.LayerState] = None,
               train: bool = False,
               targets: Optional[nn.Tensor] = None
               ) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
    """
    :param source: [B,T,in_dim], although not necessarily in that order. we require that time_dim_axis is set.
    :param state: previous state
    :param train: if set, it assumes we have given a target, and it will add losses
    :param targets: only used for training. [B,T] -> out_dim sparse targets.
    :return: (posteriors, final state),
      posteriors as log-prob, as [B,T',out_dim], although dims not necessarily in that order.
    """
    raise NotImplementedError


class HybridHMM(IHybridHMM):
  """
  Hybrid NN-HMM
  """

  def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim, *, encoder: EncoderType):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.encoder = encoder
    self.out_projection = nn.Linear(encoder.out_dim, out_dim)

  def __call__(self, source: nn.Tensor, *,
               state: Optional[nn.LayerState] = None,
               train: bool = False,
               targets: Optional[nn.Tensor] = None
               ) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
    assert source.data.time_dim_axis is not None
    in_spatial_dim = source.data.dim_tags[source.data.time_dim_axis]
    assert state is None, f"{self} stateful hybrid HMM not supported yet"
    if isinstance(self.encoder, ISeqFramewiseEncoder):
      encoder_output = self.encoder(source, spatial_dim=in_spatial_dim)
      out_spatial_dim = in_spatial_dim
    elif isinstance(self.encoder, ISeqDownsamplingEncoder):
      encoder_output, out_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim)
    else:
      raise TypeError(f"unsupported encoder type {type(self.encoder)}")
    out_embed = self.out_projection(encoder_output)
    if train:
      assert out_spatial_dim in targets.shape
      ce_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=out_embed, targets=targets, axis=self.out_dim)
      ce_loss.mark_as_loss("ce")
    return nn.log_softmax(out_embed, axis=self.out_dim), None
