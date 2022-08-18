"""
Common BLSTM-based encoder often used for end-to-end (attention, transducer) models:

SpecAugment . PreCNN . BLSTM
"""


from typing import Union, Tuple

from ...asr.specaugment import specaugment_v2
from ... import nn
from .blstm import BlstmEncoder


class BlstmCnnSpecAugEncoder(BlstmEncoder):
  """
  SpecAugment . PreCNN . BLSTM
  """

  def __init__(self,
               num_layers: int = 6, lstm_dim: nn.Dim = nn.FeatureDim("lstm-dim", 1024),
               time_reduction: Union[int, Tuple[int, ...]] = 6,
               with_specaugment=True,
               l2=0.0001, dropout=0.3, rec_weight_dropout=0.0,):
    super(BlstmCnnSpecAugEncoder, self).__init__(
      num_layers=num_layers, dim=lstm_dim, time_reduction=time_reduction,
      l2=l2, dropout=dropout, rec_weight_dropout=rec_weight_dropout)

    self.with_specaugment = with_specaugment
    self.pre_conv_net = PreConvNet()

  def __call__(self, x, *, spatial_dim: nn.Dim) -> nn.Tensor:
    if self.with_specaugment:
      x = specaugment_v2(x, spatial_dim=spatial_dim)
    x = self.pre_conv_net(x, spatial_dim=spatial_dim)
    x = super(BlstmCnnSpecAugEncoder, self).__call__(x, spatial_dim=spatial_dim)
    return x


class PreConvNet(nn.Module):
  """
  2 layer pre conv net, usually used before a BLSTM
  """
  def __init__(self, filter_size=(3, 3), dim: nn.Dim = nn.FeatureDim("feat", 32)):
    super(PreConvNet, self).__init__()
    self.conv0 = nn.Conv2d(out_dim=dim, padding="same", filter_size=filter_size)
    self.conv1 = nn.Conv2d(out_dim=dim, padding="same", filter_size=filter_size)

  def __call__(self, x: nn.Tensor, *, spatial_dim: nn.Dim) -> nn.Tensor:
    extra_spatial_dim = x.feature_dim
    feat_dim = nn.FeatureDim("dummy-feature", 1)
    x = nn.expand_dim(x, dim=feat_dim)
    x, _ = self.conv0(x, in_spatial_dims=(spatial_dim, extra_spatial_dim), in_dim=feat_dim)
    feat_dim = x.feature_dim
    x, extra_spatial_dim = nn.pool1d(x, in_spatial_dim=extra_spatial_dim, pool_size=2, mode="max", padding="same")
    x, _ = self.conv1(x, in_spatial_dims=(spatial_dim, extra_spatial_dim), in_dim=feat_dim)
    x, extra_spatial_dim = nn.pool1d(x, in_spatial_dim=extra_spatial_dim, pool_size=2, mode="max", padding="same")
    x, _ = nn.merge_dims(x, axes=(extra_spatial_dim, feat_dim), out_dim=nn.FeatureDim("conv-net-feature", None))
    return x
