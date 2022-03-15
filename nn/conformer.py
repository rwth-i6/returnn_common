"""
Conformer code.
Ref: https://arxiv.org/abs/2005.08100
"""

from typing import Tuple, List, Callable, Optional, Union, Any
import copy as _copy
from .. import nn


class ConformerPositionwiseFeedForward(nn.Module):
  """
  Conformer position-wise feedforward neural network layer
      FF -> Activation -> Dropout -> FF
  """

  def __init__(self, out_dim: nn.Dim, *, ff_dim: nn.Dim, dropout: float,
               activation: Callable[[nn.Tensor], nn.Tensor]):
    """
    :param out_dim: output feature dimension
    :param ff_dim: dimension of the feed-forward layers
    :param dropout: dropout value
    :param activation: activation function
    """
    super().__init__()

    self.dropout = dropout
    self.activation = activation

    self.linear_ff = nn.Linear(ff_dim)
    self.linear_out = nn.Linear(out_dim)

  @nn.scoped
  def __call__(self, inp: nn.Tensor) -> nn.Tensor:
    """forward"""
    x_ff1 = self.linear_ff(inp)
    x_act = self.activation(x_ff1)
    x_drop = nn.dropout(x_act, axis=inp.feature_dim, dropout=self.dropout)
    x_ff2 = self.linear_out(x_drop)
    return x_ff2


class ConformerConvBlock(nn.Module):
  """
  Conformer convolution block
      FF -> GLU -> depthwise conv -> BN -> Swish -> FF
  """

  def __init__(self, out_dim: nn.Dim, *, kernel_size: int, norm: Union[nn.BatchNorm, Any]):
    """
    :param out_dim: output feature dimension
    :param kernel_size: kernel size of depthwise convolution
    :param norm: Batch norm originally
    """
    super().__init__()

    self.positionwise_conv1 = nn.Linear(2 * out_dim)
    self.depthwise_conv = nn.Conv1d(
      out_dim=out_dim, filter_size=kernel_size, groups=out_dim.dimension, padding='same')
    self.positionwise_conv2 = nn.Linear(out_dim)
    self.norm = norm

  @nn.scoped
  def __call__(self, inp: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
    """forward"""
    x_conv1 = self.positionwise_conv1(inp)
    x_act = nn.gating(x_conv1)
    x_depthwise_conv, _ = self.depthwise_conv(x_act, in_spatial_dim=axis)
    x_normed = self.norm(x_depthwise_conv)
    x_swish = nn.swish(x_normed)
    x_conv2 = self.positionwise_conv2(x_swish)
    return x_conv2


class ConformerConvSubsample(nn.Module):
  """
  Conv 2D block with optional max-pooling
  """

  def __init__(
        self, *, filter_sizes: List[Tuple[int, int]], out_dims: List[nn.Dim], dropout: float,
        pool_sizes: Optional[List[Tuple[int, int]]] = None,
        activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu,
        padding: str = 'same'):
    """
    :param filter_sizes: a list of filter sizes for the conv layer
    :param out_dims: the number of output channels. last element is the output feature dimension
    :param dropout: the dropout value
    :param pool_sizes: a list of pooling factors applied after conv layer
    :param activation: the activation function
    :param padding: 'same' or 'valid'
    """
    super().__init__()

    self.dropout = dropout
    self.pool_sizes = pool_sizes
    self.activation = activation

    self.conv_layers = nn.ModuleList()
    assert len(filter_sizes) == len(out_dims) > 0
    for filter_size, out_dim in zip(filter_sizes, out_dims):
      self.conv_layers.append(
        nn.Conv2d(filter_size=filter_size, out_dim=out_dim, padding=padding))
    self.out_dim = out_dims[-1]

  @nn.scoped
  def __call__(self, inp: nn.Tensor, *, in_spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
    """forward"""
    in_spatial_dims = [in_spatial_dim, inp.feature_dim]
    in_dim = nn.FeatureDim("dummy-input-feature-dim", 1)
    x = nn.expand_dim(inp, dim=in_dim)
    for i, conv_layer in enumerate(self.conv_layers):
      x, in_spatial_dims = conv_layer(x, in_dim=in_dim, in_spatial_dims=in_spatial_dims)
      in_dim = conv_layer.out_dim
      x = self.activation(x)
      if self.pool_sizes and i < len(self.pool_sizes):
        x, in_spatial_dims = nn.pool2d(
          x, in_spatial_dims=in_spatial_dims,
          pool_size=self.pool_sizes[i], padding='same', mode='max')
      if self.dropout:
        x = nn.dropout(x, axis=in_dim, dropout=self.dropout)
    out, out_spatial_dim = nn.merge_dims(x, axes=in_spatial_dims)
    return out, out_spatial_dim


class ConformerEncoderLayer(nn.Module):
  """
  Represents a conformer block
  """

  def __init__(
        self,
        out_dim: nn.Dim = nn.FeatureDim("conformer-enc-default-out-dim", 512),
        *,
        ff_dim: nn.Dim = nn.NotSpecified,
        ff_activation: Callable[[nn.Tensor], nn.Tensor] = nn.swish,
        dropout: float = 0.1,
        conv_kernel_size: int = 32,
        conv_norm: Union[nn.BatchNorm, Any] = nn.NotSpecified,
        num_heads: int = 8,
        att_dropout: float = 0.1,
        ):
    """
    :param out_dim: the output feature dimension
    :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
    :param ff_activation: activation function for feed-forward network
    :param dropout: the dropout value for the FF block
    :param conv_kernel_size: the kernel size of depthwise convolution in the conv block
    :param conv_norm: used for the conv block. Batch norm originally
    :param num_heads: the number of attention heads
    :param att_dropout: attention dropout value
    """
    super().__init__()

    self.dropout = dropout
    self.out_dim = out_dim

    if ff_dim is nn.NotSpecified:
      ff_dim = out_dim * 4
    self.ffn1 = ConformerPositionwiseFeedForward(
      out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=ff_activation)
    self.ffn1_layer_norm = nn.LayerNorm()

    self.ffn2 = ConformerPositionwiseFeedForward(
      out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, activation=ff_activation)
    self.ffn2_layer_norm = nn.LayerNorm()

    if conv_norm is nn.NotSpecified:
      conv_norm = nn.BatchNorm(use_mask=False)
    self.conv_block = ConformerConvBlock(
      out_dim=out_dim, kernel_size=conv_kernel_size, norm=conv_norm)
    self.conv_layer_norm = nn.LayerNorm()

    self.self_att = nn.SelfAttention(
      key_dim_total=out_dim, value_dim_total=out_dim, num_heads=num_heads, att_dropout=att_dropout)
    self.self_att_layer_norm = nn.LayerNorm()

    self.final_layer_norm = nn.LayerNorm()

  @nn.scoped
  def __call__(self, inp: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
    """forward"""
    # FFN
    x_ffn1_ln = self.ffn1_layer_norm(inp, in_dim=inp.feature_dim)
    x_ffn1 = self.ffn1(x_ffn1_ln)
    x_ffn1_out = 0.5 * nn.dropout(x_ffn1, axis=inp.feature_dim, dropout=self.dropout) + inp

    # MHSA
    x_mhsa_ln = self.self_att_layer_norm(x_ffn1_out, in_dim=inp.feature_dim)
    x_mhsa = self.self_att(x_mhsa_ln, axis=axis)
    x_mhsa_out = x_mhsa + x_ffn1_out

    # Conv
    x_conv_ln = self.conv_layer_norm(x_mhsa_out, in_dim=inp.feature_dim)
    x_conv = self.conv_block(x_conv_ln, axis=axis)
    x_conv_out = nn.dropout(x_conv, axis=inp.feature_dim, dropout=self.dropout) + x_mhsa_out

    # FFN
    x_ffn2_ln = self.ffn2_layer_norm(x_conv_out, in_dim=inp.feature_dim)
    x_ffn2 = self.ffn2(x_ffn2_ln)
    x_ffn2_out = 0.5 * nn.dropout(x_ffn2, axis=inp.feature_dim, dropout=self.dropout) + x_conv_out

    # last LN layer
    return self.final_layer_norm(x_ffn2_out, in_dim=inp.feature_dim)


class ConformerEncoder(nn.Module):
  """
  Represents Conformer encoder architecture
  """

  def __init__(self,
               out_dim: nn.Dim = nn.FeatureDim("conformer-enc-default-out-dim", 512),
               *,
               num_layers: int,
               ff_dim: nn.Dim = nn.NotSpecified,
               ff_activation: Callable[[nn.Tensor], nn.Tensor] = nn.swish,
               dropout: float = 0.1,
               conv_kernel_size: int = 32,
               conv_norm: Union[nn.BatchNorm, Any] = nn.NotSpecified,
               num_heads: int = 8,
               att_dropout: float = 0.1,
               custom_encoder_layer: Optional[Union[ConformerEncoderLayer, Any]] = None):
    """
    :param out_dim: the output feature dimension
    :param num_layers: the number of encoder layers
    :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
    :param ff_activation: activation funtion for feed-forward network
    :param dropout: the dropout value for the FF block
    :param conv_kernel_size: the kernel size of depthwise convolution in the conv block
    :param conv_norm: used for the conv block. Batch norm originally
    :param num_heads: the number of attention heads
    :param att_dropout: attention dropout value
    :param custom_encoder_layer: an instance of :class:`ConformerEncoderLayer` or similar
    """
    super().__init__()

    self.out_dim = out_dim
    self.dropout = dropout

    self.conv_subsample_layer = ConformerConvSubsample(
      filter_sizes=[(3, 3), (3, 3)],
      pool_sizes=[(2, 2), (2, 2)],
      out_dims=[self.out_dim.copy(same_as_self=False, description="intermediate_out_sub_sample"), self.out_dim],
      dropout=dropout)

    self.linear = nn.Linear(self.out_dim, with_bias=False)

    if custom_encoder_layer:
      encoder_layer = custom_encoder_layer
    else:
      encoder_layer = ConformerEncoderLayer(
        out_dim=out_dim, ff_dim=ff_dim, ff_activation=ff_activation, dropout=dropout,
        conv_kernel_size=conv_kernel_size, conv_norm=conv_norm, num_heads=num_heads, att_dropout=att_dropout)

    self.layers = nn.Sequential(_copy.deepcopy(encoder_layer) for _ in range(num_layers))

  @nn.scoped
  def __call__(self, inp: nn.Tensor, *, in_spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
    """forward"""
    x_subsample, out_spatial_dim = self.conv_subsample_layer(inp, in_spatial_dim=in_spatial_dim)
    x_linear = self.linear(x_subsample)
    x = nn.dropout(x_linear, axis=self.linear.out_dim, dropout=self.dropout)
    x = self.layers(x, axis=out_spatial_dim)
    return x, out_spatial_dim
