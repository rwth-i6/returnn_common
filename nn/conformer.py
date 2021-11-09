"""
Conformer code.
Ref: https://arxiv.org/abs/2005.08100
"""

from typing import Tuple, List, Callable, Optional, Dict, Any
from .. import nn
import copy


class ConformerPositionwiseFeedForward(nn.Module):
  """
  Conformer position-wise feedforward neural network layer
      FF -> Activation -> Dropout -> FF
  """

  def __init__(self, out_dim: int, dim_ff: int, dropout: float, activation: Callable[[nn.LayerRef], nn.LayerRef]):
    """
    :param out_dim: output feature dimension
    :param dim_ff: dimension of the feed-forward layers
    :param dropout: dropout value
    :param activation: activation function
    """
    super().__init__()

    self.dropout = dropout
    self.activation = activation

    self.linear_ff = nn.Linear(n_out=dim_ff)
    self.linear_out = nn.Linear(n_out=out_dim)

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    x_ff1 = self.linear_ff(inp)
    x_act = self.activation(x_ff1)
    x_drop = nn.dropout(x_act, dropout=self.dropout)
    x_ff2 = self.linear_out(x_drop)
    return x_ff2


class ConformerConvBlock(nn.Module):
  """
  Conformer convolution block
      FF -> GLU -> depthwise conv -> BN -> Swish -> FF
  """

  def __init__(self, out_dim: int, kernel_size: int, batch_norm_opts: Optional[Dict[str, Any]] = None):
    """
    :param out_dim: output feature dimension
    :param kernel_size: kernel size of depthwise convolution
    :param batch_norm_opts: batch norm options
    """
    super().__init__()

    self.positionwise_conv1 = nn.Linear(n_out=out_dim * 2)
    self.depthwise_conv = nn.Conv(n_out=out_dim, filter_size=(kernel_size,), groups=out_dim, padding='same')
    self.positionwise_conv2 = nn.Linear(n_out=out_dim)

    if batch_norm_opts is None:
      batch_norm_opts = {}
    batch_norm_opts = batch_norm_opts.copy()
    batch_norm_opts.setdefault('epsilon', 1e-5)
    batch_norm_opts.setdefault('momentum', 0.1)
    self.batch_norm = nn.BatchNorm(update_sample_only_in_training=True, delay_sample_update=True, **batch_norm_opts)

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    x_conv1 = self.positionwise_conv1(inp)
    x_act = nn.glu(x_conv1)
    x_depthwise_conv = self.depthwise_conv(x_act)
    x_bn = self.batch_norm(x_depthwise_conv)
    x_swish = nn.swish(x_bn)
    x_conv2 = self.positionwise_conv2(x_swish)
    return x_conv2


class ConformerConvSubsample(nn.Module):
  """
  Conv 2D block with optional max-pooling
  """

  def __init__(
        self, filter_sizes: List[Tuple[int, int]], channel_sizes: List[int], dropout: float,
        pool_sizes: Optional[List[Tuple[int, int]]] = None, activation: Callable[[nn.LayerRef], nn.LayerRef] = nn.relu,
        padding: str = 'same'):
    """
    :param filter_sizes: a list of filter sizes for the conv layer
    :param channel_sizes: the number of output channels
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
    assert len(filter_sizes) == len(channel_sizes)
    for filter_size, channel_size in zip(filter_sizes, channel_sizes):
      self.conv_layers.append(
        nn.Conv(filter_size=filter_size, n_out=channel_size, padding=padding))

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    x = nn.split_dims(inp, axis='F', dims=(-1, 1))
    for i, conv_layer in enumerate(self.conv_layers):
      x = conv_layer(x)
      x = self.activation(x)
      if self.pool_sizes and i < len(self.pool_sizes):
        x = nn.pool(x, pool_size=self.pool_sizes[i], padding='same', mode='max')
      if self.dropout:
        x = nn.dropout(x, dropout=self.dropout)
    out = nn.merge_dims(x, axes='static')
    return out


class ConformerEncoderLayer(nn.Module):
  """
  Represents a conformer block
  """

  def __init__(
        self, conv_kernel_size: int = 32, activation_ff: Callable[[nn.LayerRef], nn.LayerRef] = nn.swish,
        dim_ff: int = 2048, dropout: float = 0.1, att_dropout: float = 0.1, out_dim: int = 512, num_heads: int = 8,
        batch_norm_opts: Optional[Dict[str, Any]] = None):
    """
    :param conv_kernel_size: the kernel size of depthwise convolution
    :param activation_ff: activation funtion for feed-forward network
    :param dim_ff: the dimension of feed-forward layers
    :param dropout: the dropout value
    :param att_dropout: attention dropout value
    :param out_dim: the output feature dimension
    :param num_heads: the number of attention heads
    :param batch_norm_opts: passed to :class:`nn.BatchNorm`
    """
    super().__init__()

    self.dropout = dropout
    self.out_dim = out_dim

    self.ffn1 = ConformerPositionwiseFeedForward(
      out_dim=out_dim, dim_ff=dim_ff, dropout=dropout, activation=activation_ff)

    self.ffn2 = ConformerPositionwiseFeedForward(
      out_dim=out_dim, dim_ff=dim_ff, dropout=dropout, activation=activation_ff)

    self.conv_block = ConformerConvBlock(
      out_dim=out_dim, kernel_size=conv_kernel_size, batch_norm_opts=batch_norm_opts)

    self.self_att = nn.SelfAttention(axis='T', key_dim_total=out_dim, value_dim_total=out_dim, num_heads=num_heads)

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    # FFN
    x_ffn1_ln = nn.layer_norm(inp)
    x_ffn1 = self.ffn1(x_ffn1_ln)
    x_ffn1_out = 0.5 * nn.dropout(x_ffn1, dropout=self.dropout) + inp

    # MHSA
    x_mhsa_ln = nn.layer_norm(x_ffn1_out)
    x_mhsa = self.self_att(x_mhsa_ln)
    x_mhsa_out = x_mhsa + x_ffn1_out

    # Conv
    x_conv_ln = nn.layer_norm(x_mhsa_out)
    x_conv = self.conv_block(x_conv_ln)
    x_conv_out = nn.dropout(x_conv, dropout=self.dropout) + x_mhsa_out

    # FFN
    x_ffn2_ln = nn.layer_norm(x_conv_out)
    x_ffn2 = self.ffn2(x_ffn2_ln)
    x_ffn2_out = 0.5 * nn.dropout(x_ffn2, dropout=self.dropout) + x_conv_out

    # last LN layer
    return nn.layer_norm(x_ffn2_out)


class ConformerEncoder(nn.Module):
  """
  Represents Conformer encoder architecture
  """

  def __init__(self, encoder_layer: ConformerEncoderLayer, num_layers: int):
    """
    :param encoder_layer: an instance of :class:`ConformerEncoderLayer`
    :param num_layers: the number of encoder layers
    """
    super().__init__()

    self.dropout = encoder_layer.dropout
    self.out_dim = encoder_layer.out_dim

    self.conv_subsample_layer = ConformerConvSubsample(
      filter_sizes=[(3, 3), (3, 3)], pool_sizes=[(2, 2), (2, 2)], channel_sizes=[self.out_dim, self.out_dim],
      dropout=self.dropout)

    self.linear = nn.Linear(n_out=self.out_dim, with_bias=False)

    self.layers = nn.Sequential(copy.deepcopy(encoder_layer) for _ in range(num_layers))

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    x_subsample = self.conv_subsample_layer(inp)
    x_linear = self.linear(x_subsample)
    x = nn.dropout(x_linear, dropout=self.dropout)
    x = self.layers(x)
    return x
