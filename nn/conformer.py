"""
Conformer code.
Ref: https://arxiv.org/abs/2005.08100
"""

from typing import Tuple, List, Union
from .. import nn


class _PositionwiseFeedForward(nn.Module):
  """
  Conformer position-wise feedforward neural network layer
      FF -> Activation -> Dropout -> FF
  """

  def __init__(self, out_dim: int, dim_ff: int, dropout: float, activation):
    """
    :param out_dim:
    :param dim_ff:
    :param dropout:
    :param activation:
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


class _ConformerConvBlock(nn.Module):
  """
  Conformer convolution block
      FF -> GLU -> depthwise conv -> BN -> Swish -> FF
  """

  def __init__(self, out_dim: int, kernel_size: int, batch_norm_eps: float = 1e-5,
      batch_norm_momentum: float = 0.1, batch_norm_other_opts=None):
    """
    :param out_dim:
    :param kernel_size:
    """
    super().__init__()

    self.positionwise_conv1 = nn.Linear(n_out=out_dim * 2)
    self.depthwise_conv = nn.Conv(n_out=out_dim, filter_size=(kernel_size,), groups=out_dim, padding='same')
    self.positionwise_conv2 = nn.Linear(n_out=out_dim)

    if batch_norm_other_opts is None:
      batch_norm_other_opts = {}
    self.batch_norm = nn.BatchNorm(
      epsilon=batch_norm_eps, momentum=batch_norm_momentum, update_sample_only_in_training=True,
      delay_sample_update=True, **batch_norm_other_opts)

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    x_conv1 = self.positionwise_conv1(inp)
    x_act = nn.glu(x_conv1)
    x_depthwise_conv = self.depthwise_conv(x_act)
    x_bn = self.batch_norm(x_depthwise_conv)
    x_swish = nn.swish(x_bn)
    x_conv2 = self.positionwise_conv2(x_swish)
    return x_conv2


class _ConformerConvSubsampleLayer(nn.Module):
  """
  Conv 2D block with optional max-pooling
  """

  def __init__(self, filter_sizes: List[Tuple[int, ...]], pool_sizes: Union[List[Tuple[int, ...]], None],
      channel_sizes: List[int], dropout: float = 0.3, activation: str = 'relu',
      padding: str = 'same'):
    """
    :param filter_sizes:
    :param pool_sizes:
    :param channel_sizes:
    :param dropout:
    :param activation:
    :param padding:
    """
    super().__init__()

    self.dropout = dropout
    self.pool_sizes = pool_sizes

    self.conv_layers = nn.ModuleList()
    for filter_size, channel_size in zip(filter_sizes, channel_sizes):
      self.conv_layers.append(
        nn.Conv(activation=activation, filter_size=filter_size, n_out=channel_size, padding=padding))

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    x = nn.split_dims(inp, axis='F', dims=(-1, 1))
    for i, conv_layer in enumerate(self.conv_layers):
      x = conv_layer(x)
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

  def __init__(self, conv_kernel_size: int, activation_ff, dim_ff: int, dropout: float, att_dropout: float,
      enc_key_dim: int, num_heads: int):
    """
    :param conv_kernel_size:
    :param activation_ff:
    :param ff_dim:
    :param dropout:
    :param att_dropout:
    :param enc_key_dim:
    :param num_heads:
    """
    super().__init__()

    self.dropout = dropout

    self.ffn1 = _PositionwiseFeedForward(
      out_dim=enc_key_dim, dim_ff=dim_ff, dropout=dropout, activation=activation_ff)

    self.ffn2 = _PositionwiseFeedForward(
      out_dim=enc_key_dim, dim_ff=dim_ff, dropout=dropout, activation=activation_ff)

    self.conv_block = _ConformerConvBlock(out_dim=enc_key_dim, kernel_size=conv_kernel_size)

    self.self_att = MultiheadAttention(enc_key_dim, num_heads, dropout=att_dropout)  # TODO: to be implemented

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

  def __init__(self, encoder_layer: nn.Module, num_blocks: int, conv_kernel_size: int = 32,
      activation_ff=nn.swish, dim_ff: int = 512, dropout: float = 0.1, att_dropout: float = 0.1, enc_key_dim: int = 256,
      num_heads: int = 4):
    """
    :param encoder_layer:
    :param num_blocks:
    :param conv_kernel_size:
    :param ff_act:
    :param ff_dim:
    :param dropout:
    :param att_dropout:
    :param enc_key_dim:
    :param att_n_heads:
    """
    super().__init__()

    self.dropout = dropout

    self.conv_subsample_layer = _ConformerConvSubsampleLayer(
      filter_sizes=[(3, 3), (3, 3)], pool_sizes=[(2, 2), (2, 2)], channel_sizes=[enc_key_dim, enc_key_dim],
      dropout=dropout)

    self.linear = nn.Linear(n_out=enc_key_dim, with_bias=False)

    self.conformer_blocks = nn.Sequential(
      encoder_layer(
        conv_kernel_size=conv_kernel_size, activation_ff=activation_ff, dim_ff=dim_ff, dropout=dropout,
        att_dropout=att_dropout, enc_key_dim=enc_key_dim, num_heads=num_heads
      )
      for _ in range(num_blocks)
    )

  def forward(self, inp: nn.LayerRef) -> nn.LayerRef:
    x_subsample = self.conv_subsample_layer(inp)
    x_linear = self.linear(x_subsample)
    x = nn.dropout(x_linear, dropout=self.dropout)
    x = self.conformer_blocks(x)
    return x
