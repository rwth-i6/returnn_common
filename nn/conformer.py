"""
Conformer code.
Ref: https://arxiv.org/abs/2005.08100
"""

from typing import Tuple, List, Union
from . import Module, ModuleList, LayerRef, Linear, dropout, layer_norm, batch_norm, Conv, swish, glu, split_dims, \
    merge_dims, pool


class _PositionwiseFeedForward(Module):
    """
    Conformer position-wise feedforward neural network layer
        FF -> Activation -> Dropout -> FF
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float, activation, l2: float = 0.0):
        """
        :param d_model:
        :param d_ff:
        :param dropout:
        :param activation:
        :param l2:
        """
        super().__init__()

        self.dropout = dropout
        self.activation = activation

        self.linear1 = Linear(n_out=d_ff, l2=l2)
        self.linear2 = Linear(n_out=d_model, l2=l2)

    def forward(self, inp: LayerRef) -> LayerRef:
        return self.linear2(dropout(self.activation(self.linear1(inp)), dropout=self.dropout))


class _ConformerConvBlock(Module):
    """
    Conformer convolution block
        FF -> GLU -> depthwise conv -> BN -> Swish -> FF
    """

    def __init__(self, d_model: int, kernel_size: Tuple[int], l2: float = 0.0):
        """
        :param d_model:
        :param kernel_size:
        :param l2:
        """
        super().__init__()

        self.positionwise_conv1 = Linear(n_out=d_model * 2, l2=l2)
        self.depthwise_conv = Conv(n_out=d_model, filter_size=kernel_size, groups=d_model, l2=l2, padding='same')
        self.positionwise_conv2 = Linear(n_out=d_model, l2=l2)

    def forward(self, inp: LayerRef) -> LayerRef:
        x_conv1 = self.positionwise_conv1(inp)
        x_act = glu(x_conv1)
        x_depthwise_conv = self.depthwise_conv(x_act)
        x_bn = batch_norm(x_depthwise_conv)
        x_swish = swish(x_bn)
        x_conv2 = self.positionwise_conv2(x_swish)
        return x_conv2


class _ConformerConvSubsampleLayer(Module):
    """
    Conv 2D block with optional max-pooling
    """

    def __init__(self, filter_sizes: List[Tuple[int, ...]], pool_sizes: Union[List[Tuple[int, ...]], None],
                 channel_sizes: List[int], l2: float = 0.0, dropout: float = 0.3, act: str = 'relu',
                 padding: str = 'same'):
        """
        :param filter_sizes:
        :param pool_sizes:
        :param channel_sizes:
        :param l2:
        :param dropout:
        :param act:
        :param padding:
        """
        super().__init__()

        self.dropout = dropout
        self.pool_sizes = pool_sizes

        self.conv_layers = ModuleList()
        for filter_size, channel_size in zip(filter_sizes, channel_sizes):
            self.conv_layers.append(
                Conv(l2=l2, activation=act, filter_size=filter_size, n_out=channel_size, padding=padding))

    def forward(self, inp: LayerRef) -> LayerRef:
        x = split_dims(inp, axis='F', dims=(-1, 1))
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if self.pool_sizes and i < len(self.pool_sizes):
                x = pool(x, pool_size=self.pool_sizes[i], padding='same', mode='max')
            if self.dropout:
                x = dropout(x, dropout=self.dropout)
        out = merge_dims(x, axes='static')
        return out


class ConformerEncoderLayer(Module):
    """
    Represents a conformer block
    """

    def __init__(self, conv_kernel_size: Tuple[int], ff_act, ff_dim: int, dropout: float, att_dropout: float,
                 enc_key_dim: int, att_n_heads: int, l2: float):
        """
        :param conv_kernel_size:
        :param ff_act:
        :param ff_dim:
        :param dropout:
        :param att_dropout:
        :param enc_key_dim:
        :param att_n_heads:
        :param l2:
        """
        super().__init__()

        self.dropout = dropout

        self.ffn1 = _PositionwiseFeedForward(
            d_model=enc_key_dim, d_ff=ff_dim, dropout=dropout, activation=ff_act, l2=l2)

        self.ffn2 = _PositionwiseFeedForward(
            d_model=enc_key_dim, d_ff=ff_dim, dropout=dropout, activation=ff_act, l2=l2)

        self.conv_module = _ConformerConvBlock(d_model=enc_key_dim, kernel_size=conv_kernel_size)

        self.mhsa_module = MultiheadAttention(d_model, att_n_heads, dropout=att_dropout)  # TODO: to be implemented

    def forward(self, inp: LayerRef) -> LayerRef:
        # FFN
        x_ffn1_ln = layer_norm(inp)
        x_ffn1 = self.ffn1(x_ffn1_ln)
        x_ffn1_out = 0.5 * dropout(x_ffn1, dropout=self.dropout) + inp

        # MHSA
        x_mhsa_ln = layer_norm(x_ffn1_out)
        x_mhsa = self.mhsa_module(x_mhsa_ln)
        x_mhsa_out = x_mhsa + x_ffn1_out

        # Conv
        x_conv_ln = layer_norm(x_mhsa_out)
        x_conv = self.conv_module(x_conv_ln)
        x_conv_out = dropout(x_conv, dropout=self.dropout) + x_mhsa_out

        # FFN
        x_ffn2_ln = layer_norm(x_conv_out)
        x_ffn2 = self.ffn2(x_ffn2_ln)
        x_ffn2_out = 0.5 * dropout(x_ffn2, dropout=self.dropout) + x_conv_out

        # last LN layer
        return layer_norm(x_ffn2_out)


class ConformerEncoder(Module):
    """
    Represents Conformer encoder architecture
    """

    def __init__(self, encoder_layer: Module, num_blocks: int, conv_kernel_size: Tuple[int, ...] = (32,), ff_act=swish,
                 ff_dim: int = 512, dropout: float = 0.1, att_dropout: float = 0.1, enc_key_dim: int = 256,
                 att_n_heads: int = 4, l2: float = 0.0):
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
        :param l2:
        """
        super().__init__()

        self.dropout = dropout

        self.conv_subsample_layer = _ConformerConvSubsampleLayer(
            filter_sizes=[(3, 3), (3, 3)], pool_sizes=[(2, 2), (2, 2)], channel_sizes=[enc_key_dim, enc_key_dim],
            l2=l2, dropout=dropout)

        self.linear = Linear(n_out=enc_key_dim, l2=l2, with_bias=False)

        self.conformer_blocks = ModuleList([
            encoder_layer(
                conv_kernel_size=conv_kernel_size, ff_act=ff_act, ff_dim=ff_dim, dropout=dropout,
                att_dropout=att_dropout, enc_key_dim=enc_key_dim, att_n_heads=att_n_heads, l2=l2
            )
            for _ in range(num_blocks)
        ])

    def forward(self, inp: LayerRef) -> LayerRef:
        x_subsample = self.conv_subsample_layer(inp)
        x_linear = self.linear(x_subsample)
        x = dropout(x_linear, dropout=self.dropout)
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x)
        return x
