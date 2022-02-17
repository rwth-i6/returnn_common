"""
Transformer Modules

The API is partly inspired from:
https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
"""

from __future__ import annotations

import copy as _copy
from typing import Optional, Any, Union, Callable, Tuple
from .. import nn


class TransformerEncoderLayer(nn.Module):
  """
  Defines one layer of a standard transformer encoder
  """
  def __init__(self, out_dim: nn.Dim, *,
               self_attention: Union[nn.SelfAttention, Any],
               dim_ff: nn.Dim,
               dropout: float = 0.1,
               activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu,
               norm_eps: float = 1e-6,
               norm_first: bool = True,
               norm=nn.layer_norm) -> None:
    """
    :param out_dim: output dimension, PyTorch name: d_model
    :param self_attention: module which does self attention
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param dropout: Dropout value, PyTorch name: dropout
    :param activation: activation function
    :param norm_eps: Epsilon value for layer normalization
    :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
    :param norm: normalization function
    """
    super().__init__()
    self.self_attn = _copy.deepcopy(self_attention)

    self.linear_ff = nn.Linear(dim_ff)
    self.linear_out = nn.Linear(out_dim)
    self.activation = activation
    self.norm_first = norm_first
    self.norm_eps = norm_eps
    self.norm = norm
    self.dropout = dropout

  @nn.scoped
  def __call__(self, inp: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
    """
    Two possible forward variants of encoder, defined by self.norm_first.
    The input has shape {B, T, F}.
    """
    if self.norm_first:
      inp = inp + self._self_attention_block(self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim), axis=axis)
      inp = inp + self._feed_forward_block(self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim))
    else:
      inp = self.norm(inp + self._self_attention_block(inp, axis=axis), epsilon=self.norm_eps, in_dim=inp.feature_dim)
      inp = self.norm(inp + self._feed_forward_block(inp), epsilon=self.norm_eps, in_dim=inp.feature_dim)

    return inp

  def _self_attention_block(self, inp: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
    inp = self.self_attn(inp, axis=axis)
    return nn.dropout(inp, self.dropout, axis=inp.feature_dim)

  def _feed_forward_block(self, inp: nn.Tensor) -> nn.Tensor:
    inp = self.linear_ff(inp)
    inp = self.activation(inp)
    inp = nn.dropout(inp, dropout=self.dropout, axis=inp.feature_dim)
    inp = self.linear_out(inp)
    inp = nn.dropout(inp, dropout=self.dropout, axis=inp.feature_dim)
    return inp


class TransformerEncoder(nn.Module):
  """
  Defines the full Encoder of the standard transformer
  """
  def __init__(self, encoder_layer: Union[TransformerEncoderLayer, Any], *, num_layers: int,
               norm=nn.layer_norm, norm_eps: float = 1e-6):
    """
    :param encoder_layer: Encoder layer to be stacked num_layers times
    :param num_layers: Number of layers
    :param norm: normalization function
    :param norm_eps: Epsilon value for layer normalization
    """
    super().__init__()
    import copy
    self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    self.num_layers = num_layers
    self.norm = norm
    self.norm_eps = norm_eps

  @nn.scoped
  def __call__(self, inp: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
    """
    Applies every encoder layer initialized in self.layers.
    """
    output = inp

    for mod in self.layers:
      output = mod(output, axis=axis)

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps, in_dim=output.feature_dim)

    return output


class TransformerDecoderLayer(nn.Module):
  """
  Defines one layer of a standard transformer decoder
  """
  def __init__(self, out_dim: nn.Dim, *,
               enc_dec_attention: nn.AttentionFunc,
               causal_self_attention: Union[nn.CausalSelfAttention, Any],
               dim_ff: nn.Dim,
               dropout: float = 0.1,
               activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu,
               norm_eps: float = 1e-6,
               norm_first: bool = True,
               norm=nn.layer_norm):
    """
    :param out_dim: output dimension, PyTorch name: d_model
    :param enc_dec_attention: module or func which does encoder decoder attention
    :param causal_self_attention: module or func which does causal self attention
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param dropout: Dropout value
    :param activation: activation function
    :param norm_eps: Epsilon value for layer normalization
    :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
    :param norm: normalization function
    """
    super().__init__()
    self.self_attn = _copy.deepcopy(causal_self_attention)
    self.attn = enc_dec_attention

    self.linear_ff = nn.Linear(dim_ff)
    self.linear_out = nn.Linear(out_dim)

    self.norm = norm
    self.norm_first = norm_first
    self.norm_eps = norm_eps
    self.activation = activation
    self.dropout = dropout

  @nn.scoped
  def __call__(self, inp: nn.Tensor, *,
               axis: nn.Dim,
               memory: nn.Tensor,
               memory_spatial_axis: nn.Dim,
               state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
    """
    Two possible forward variants of decoder, defined by self.norm_first, inp and memory have shape {B, T, F}
    """
    new_state = nn.LayerState()
    if self.norm_first:
      x_, new_state.self_attn = self._self_attention_block(
        self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim), axis=axis, state=state.self_attn)
      inp = inp + x_
      inp = inp + self._multi_head_attention_block(
        self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim), mem=memory, mem_axis=memory_spatial_axis)
      inp = inp + self._feed_forward_block(self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim))
    else:
      x_, new_state.self_attn = self._self_attention_block(inp, axis=axis, state=state.self_attn)
      inp = self.norm(inp + x_, epsilon=self.norm_eps, in_dim=inp.feature_dim)
      inp = self.norm(
        inp + self._multi_head_attention_block(inp, mem=memory, mem_axis=memory_spatial_axis),
        epsilon=self.norm_eps, in_dim=inp.feature_dim)
      inp = self.norm(inp + self._feed_forward_block(inp), epsilon=self.norm_eps, in_dim=inp.feature_dim)

    return inp, new_state

  def default_initial_state(self) -> nn.LayerState:
    """
    initial state declaration
    """
    return nn.LayerState(
      self_attn=self.self_attn.default_initial_state())

  def _self_attention_block(self,
                            inp: nn.Tensor, *, axis: nn.Dim, state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
    inp, new_state = self.self_attn(inp, axis=axis, state=state)
    return nn.dropout(inp, self.dropout, axis=inp.feature_dim), new_state

  def _multi_head_attention_block(self, inp: nn.Tensor, *, mem: nn.Tensor, mem_axis: nn.Dim) -> nn.Tensor:
    inp = self.attn(query=inp, keys=mem, values=mem, axis=mem_axis, key_dim=inp.feature_dim)
    return nn.dropout(inp, self.dropout, axis=inp.feature_dim)

  def _feed_forward_block(self, inp: nn.Tensor) -> nn.Tensor:
    inp = self.linear_ff(inp)
    inp = self.activation(inp)
    inp = nn.dropout(inp, dropout=self.dropout, axis=inp.feature_dim)
    inp = self.linear_out(inp)
    inp = nn.dropout(inp, dropout=self.dropout, axis=inp.feature_dim)
    return inp


class TransformerDecoder(nn.Module):
  """
  Defines the full decoder of the standard transformer
  """
  def __init__(self, *,
               decoder_layer: Union[TransformerDecoderLayer, Any],
               num_layers: int,
               norm=nn.layer_norm,
               norm_eps: float = 1e-6):
    """
    :param decoder_layer: Decoder layer to be stacked num_layers times
    :param num_layers: Number of layers
    :param norm: normalization function for output layer normalization
    :param norm_eps: Epsilon value for output layer normalization
    """
    super().__init__()
    import copy
    self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    self.num_layers = num_layers
    self.norm = norm
    self.norm_eps = norm_eps

  @nn.scoped
  def __call__(self, inp: nn.Tensor, *,
               axis: nn.Dim,
               memory: nn.Tensor,
               memory_spatial_axis: nn.Dim,
               state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
    """
    Applies every decoder layer initialized in self.layers.
    """
    output = inp

    for key, mod in self.layers.named_children():
      output, state[key] = mod(
        output, axis=axis, memory=memory, memory_spatial_axis=memory_spatial_axis, state=state[key])

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps, in_dim=output.feature_dim)

    return output, state

  def default_initial_state(self) -> nn.LayerState:
    """
    initial state declaration
    """
    return nn.LayerState({key: mod.default_initial_state() for (key, mod) in self.layers.named_children()})


class Transformer(nn.Module):
  """
  Standard Transformer Module
  """
  def __init__(self,
               out_dim: nn.Dim = nn.FeatureDim("output_dim", 512),
               *,
               target_vocab: nn.Dim,
               target_bos_symbol: int = 0,
               target_eos_symbol: int = 0,
               num_heads: Union[nn.Dim, int] = 8,
               num_encoder_layers: int = 6,
               num_decoder_layers: int = 6,
               dim_ff: nn.Dim = nn.NotSpecified,
               dropout: float = 0.1,
               att_dropout: float = 0.1,
               activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu,
               custom_encoder: Optional[Union[TransformerEncoder, Any]] = None,
               custom_decoder: Optional[Union[TransformerDecoder, Any]] = None,
               custom_encoder_layer: Optional[Union[TransformerEncoderLayer, Any]] = None,
               custom_decoder_layer: Optional[Union[TransformerDecoderLayer, Any]] = None,
               norm_eps: float = 1e-6,
               norm=nn.layer_norm,
               norm_first: bool = True,
               dec_causal_self_attention=None,
               enc_self_attention=None,
               enc_dec_attention=None
               ) -> None:
    """
    Default parameters as in the original paper https://arxiv.org/pdf/1706.03762.pdf only modification to this is
    norm_first which would be False in the paper, but empirically performs better with True, thus being True by default.

    :param out_dim: output dimension, PyTorch name: d_model
    :param target_vocab:
    :param target_bos_symbol: beginning of sentence/sequence symbol
    :param target_eos_symbol: end of sentence/sequence symbol
    :param num_heads: number heads, PyTorch name: nhead
    :param num_encoder_layers: Number of encoder layers
    :param num_decoder_layers: Number of decoder layers
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward. 4 * out_dim by default.
    :param dropout: Dropout value, PyTorch name: dropout
    :param att_dropout: dropout value for attention
    :param activation: activation function
    :param custom_encoder: Custom Encoder to replace the standard encoder
    :param custom_decoder: Custom Decoder to replace the standard decoder
    :param custom_encoder_layer: Custom Encoder layer to replace the standard layer if custom_encoder and
      custom_encoder_layer are given custom_encoder will be preferred
    :param custom_decoder_layer: Custom Decoder layer to replace the standard layer if custom_decoder and
      custom_decoder_layer are given custom_decoder will be preferred
    :param norm_eps: Epsilon value for layer normalization
    :param norm: function for layer normalization
    :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
    :param dec_causal_self_attention: module which does stepwise self attention for the decoder
    :param enc_self_attention: module which does self attention for the encoder
    :param enc_dec_attention: module which does encoder decoder attention
    """
    super().__init__()

    if isinstance(num_heads, int):
      num_heads = nn.SpatialDim("num_heads", num_heads)

    if custom_encoder is not None:
      self.encoder = custom_encoder
    else:
      if custom_encoder_layer is not None:
        encoder_layer = custom_encoder_layer
      else:
        if enc_self_attention is None:
          enc_self_attention = nn.SelfAttention(
            key_dim_total=out_dim, value_dim_total=out_dim, num_heads=num_heads, att_dropout=att_dropout)
        encoder_layer = TransformerEncoderLayer(
          out_dim=out_dim, dim_ff=dim_ff, dropout=dropout, activation=activation, norm_eps=norm_eps, norm=norm,
          norm_first=norm_first, self_attention=enc_self_attention)
      self.encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=norm, norm_eps=norm_eps)

    self.target_vocab = target_vocab
    self.target_bos_symbol = target_bos_symbol
    self.target_eos_symbol = target_eos_symbol
    self.target_embedding = nn.Linear(out_dim)

    if custom_decoder is not None:
      self.decoder = custom_decoder
    else:
      if custom_decoder_layer is not None:
        decoder_layer = custom_decoder_layer
      else:
        if dec_causal_self_attention is None:
          dec_causal_self_attention = nn.CausalSelfAttention(
            key_dim_total=out_dim, value_dim_total=out_dim, num_heads=num_heads, att_dropout=att_dropout)
        if enc_dec_attention is None:
          enc_dec_attention = nn.dot_attention
        if dim_ff is nn.NotSpecified:
          dim_ff = out_dim * 4
        decoder_layer = TransformerDecoderLayer(
          out_dim=out_dim, dim_ff=dim_ff, dropout=dropout, activation=activation, norm_eps=norm_eps, norm=norm,
          norm_first=norm_first, causal_self_attention=dec_causal_self_attention, enc_dec_attention=enc_dec_attention)
      self.decoder = TransformerDecoder(
        decoder_layer=decoder_layer, num_layers=num_decoder_layers, norm=norm, norm_eps=norm_eps)

    self.output_projection = nn.Linear(target_vocab)

    self.norm_eps = norm_eps
    self.out_dim = out_dim
    self.num_heads = num_heads
    self.norm = norm

  @nn.scoped
  def __call__(self, source: nn.Tensor, *,
               source_spatial_axis: nn.Dim,
               target: Optional[Union[nn.Tensor, nn.SearchFuncInterface]] = None,
               initial_state: Optional[nn.LayerState] = None,
               ) -> Tuple[nn.Tensor, nn.LayerState]:
    """
    Forward step of Transformer
    """
    memory = self.encoder(source, axis=source_spatial_axis)
    search = None
    if isinstance(target, nn.SearchFuncInterface):
      search = target
      target = None
    loop = nn.Loop()
    loop.state = initial_state if initial_state else self.default_initial_state()
    with loop:
      prev_target_embed = self.target_embedding(loop.state.target)
      output, loop.state.decoder = self.decoder(
        prev_target_embed, axis=nn.single_step_dim,
        memory=memory, memory_spatial_axis=source_spatial_axis, state=loop.state.decoder)
      logits = self.output_projection(output)
      target = loop.unstack(target) if target is not None else None
      if search:
        search.apply_loop(loop)
        loop.state.target = search.choice(output=logits, output_type="logits")
        loop.end(loop.state.target == self.target_eos_symbol, include_eos=False)
      else:
        assert target is not None
        loop.state.target = target
      outputs = loop.stack(loop.state.target)
    return outputs, loop.state

  def default_initial_state(self) -> nn.LayerState:
    """
    initial state declaration
    """
    return nn.LayerState(
      target=nn.constant(value=self.target_bos_symbol, shape=[nn.batch_dim], sparse_dim=self.target_vocab),
      decoder=self.decoder.default_initial_state())
