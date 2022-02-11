"""
Transformer Modules
"""

from __future__ import annotations

import copy
from typing import Optional, Any, Union, Callable, Tuple
from .. import nn


class TransformerEncoderLayer(nn.Module):
  """
  Defines one layer of a standard transformer encoder
  """
  def __init__(self, output_dim: nn.Dim, *, self_attention, dim_ff: nn.Dim, dropout: float = 0.1,
               activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu, norm_eps: float = 1e-6,
               norm_first: bool = True, norm=nn.layer_norm) -> None:
    """
    :param output_dim: output dimension, PyTorch name: d_model
    :param self_attention: module which does self attention
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param dropout: Dropout value, PyTorch name: dropout
    :param activation: activation function
    :param norm_eps: Epsilon value for layer normalization
    :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
    :param norm: normalization function
    """
    super().__init__()
    self.self_attn = copy.deepcopy(self_attention)

    self.linear_ff = nn.Linear(dim_ff)
    self.linear_out = nn.Linear(output_dim)
    self.activation = activation
    self.norm_first = norm_first
    self.norm_eps = norm_eps
    self.norm = norm
    self.dropout = dropout

  def __call__(self, inp: nn.Tensor) -> nn.Tensor:
    """
    Two possible forward variants of encoder, defined by self.norm_first.
    The input has shape {B, T, F}.
    """
    if self.norm_first:
      inp = inp + self._self_attention_block(self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim))
      inp = inp + self._feed_forward_block(self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim))
    else:
      inp = self.norm(inp + self._self_attention_block(inp), epsilon=self.norm_eps, in_dim=inp.feature_dim)
      inp = self.norm(inp + self._feed_forward_block(inp), epsilon=self.norm_eps, in_dim=inp.feature_dim)

    return inp

  def _self_attention_block(self, inp: nn.Tensor) -> nn.Tensor:
    inp = self.self_attn(inp)
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

  def __call__(self, inp: nn.Tensor) -> nn.Tensor:
    """
    Applies every encoder layer initialized in self.layers.
    """
    output = inp

    for mod in self.layers:
      output = mod(output)

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps, in_dim=output.feature_dim)

    return output


class TransformerDecoderLayerStep(nn.Module):
  """
  Defines one layer of a standard transformer decoder
  """
  def __init__(self, output_dim: nn.Dim, *, enc_dec_attention, self_attention_step, dim_ff: nn.Dim,
               dropout: float = 0.1, activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu, norm_eps: float = 1e-6,
               norm_first: bool = True, norm=nn.layer_norm):
    """
    :param output_dim: output dimension, PyTorch name: d_model
    :param enc_dec_attention: module which does encoder decoder attention
    :param self_attention_step: module which does stepwise self attention
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
    :param dropout: Dropout value, PyTorch name: dropout
    :param activation: activation function
    :param norm_eps: Epsilon value for layer normalization
    :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
    :param norm: normalization function
    """
    super().__init__()
    self.self_attn = copy.deepcopy(self_attention_step)
    self.attn = enc_dec_attention

    self.linear_ff = nn.Linear(dim_ff)
    self.linear_out = nn.Linear(output_dim)

    self.norm = norm
    self.norm_first = norm_first
    self.norm_eps = norm_eps
    self.activation = activation
    self.dropout = dropout

  def __call__(self, inp: nn.Tensor, *, memory: nn.Tensor,
               state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
    """
    Two possible forward variants of decoder, defined by self.norm_first, inp and memory have shape {B, T, F}
    """
    if self.norm_first:
      x_, new_state = self._self_attention_block(
        self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim), state=state)
      inp = inp + x_
      inp = inp + self._multi_head_attention_block(
        self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim), memory)
      inp = inp + self._feed_forward_block(self.norm(inp, epsilon=self.norm_eps, in_dim=inp.feature_dim))
    else:
      x_, new_state = self._self_attention_block(inp, state=state)
      inp = self.norm(inp + x_, epsilon=self.norm_eps, in_dim=inp.feature_dim)
      inp = self.norm(
        inp + self._multi_head_attention_block(inp, memory), epsilon=self.norm_eps, in_dim=inp.feature_dim)
      inp = self.norm(inp + self._feed_forward_block(inp), epsilon=self.norm_eps, in_dim=inp.feature_dim)

    return inp, new_state

  def initial_state(self) -> nn.LayerState:
    """
    initial state declaration
    """
    return nn.LayerState(
      self_attn=self.self_attn.initial_state(),
      attn=self.attn.initial_state())

  def _self_attention_block(self, inp: nn.Tensor, *, state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
    inp, new_state = self.self_attn(inp, state=state)
    return nn.dropout(inp, self.dropout, axis=inp.feature_dim), new_state

  def _multi_head_attention_block(self, inp: nn.Tensor, mem: nn.Tensor) -> nn.Tensor:
    inp = self.attn(inp, mem, mem)
    return nn.dropout(inp, self.dropout, axis=inp.feature_dim)

  def _feed_forward_block(self, inp: nn.Tensor) -> nn.Tensor:
    inp = self.linear_ff(inp)
    inp = self.activation(inp)
    inp = nn.dropout(inp, dropout=self.dropout, axis=inp.feature_dim)
    inp = self.linear_out(inp)
    inp = nn.dropout(inp, dropout=self.dropout, axis=inp.feature_dim)
    return inp


class TransformerDecoderStep(nn.Module):
  """
  Defines the full Decoder of the standard transformer
  """
  def __init__(self, decoder_layer: Union[TransformerDecoderLayerStep, Any], num_layers: int,
               norm=nn.layer_norm, norm_eps: float = 1e-6):
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

  def __call__(self, inp: nn.Tensor, *, memory: nn.Tensor,
               state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
    """
    Applies every decoder layer initialized in self.layers.
    """
    output = inp

    for key, mod in self.layers.named_children():
      output, state[key] = mod(output, memory=memory, state=state[key])

    if self.norm is not None:
      output = self.norm(output, epsilon=self.norm_eps, in_dim=output.feature_dim)

    return output, state

  def initial_state(self) -> nn.LayerState:
    """
    initial state declaration
    """
    return nn.LayerState({key: mod.initial_state() for (key, mod) in self.layers.named_children()})


class Transformer(nn.Module):
  """
  Standard Transformer Module
  """
  def __init__(self,
               output_dim: nn.Dim = nn.FeatureDim("output_dim", 512),
               num_heads: int = 8,
               num_encoder_layers: int = 6,
               num_decoder_layers: int = 6,
               dim_ff: nn.Dim = nn.FeatureDim("ff_dim", 2048),
               dropout: float = 0.1,
               att_dropout: float = 0.1,
               activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu,
               custom_encoder: Optional[Any] = None,
               custom_decoder: Optional[Any] = None,
               custom_encoder_layer: Optional[nn.Tensor] = None,
               custom_decoder_layer: Optional[nn.Tensor] = None,
               norm_eps: float = 1e-6,
               norm=nn.layer_norm,
               norm_first: bool = True,
               dec_self_attention_step=None,
               enc_self_attention=None,
               enc_dec_attention=None
               ) -> None:
    """
    Default parameters as in the original paper https://arxiv.org/pdf/1706.03762.pdf only modification to this is
    norm_first which would be False in the paper, but empirically performs better with True, thus being True by default.
    :param output_dim: output dimension, PyTorch name: d_model
    :param num_heads: number heads, PyTorch name: nhead
    :param num_encoder_layers: Number of encoder layers
    :param num_decoder_layers: Number of decoder layers
    :param dim_ff: dimension of feedforward layer, PyTorch name: dim_feedforward
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
    :param dec_self_attention_step: module which does stepwise self attention for the decoder
    :param enc_self_attention: module which does self attention for the encoder
    :param enc_dec_attention: module which does encoder decoder attention
    """
    super().__init__()

    if custom_encoder is not None:
      self.encoder = custom_encoder
    else:
      if custom_encoder_layer is not None:
        encoder_layer = custom_encoder_layer
      else:
        if enc_self_attention is None:
          enc_self_attention = nn.SelfAttention(
            key_dim_total=output_dim, value_dim_total=output_dim, num_heads=num_heads, att_dropout=att_dropout)
        encoder_layer = TransformerEncoderLayer(
          output_dim=output_dim, dim_ff=dim_ff, dropout=dropout, activation=activation, norm_eps=norm_eps, norm=norm,
          norm_first=norm_first, self_attention=enc_self_attention)
      self.encoder = TransformerEncoder(
        encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=norm, norm_eps=norm_eps)

    if custom_decoder is not None:
      self.decoder = custom_decoder
    else:
      if custom_decoder_layer is not None:
        decoder_layer = custom_decoder_layer
      else:
        if dec_self_attention_step is None:
          dec_self_attention_step = nn.SelfAttention(
            key_dim_total=output_dim, value_dim_total=output_dim, num_heads=num_heads, att_dropout=att_dropout)
        if enc_dec_attention is None:
          enc_dec_attention = nn.dot_attention
        decoder_layer = TransformerDecoderLayerStep(
          output_dim=output_dim, dim_ff=dim_ff, dropout=dropout, activation=activation, norm_eps=norm_eps, norm=norm,
          norm_first=norm_first, self_attention_step=dec_self_attention_step, enc_dec_attention=enc_dec_attention)
      self.decoder = TransformerDecoderStep(
        decoder_layer=decoder_layer, num_layers=num_decoder_layers, norm=norm, norm_eps=norm_eps)

    self.norm_eps = norm_eps
    self.output_dim = output_dim
    self.num_heads = num_heads
    self.norm = norm

  def __call__(self, source: nn.Tensor, *, target: Optional[nn.Tensor] = None,
               initial_state: Optional[nn.LayerState] = None,
               search: bool, beam_size: Optional[int] = None, eos_symbol: Optional[int] = None,
               ) -> Tuple[nn.Tensor, nn.LayerState]:
    """
    Forward step of Transformer
    """
    memory = self.encoder(source)
    if not initial_state:
      initial_state = self.initial_state()
    with nn.Loop() as loop:
      loop.state = initial_state
      logits, loop.state.decoder = self.decoder(loop.state.target, memory=memory, state=loop.state.decoder)
      target = loop.unstack(target) if target is not None else None
      if search:
        loop.state.target = nn.choice(logits, input_type="logits", target=target, search=True, beam_size=beam_size)
        loop.end(loop.state.target == eos_symbol, include_eos=False)
      else:
        assert target is not None
        loop.state.target = target
      outputs = loop.stack(loop.state.target)
    return outputs, loop.state

  def initial_state(self, initial_target: nn.Tensor = 0) -> nn.LayerState:
    """
    initial state declaration
    """
    return nn.LayerState(
      target=initial_target,
      decoder=self.decoder.initial_state())