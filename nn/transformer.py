"""
Transformer Modules

The API is partly inspired from:
https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
"""

from __future__ import annotations

import copy as _copy
from typing import Optional, Any, Union, Callable, Tuple, Sequence
from .. import nn


class TransformerEncoderLayer(nn.Module):
    """
    Defines one layer of a standard transformer encoder
    """

    def __init__(
        self,
        out_dim: nn.Dim,
        *,
        self_attention: Union[nn.SelfAttention, Any],
        ff_dim: nn.Dim,
        ff_activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
        norm_first: bool = True,
        norm=nn.LayerNorm,
    ) -> None:
        """
        :param out_dim: output dimension, PyTorch name: d_model
        :param self_attention: module which does self attention
        :param ff_dim: dimension of feedforward layer, PyTorch name: dim_feedforward
        :param ff_activation: activation function
        :param dropout: Dropout value, PyTorch name: dropout
        :param norm_eps: Epsilon value for layer normalization
        :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
        :param norm: normalization function, e.g. nn.LayerNorm
        """
        super().__init__()
        self.self_attn = _copy.deepcopy(self_attention)

        self.linear_ff = nn.Linear(out_dim, ff_dim)
        self.linear_out = nn.Linear(ff_dim, out_dim)
        self.activation = ff_activation
        self.norm_first = norm_first
        assert isinstance(norm, type)
        self.self_att_norm = norm(out_dim, eps=norm_eps)
        self.ff_norm = norm(out_dim)
        self.dropout = dropout

    def __call__(self, inp: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
        """
        Two possible forward variants of encoder, defined by self.norm_first.
        The input has shape {B, T, F}.
        """
        if self.norm_first:
            inp = inp + self._self_attention_block(self.self_att_norm(inp), axis=axis)
            inp = inp + self._feed_forward_block(self.ff_norm(inp))
        else:
            inp = self.self_att_norm(inp + self._self_attention_block(inp, axis=axis))
            inp = self.ff_norm(inp + self._feed_forward_block(inp))

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

    def __init__(
        self,
        encoder_layer: Union[TransformerEncoderLayer, Any],
        *,
        num_layers: int,
        norm: Optional[Callable[[nn.Tensor], nn.Tensor]],
    ):
        """
        :param encoder_layer: Encoder layer to be stacked num_layers times (copies of it, no param sharing)
        :param num_layers: Number of layers
        :param norm: normalization function, e.g. nn.LayerNorm()
        """
        super().__init__()
        import copy

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        self.num_layers = num_layers
        self.norm = norm

    def __call__(self, inp: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
        """
        Applies every encoder layer initialized in self.layers.
        """
        output = inp

        for mod in self.layers:
            output = mod(output, axis=axis)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    """
    Defines one layer of a standard transformer decoder
    """

    def __init__(
        self,
        out_dim: nn.Dim,
        *,
        enc_dec_attention: nn.AttentionFunc,
        causal_self_attention: Union[nn.CausalSelfAttention, Any],
        ff_dim: nn.Dim,
        ff_activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu,
        dropout: float = 0.1,
        norm_eps: float = 1e-6,
        norm_first: bool = True,
        norm=nn.LayerNorm,
    ):
        """
        :param out_dim: output dimension, PyTorch name: d_model
        :param enc_dec_attention: module or func which does encoder decoder attention
        :param causal_self_attention: module or func which does causal self attention
        :param ff_dim: dimension of feedforward layer, PyTorch name: dim_feedforward
        :param ff_activation: activation function
        :param dropout: Dropout value
        :param norm_eps: Epsilon value for layer normalization
        :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
        :param norm: normalization function, e.g. nn.LayerNorm()
        """
        super().__init__()
        self.self_attn = _copy.deepcopy(causal_self_attention)
        self.attn = enc_dec_attention

        self.linear_ff = nn.Linear(out_dim, ff_dim)
        self.linear_out = nn.Linear(ff_dim, out_dim)

        assert isinstance(norm, type)
        self.self_att_norm = norm(out_dim, eps=norm_eps)
        self.cross_att_norm = norm(out_dim, eps=norm_eps)
        self.ff_norm = norm(out_dim, eps=norm_eps)
        self.norm_first = norm_first
        self.activation = ff_activation
        self.dropout = dropout

    def __call__(
        self, inp: nn.Tensor, *, axis: nn.Dim, memory: nn.Tensor, memory_spatial_axis: nn.Dim, state: nn.LayerState
    ) -> Tuple[nn.Tensor, nn.LayerState]:
        """
        Two possible forward variants of decoder, defined by self.norm_first, inp and memory have shape {B, T, F}
        """
        new_state = nn.LayerState()
        if self.norm_first:
            x_, new_state.self_attn = self._self_attention_block(
                self.self_att_norm(inp), axis=axis, state=state.self_attn
            )
            inp = inp + x_
            inp = inp + self._multi_head_attention_block(
                self.cross_att_norm(inp), mem=memory, mem_axis=memory_spatial_axis
            )
            inp = inp + self._feed_forward_block(self.ff_norm(inp))
        else:
            x_, new_state.self_attn = self._self_attention_block(inp, axis=axis, state=state.self_attn)
            inp = self.self_att_norm(inp + x_)
            inp = self.cross_att_norm(
                inp + self._multi_head_attention_block(inp, mem=memory, mem_axis=memory_spatial_axis)
            )
            inp = self.ff_norm(inp + self._feed_forward_block(inp))

        return inp, new_state

    def _self_attention_block(
        self, inp: nn.Tensor, *, axis: nn.Dim, state: nn.LayerState
    ) -> Tuple[nn.Tensor, nn.LayerState]:
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

    def __init__(
        self,
        *,
        decoder_layer: Union[TransformerDecoderLayer, Any],
        num_layers: int,
        norm: Optional[Callable[[nn.Tensor], nn.Tensor]],
    ):
        """
        :param decoder_layer: Decoder layer to be stacked num_layers times
        :param num_layers: Number of layers
        :param norm: normalization function for output layer normalization
        """
        super().__init__()
        import copy

        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

        self.num_layers = num_layers
        self.norm = norm

    def __call__(
        self, inp: nn.Tensor, *, axis: nn.Dim, memory: nn.Tensor, memory_spatial_axis: nn.Dim, state: nn.LayerState
    ) -> Tuple[nn.Tensor, nn.LayerState]:
        """
        Applies every decoder layer initialized in self.layers.
        """
        output = inp

        for key, mod in self.layers.named_children():
            output, state[key] = mod(
                output, axis=axis, memory=memory, memory_spatial_axis=memory_spatial_axis, state=state[key]
            )

        if self.norm is not None:
            output = self.norm(output)

        return output, state

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> nn.LayerState:
        """
        initial state declaration
        """
        return self.layers.default_initial_state(batch_dims=batch_dims)


class Transformer(nn.Module):
    """
    Standard Transformer Module
    """

    def __init__(
        self,
        *,
        model_dim: nn.Dim = nn.FeatureDim("transformer_default_model_dim", 512),
        target_dim: nn.Dim,
        target_bos_symbol: int = 0,
        target_eos_symbol: int = 0,
        num_heads: Union[nn.Dim, int] = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        ff_dim: nn.Dim = nn.NotSpecified,
        ff_activation: Callable[[nn.Tensor], nn.Tensor] = nn.relu,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
        encoder: Optional[Union[TransformerEncoder, Any]] = None,
        decoder: Optional[Union[TransformerDecoder, Any]] = None,
        encoder_layer: Optional[Union[TransformerEncoderLayer, Any]] = None,
        decoder_layer: Optional[Union[TransformerDecoderLayer, Any]] = None,
        norm_eps: float = 1e-6,
        norm=nn.LayerNorm,
        norm_first: bool = True,
        dec_causal_self_attention=None,
        enc_self_attention=None,
        cross_attention=None,
    ) -> None:
        """
        Default parameters as in the original paper https://arxiv.org/pdf/1706.03762.pdf only modification to this is
        norm_first which would be False in the paper, but empirically performs better with True,
        thus being True by default.

        :param model_dim: model dimension, out_dim of encoder and decoder. PyTorch name: d_model
        :param target_dim: includes the target vocab. dim for the final softmax in the decoder
        :param target_bos_symbol: beginning of sentence/sequence symbol
        :param target_eos_symbol: end of sentence/sequence symbol
        :param num_heads: number heads, PyTorch name: nhead
        :param num_encoder_layers: Number of encoder layers
        :param num_decoder_layers: Number of decoder layers
        :param ff_dim: dimension of feedforward layer, PyTorch name: dim_feedforward. 4 * out_dim by default.
        :param ff_activation: activation function
        :param dropout: Dropout value, PyTorch name: dropout
        :param att_dropout: dropout value for attention
        :param encoder: Custom Encoder to replace the standard encoder
        :param decoder: Custom Decoder to replace the standard decoder
        :param encoder_layer: Custom Encoder layer to replace the standard layer if custom_encoder and
          custom_encoder_layer are given custom_encoder will be preferred.
          Copies of it will be made for each layer, so there is no automatic param sharing.
        :param decoder_layer: Custom Decoder layer to replace the standard layer if custom_decoder and
          custom_decoder_layer are given custom_decoder will be preferred
          Copies of it will be made for each layer, so there is no automatic param sharing.
        :param norm_eps: Epsilon value for layer normalization
        :param norm: function for layer normalization
        :param norm_first: if ``True`` will perform normalization before other att and ff operations, otherwise after
        :param dec_causal_self_attention: module which does stepwise self attention for the decoder
        :param enc_self_attention: module which does self attention for the encoder
        :param cross_attention: module which does encoder decoder attention
        """
        super().__init__()

        if isinstance(num_heads, int):
            num_heads = nn.SpatialDim("num_heads", num_heads)

        if ff_dim is nn.NotSpecified:
            ff_dim = model_dim * 4

        if encoder is not None:
            self.encoder = encoder
            assert enc_self_attention is None
        else:
            if encoder_layer is not None:
                assert enc_self_attention is None
            else:
                if enc_self_attention is None:
                    enc_self_attention = nn.SelfAttention(
                        in_dim=model_dim,
                        proj_dim=model_dim,
                        key_dim_total=model_dim,
                        value_dim_total=model_dim,
                        num_heads=num_heads,
                        att_dropout=att_dropout,
                    )
                encoder_layer = TransformerEncoderLayer(
                    out_dim=model_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    ff_activation=ff_activation,
                    norm_eps=norm_eps,
                    norm=norm,
                    norm_first=norm_first,
                    self_attention=enc_self_attention,
                )
            self.encoder = TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=num_encoder_layers, norm=norm(model_dim, eps=norm_eps)
            )

        self.target_dim = target_dim
        self.target_bos_symbol = target_bos_symbol
        self.target_eos_symbol = target_eos_symbol
        self.target_embedding = nn.Linear(target_dim, model_dim)

        if decoder is not None:
            self.decoder = decoder
            assert dec_causal_self_attention is None
            assert cross_attention is None
        else:
            if decoder_layer is not None:
                assert dec_causal_self_attention is None
                assert cross_attention is None
            else:
                if dec_causal_self_attention is None:
                    dec_causal_self_attention = nn.CausalSelfAttention(
                        in_dim=model_dim,
                        proj_dim=model_dim,
                        key_dim_total=model_dim,
                        value_dim_total=model_dim,
                        num_heads=num_heads,
                        att_dropout=att_dropout,
                    )
                if cross_attention is None:
                    cross_attention = nn.dot_attention
                decoder_layer = TransformerDecoderLayer(
                    out_dim=model_dim,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    ff_activation=ff_activation,
                    norm_eps=norm_eps,
                    norm=norm,
                    norm_first=norm_first,
                    causal_self_attention=dec_causal_self_attention,
                    enc_dec_attention=cross_attention,
                )
            self.decoder = TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=num_decoder_layers, norm=norm(model_dim, eps=norm_eps)
            )

        self.output_projection = nn.Linear(model_dim, target_dim)

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.norm = norm
        self.norm_eps = norm_eps

    def __call__(
        self,
        source: nn.Tensor,
        *,
        source_spatial_axis: nn.Dim,
        target: Optional[Union[nn.Tensor, nn.SearchFuncInterface]] = None,
        target_spatial_axis: Optional[nn.Dim] = None,
        state: Optional[nn.LayerState] = None,
    ) -> Tuple[nn.Tensor, nn.Tensor, Optional[nn.Tensor], nn.LayerState]:
        """
        Forward step of Transformer

        :return: memory (encoder output), out logits, out labels (only when doing search), final state
        """
        assert (
            self.model_dim in source.dims_set
        ), f"{self}: Input {source} feature dimension is not matching Transformer model dimension {self.model_dim}"
        memory = self.encoder(source, axis=source_spatial_axis)
        search = None
        if isinstance(target, nn.SearchFuncInterface):
            search = target
            target = None
        if target is not None:
            assert target_spatial_axis, f"{self}: Target spatial axis must be specified when target is given"
        loop = nn.Loop(axis=target_spatial_axis)
        loop.state = (
            state
            if state
            else self.default_initial_state(
                batch_dims=memory.remaining_dims(remove=(source_spatial_axis, memory.feature_dim))
            )
        )
        with loop:
            prev_target_embed = self.target_embedding(loop.state.target)
            output, loop.state.decoder = self.decoder(
                prev_target_embed,
                axis=nn.single_step_dim,
                memory=memory,
                memory_spatial_axis=source_spatial_axis,
                state=loop.state.decoder,
            )
            logits = self.output_projection(output)
            target = loop.unstack(target) if target is not None else None
            if search:
                search.apply_loop(loop)
                choice = search.choice(probs=logits, probs_type="logits")
                loop.state.target = choice
                loop.end(loop.state.target == self.target_eos_symbol, include_eos=False)
            else:
                assert target is not None
                loop.state.target = target
            out_labels = loop.stack(loop.state.target) if target is None else None
            out_logits = loop.stack(logits)

        return memory, out_logits, out_labels, loop.state

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> nn.LayerState:
        """
        initial state declaration
        """
        return nn.LayerState(
            target=nn.constant(value=self.target_bos_symbol, shape=batch_dims, sparse_dim=self.target_dim),
            decoder=self.decoder.default_initial_state(batch_dims=batch_dims),
        )
