"""
Attention, self-attention, auto-regressive self-attention
"""

from typing import Tuple, Union, Optional, Sequence, Protocol
import weakref
from .. import nn


class AttentionFunc(Protocol):
    """Protocol defining a generic attention function"""

    def __call__(
        self,
        query: nn.Tensor,
        keys: nn.Tensor,
        values: nn.Tensor,
        *,
        key_dim: nn.Dim,
        axis: nn.Dim,
        att_dropout: float = 0.1,
    ):
        ...


def dot_attention(
    query: nn.Tensor, keys: nn.Tensor, values: nn.Tensor, *, key_dim: nn.Dim, axis: nn.Dim, att_dropout: float = 0.1
) -> nn.Tensor:
    """
    Calculates attention over the given axis, for given key dim.
    Any other unrelated axes do not matter here.
    This can be used for multi-head or single head.
    The query can have other dimensions or not.
    """
    query *= key_dim.dimension**-0.5
    energy = nn.dot(query, keys, reduce=key_dim, name="energy")
    att_weights = nn.softmax(energy, axis=axis, name="att_weights")
    att_weights = nn.dropout(att_weights, att_dropout, axis=axis)
    att = nn.dot(att_weights, values, reduce=axis, name="att")
    return att


# noinspection PyAbstractClass
class GenericSelfAttention(nn.Module):
    """
    Shared base class for (non-causal) self attention (:class:`SelfAttention`)
    and causal self attention (:class:`CausalSelfAttention`).

    It uses :func:`dot_attention` for multi-headed dot-attention.
    """

    def __init__(
        self,
        in_dim: nn.Dim,
        proj_dim: Optional[nn.Dim],
        *,
        key_dim_total: nn.Dim,
        value_dim_total: nn.Dim,
        num_heads: Union[int, nn.Dim],
        with_bias: bool = True,
        att_dropout: float = 0.1,
    ):
        """
        :param in_dim: input dim
        :param proj_dim: if given, will add a final linear projection to this dim.
          otherwise no projection after the attention
        :param key_dim_total: total key dim. should be a multiple of num_heads
        :param value_dim_total: total value dim. should be a multiple of num_heads
        :param num_heads: number of heads
        :param with_bias: whether to add bias to qkv and proj linear projections.
          Was False in original Transformer, but many recent implementations use True by default.
          Also see: https://github.com/rwth-i6/returnn_common/issues/234.
        :param att_dropout: dropout for attention weights
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = proj_dim if proj_dim else value_dim_total
        if isinstance(num_heads, int):
            num_heads = nn.SpatialDim("num_heads", num_heads)
        self.key_dim_total = key_dim_total
        self.key_dim_per_head = key_dim_total.div_left(num_heads)
        self.value_dim_total = value_dim_total
        self.value_dim_per_head = value_dim_total.div_left(num_heads)
        self.num_heads = num_heads
        self.qkv_dim_total = 2 * key_dim_total + value_dim_total
        self.qkv_dim_per_head = 2 * self.key_dim_per_head + self.value_dim_per_head
        self.qkv = nn.Linear(in_dim, self.qkv_dim_total, with_bias=with_bias)
        # In Fairseq MultiheadAttention, they use:
        #   nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))  (same for q_proj, v_proj),
        # where xavier_uniform_ means:
        #   std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        #   a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        #   _no_grad_uniform_(tensor, -a, a)
        # Out nn.init.VarianceScaling with mode="fan_avg", distribution="uniform":
        #   scale = scale * 2.0 / float(fan_in + fan_out)
        #   limit = math.sqrt(3.0 * scale)
        #   nn.random(distribution="uniform", minval=-limit, maxval=limit, ...)
        # Our fan_out is 3 times larger than in Fairseq, because we concatenate q,k,v.
        # Assuming fan_in = fan_out, it means a factor 2 in the denominator.
        # So our default (Glorot, which is VarianceScaling with mode="fan_avg", distribution="uniform", scale=1.0)
        # is already the same as Fairseq.
        # The bias init is different, but not sure how important this is.
        if proj_dim:
            self.proj = nn.Linear(value_dim_total, proj_dim, with_bias=with_bias)
        else:
            self.proj = None
        self.att_dropout = att_dropout

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> nn.LayerState:
        """
        For causal attention.
        """
        # Note: This dim tag is wrong. It should match to the expand_dim inside __call__.
        # So the dim tag itself should be part of the layer state, and we need to define the initial value of it here.
        # This is not really supported, in various ways, also including RETURNN.
        # We just keep this code in place to be prepared for that.
        # The reason it works right now is that we do an optimization where we replace zero init state by 0.
        expand_dim = nn.SpatialDim("self_att_expand_dim_init", 0)
        return nn.LayerState(
            k_accum=nn.LayerState(nn.zeros(list(batch_dims) + [expand_dim, self.num_heads, self.key_dim_per_head])),
            v_accum=nn.LayerState(nn.zeros(list(batch_dims) + [expand_dim, self.num_heads, self.value_dim_per_head])),
        )

    def forward_qkv(self, source: nn.Tensor) -> Tuple[nn.Tensor, nn.Tensor, nn.Tensor]:
        """
        :return: q,k,v
        """
        qkv = self.qkv(source)
        qkv = nn.split_dims(
            qkv, axis=self.qkv_dim_total, dims=(self.num_heads, self.qkv_dim_per_head), name="qkv_split_dims"
        )
        q, k, v = nn.split(
            qkv,
            axis=self.qkv_dim_per_head,
            out_dims=(self.key_dim_per_head, self.key_dim_per_head, self.value_dim_per_head),
            name="qkv_split",
        )
        return q, k, v

    def __call__(
        self, source: nn.Tensor, *, axis: nn.Dim, causal: Optional[bool] = None, state: Optional[nn.LayerState] = None
    ) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
        """forward"""
        q, k, v = self.forward_qkv(source)
        if axis == nn.single_step_dim:
            assert causal is None or causal  # always causal for single step
            assert state
            loop = nn.NameCtx.inner_loop()
            hist_dim = nn.SpatialDim(
                f"{loop.axis.description if loop else nn.NameCtx.current_ctx().get_abs_name()}:kv-history"
            )
            new_state = nn.LayerState()
            k, _, new_state.k_accum = nn.cum_concat_step(
                k, state=state.k_accum, out_spatial_dim=hist_dim, name="k_accum"
            )
            v, _, new_state.v_accum = nn.cum_concat_step(
                v, state=state.v_accum, out_spatial_dim=hist_dim, name="v_accum"
            )
        else:
            new_state = None
            assert not state
            if causal:
                raise NotImplementedError(
                    "Causal attention on sequence level not implemented. "
                    "We can easily extend CumConcatLayer on RETURNN side for this, to accept any axis argument. "
                    "However, normally any causal attention should always be inside a loop "
                    "and this should never be needed."
                )
            hist_dim = nn.SpatialDim(f"{axis.description}:{'kv-history' if causal else 'kv'}")
            k, _ = nn.replace_dim(k, in_dim=axis, out_dim=hist_dim, name="k_new_dim")
            v, _ = nn.replace_dim(v, in_dim=axis, out_dim=hist_dim, name="v_new_dim")
        att = dot_attention(q, k, v, key_dim=self.key_dim_per_head, axis=hist_dim, att_dropout=self.att_dropout)
        output, _ = nn.merge_dims(
            att, axes=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total, name="output"
        )
        if self.proj:
            output = self.proj(output)
        return output, new_state


class SelfAttention(GenericSelfAttention):
    """
    Classic self attention on sequence level
    """

    def __call__(self, source: nn.Tensor, *, axis: nn.Dim, **_kwargs) -> nn.Tensor:
        """forward"""
        assert not _kwargs
        out, _ = super().__call__(source, axis=axis, causal=False)
        return out


class CausalSelfAttention(GenericSelfAttention):
    """
    Classic causal self attention
    """

    def __call__(
        self, source: nn.Tensor, *, axis: nn.Dim, state: Optional[nn.LayerState] = None, **_kwargs
    ) -> Tuple[nn.Tensor, nn.LayerState]:
        """forward"""
        assert not _kwargs
        out, state = super().__call__(source, causal=True, axis=axis, state=state)
        return out, state


class RelPosSelfAttention(GenericSelfAttention):
    """
    Self-attention with relative positional encoding.
    This covers both Shawn et al. self-att rel pos 2018 (https://arxiv.org/abs/1803.02155),
    and Dai et al. Transformer-XL style 2019 (https://arxiv.org/abs/1901.02860).

    It uses :func:`relative_positional_encoding` or :class:`LearnedRelativePositionalEncoding`.

    To get Shawn et al. self-att rel pos 2018 / RETURNN SelfAttentionLayer + RelativePositionalEncodingLayer:
    - with_bias = False (at least that was the RETURNN behavior)
    - with_linear_pos = False
    - with_pos_bias = False
    - learnable_pos_emb = True
    - separate_pos_emb_per_head = False (at least that was the RETURNN default)

    To get Dai et al. Transformer-XL style 2019:
    - with_bias = False would be like the paper, however, in most implementations it is True (default)
    - with_linear_pos = True (default)
    - with_pos_bias = True (default)
    - learnable_pos_emb = True (default)
    - separate_pos_emb_per_head = True (default)

    Further details:
    https://github.com/rwth-i6/returnn_common/wiki/Relative-positional-encoding

    Code references, partly adapted from there:
    https://github.com/espnet/espnet/blob/4138010fb66ad27a43e8bee48a4932829a0847ae/espnet/nets/pytorch_backend/transformer/embedding.py#L260
    https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/tf/model.py#L4
    """

    def __init__(
        self,
        in_dim: nn.Dim,
        proj_dim: Optional[nn.Dim],
        *,
        key_dim_total: nn.Dim,
        value_dim_total: nn.Dim,
        num_heads: Union[int, nn.Dim],
        with_bias: bool = True,
        with_linear_pos: bool = True,
        with_pos_bias: bool = True,
        learnable_pos_emb: bool = False,
        learnable_pos_emb_clipping: int = 16,
        separate_pos_emb_per_head: bool = True,
        pos_emb_dropout: float = 0.0,
        att_dropout: float = 0.1,
    ):
        super(RelPosSelfAttention, self).__init__(
            in_dim=in_dim,
            proj_dim=proj_dim,
            key_dim_total=key_dim_total,
            value_dim_total=value_dim_total,
            num_heads=num_heads,
            with_bias=with_bias,
            att_dropout=att_dropout,
        )
        self.separate_pos_emb_per_head = separate_pos_emb_per_head
        if with_linear_pos:
            self.pos_emb_feat_dim = self.in_dim
        elif separate_pos_emb_per_head:
            self.pos_emb_feat_dim = self.key_dim_total
        else:
            self.pos_emb_feat_dim = self.key_dim_per_head
        # linear transformation for positional encoding
        self.linear_pos = None
        if with_linear_pos:
            self.linear_pos = nn.Linear(
                self.in_dim, self.key_dim_total if separate_pos_emb_per_head else self.key_dim_per_head, with_bias=False
            )
        self.learned_pos_emb = None
        if learnable_pos_emb:
            self.learned_pos_emb = LearnedRelativePositionalEncoding(
                self.pos_emb_feat_dim, clipping=learnable_pos_emb_clipping
            )
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = None
        self.pos_bias_v = None
        if with_pos_bias:
            self.pos_bias_u = nn.Parameter((self.num_heads, self.key_dim_per_head))
            self.pos_bias_v = nn.Parameter((self.num_heads, self.key_dim_per_head))
            self.pos_bias_u.initial = nn.init.Glorot()
            self.pos_bias_v.initial = nn.init.Glorot()
        self.pos_emb_dropout = pos_emb_dropout

    def __call__(self, source: nn.Tensor, *, axis: nn.Dim, **_kwargs) -> nn.Tensor:
        """forward"""
        if self.learned_pos_emb is not None:
            pos_emb, pos_emb_spatial_dim = self.learned_pos_emb(axis)
        else:
            pos_emb, pos_emb_spatial_dim = relative_positional_encoding(axis, self.pos_emb_feat_dim)
        if self.pos_emb_dropout:
            pos_emb = nn.dropout(pos_emb, self.pos_emb_dropout, axis=pos_emb.dims)
        if self.linear_pos is not None:
            pos_emb = self.linear_pos(pos_emb)
        if self.separate_pos_emb_per_head:
            pos_emb = nn.split_dims(pos_emb, axis=self.key_dim_total, dims=(self.num_heads, self.key_dim_per_head))
        # pos_emb: (head, 2*time1-1, d_k)

        q, k, v = self.forward_qkv(source)
        hist_dim = nn.SpatialDim(f"{axis.description}:kv")
        k, _ = nn.replace_dim(k, in_dim=axis, out_dim=hist_dim, name="k_new_dim")
        v, _ = nn.replace_dim(v, in_dim=axis, out_dim=hist_dim, name="v_new_dim")
        q_with_bias_u = (q + self.pos_bias_u) if self.pos_bias_u is not None else q  # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v) if self.pos_bias_v is not None else q  # (batch, head, time1, d_k)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = nn.dot(q_with_bias_u, k, reduce=self.key_dim_per_head)

        # compute matrix b and matrix d
        # (batch, head, time1, 2*time1-1)
        matrix_bd = nn.dot(q_with_bias_v, pos_emb, reduce=self.key_dim_per_head)
        matrix_bd = self._rel_shift(matrix_bd, axis, pos_emb_spatial_dim, hist_dim)

        scores = matrix_ac + matrix_bd  # (batch, head, time1, time2)
        scores *= self.key_dim_per_head.dimension**-0.5
        att_weights = nn.softmax(scores, axis=hist_dim, name="att_weights")
        att_weights = nn.dropout(att_weights, self.att_dropout, axis=hist_dim)
        att = nn.dot(att_weights, v, reduce=hist_dim, name="att")
        output, _ = nn.merge_dims(
            att, axes=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total, name="output"
        )
        if self.proj:
            output = self.proj(output)
        return output

    @classmethod
    def _rel_shift(cls, x: nn.Tensor, axis: nn.Dim, pos_emb_spatial_dim: nn.Dim, hist_dim: nn.Dim) -> nn.Tensor:
        """
        :param x: [B,H,T,T*2-1]
        :param axis: T
        :param pos_emb_spatial_dim: T*2-1
        :param hist_dim: T' (equal to T but separate dim)
        :return: [B,H,T,T']
        """
        batch_dims = x.remaining_dims((axis, pos_emb_spatial_dim))
        x_padded = nn.pad(x, axes=pos_emb_spatial_dim, padding=(1, 0), value=0.0)  # [B,H,T,T*2]
        pos_emb_spatial_dim_ = 1 + pos_emb_spatial_dim

        # Reshape + slice trickery. You need to draw the 2D arrays on paper to understand this.
        # Also see similar trickery in :func:`window`.
        x_padded = nn.reshape(x_padded, (axis, pos_emb_spatial_dim_), (pos_emb_spatial_dim_, axis))  # [B,H,T*2,T]
        x_padded, pos_emb_spatial_dim_ = nn.slice(x_padded, axis=pos_emb_spatial_dim_, slice_start=1)  # [B,H,T*2-1,T]
        x_padded = nn.reshape(x_padded, (pos_emb_spatial_dim_, axis), (axis, pos_emb_spatial_dim_))  # [B,H,T,T*2-1]
        x_padded, _ = nn.slice_nd(x_padded, axis=pos_emb_spatial_dim_, size=hist_dim)  # [B,H,T,T']
        x_padded.verify_out_shape(set(batch_dims) | {axis, hist_dim})
        return x_padded


_relative_positional_encoding_cache = weakref.WeakKeyDictionary()  # root name ctx -> (spatial_dim, feat_dim) -> enc


class LearnedRelativePositionalEncoding(nn.Module):
    """
    Learnable relative positional encoding.

    E.g. as used in Shawn et al, 2018 (https://arxiv.org/abs/1803.02155).

    https://github.com/rwth-i6/returnn_common/wiki/Relative-positional-encoding
    """

    def __init__(self, feat_dim: nn.Dim, *, clipping: int = 16, dtype: str = "float32"):
        """
        :param feat_dim: feature dim, for the emb matrix and output
        :param clipping: max distance to consider. emb matrix shape is [2 * clipping + 1, feat_dim].
          The first and last frame will be the clipping frames.
        :param dtype: for the emb matrix and output
        """
        super(LearnedRelativePositionalEncoding, self).__init__()
        self.feat_dim = feat_dim
        self.clipping = clipping
        self.clipped_spatial_dim = nn.SpatialDim(
            f"{nn.NameCtx.current_ctx().get_abs_name()}:learned-rel-pos", dimension=2 * clipping + 1
        )
        self.pos_emb = nn.Parameter((self.clipped_spatial_dim, self.feat_dim), dtype=dtype)

    def __call__(self, spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
        """
        same interface as :func:`relative_positional_encoding`

        :return: tensor of shape [spatial_dim * 2 - 1, feat_dim], and the out spatial dim (spatial_dim * 2 - 1).
          In the center is the rel pos i-j=0. All to the right are for i-j>0, all to the left for i-j<0.
        """
        out_spatial_dim = spatial_dim - 1 + spatial_dim
        mat_spatial_size = self.clipping + 1
        with nn.Cond(nn.dim_value(spatial_dim) > mat_spatial_size) as cond:
            # True branch
            left = nn.gather(self.pos_emb, axis=self.clipped_spatial_dim, position=0)
            right = nn.gather(
                self.pos_emb, axis=self.clipped_spatial_dim, position=self.clipped_spatial_dim.dimension - 1
            )
            remaining_dim = spatial_dim - mat_spatial_size
            left = nn.expand_dim(left, dim=remaining_dim)
            right = nn.expand_dim(right, dim=remaining_dim)
            concat, out_spatial_dim_ = nn.concat(
                (left, remaining_dim), (self.pos_emb, self.clipped_spatial_dim), (right, remaining_dim)
            )
            concat, out_spatial_dim_ = nn.replace_dim(concat, in_dim=out_spatial_dim_, out_dim=out_spatial_dim)
            cond.true = concat

            # False branch, spatial_dim <= self.clipping
            cond.false, _ = nn.slice_nd(
                self.pos_emb,
                axis=self.clipped_spatial_dim,
                start=mat_spatial_size - nn.dim_value(spatial_dim),
                size=out_spatial_dim,
            )

        return cond.result, out_spatial_dim


def relative_positional_encoding(
    spatial_dim: nn.Dim, feat_dim: nn.Dim, *, dtype: str = "float32"
) -> Tuple[nn.Tensor, nn.Dim]:
    """
    Implements relative positional encoding, Transformer-XL style (https://arxiv.org/abs/1901.02860),
    as used for example by :class:`RelPosSelfAttention`.

    Code references, partly adapted from there:
    https://github.com/espnet/espnet/blob/4138010fb66ad27a43e8bee48a4932829a0847ae/espnet/nets/pytorch_backend/transformer/embedding.py#L260
    https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/tf/model.py#L4

    Note that this encoding is stored in a cache so that it is only calculated once.
    and then reused.

    Note that we could extend the implementation later to also buffer it
    even across mini-batches, like the ESPnet implementation does,
    e.g. by storing it in an auxiliary variable and increasing its size when needed.
    But this is not done yet, to keep the code simple.

    :return: tensor of shape [spatial_dim * 2 - 1, feat_dim], and the out spatial dim (spatial_dim * 2 - 1).
      In the center is the rel pos i-j=0. All to the right are for i-j>0, all to the left for i-j<0.
    """
    root_name_ctx = nn.NameCtx.top().root
    cache = _relative_positional_encoding_cache.setdefault(root_name_ctx, {})
    if (spatial_dim, feat_dim) in cache:
        return cache[(spatial_dim, feat_dim)]
    import math

    with nn.control_flow_ctx(None):
        position_pos = nn.range_over_dim(spatial_dim, dtype=dtype)
        position_neg = -nn.dim_value(spatial_dim) + nn.range_over_dim(spatial_dim - 1) + 1
        position_neg = nn.cast(position_neg, dtype=dtype)
        position, out_spatial_dim = nn.concat((position_neg, spatial_dim - 1), (position_pos, spatial_dim))
        feat2_dim = feat_dim.div_left(2)
        div_term = nn.exp(nn.range_over_dim(feat2_dim, dtype=dtype) * -(2.0 * math.log(10000.0) / feat_dim.dimension))
        arg_sin = nn.combine_bc(position, "*", div_term)
        arg_cos = arg_sin + math.pi / 2.0
        arg, feat_dim_ = nn.concat((arg_sin, feat2_dim), (arg_cos, feat2_dim))
        arg, feat_dim_ = nn.replace_dim(arg, in_dim=feat_dim_, out_dim=feat_dim)
        emb = nn.sin(arg)
        emb.verify_out_shape({out_spatial_dim, feat_dim})
        cache[(spatial_dim, feat_dim)] = emb, out_spatial_dim
        return emb, out_spatial_dim
