"""
Attention, self-attention, auto-regressive self-attention
"""

from typing import Tuple, Union, Optional
from .. import nn


@nn.scoped
def dot_attention(query: nn.LayerRef, keys: nn.LayerRef, values: nn.LayerRef,
                  key_dim: nn.Dim, axis: nn.Dim, att_dropout: float = 0.) -> nn.LayerRef:
  """
  Calculates attention over the given axis, for given key dim.
  Any other unrelated axes do not matter here.
  This can be used for multi-head or single head.
  The query can have other dimensions or not.
  """
  query *= key_dim.dimension ** -0.5
  energy = nn.dot(query, keys, reduce=key_dim, name="energy")
  att_weights = nn.softmax(energy, axis=axis, name="att_weights")
  att_weights = nn.dropout(att_weights, att_dropout, axis=axis)
  att = nn.dot(att_weights, values, reduce=axis, name="att")
  return att


# noinspection PyAbstractClass
class GenericSelfAttention(nn.Module):
  """
  Shared base class for self attention
  """
  def __init__(self, *, key_dim_total: nn.Dim, value_dim_total: nn.Dim, num_heads: Union[int, nn.Dim],
               att_dropout: float = 0.):
    super().__init__()
    if isinstance(num_heads, int):
      num_heads = nn.SpatialDim("num_heads", num_heads)
    self.key_dim_total = key_dim_total
    self.key_dim_per_head = key_dim_total.div_left(num_heads)
    self.value_dim_total = value_dim_total
    self.value_dim_per_head = value_dim_total.div_left(num_heads)
    self.num_heads = num_heads
    self.qkv_dim_total = 2 * key_dim_total + value_dim_total
    self.qkv_dim_per_head = 2 * self.key_dim_per_head + self.value_dim_per_head
    self.qkv = nn.Linear(self.qkv_dim_total)
    self.att_dropout = att_dropout

  def default_initial_state(self) -> nn.LayerState:
    """
    For causal attention.
    """
    pass  # TODO ...

  @nn.scoped
  def __call__(self, source: nn.LayerRef, *, axis: nn.Dim,
               causal: Optional[bool] = None, state: Optional[nn.LayerState] = None
               ) -> Tuple[nn.Layer, Optional[nn.LayerState]]:
    """forward"""
    expand_dim = nn.SpatialDim("self_att_expand_dim")
    qkv = self.qkv(source)
    qkv = nn.split_dims(
      qkv, axis=self.qkv_dim_total, dims=(self.num_heads, self.qkv_dim_per_head), name="qkv_split_dims")
    q, k, v = nn.split(
      qkv, axis=self.qkv_dim_per_head,
      out_dims=(self.key_dim_per_head, self.key_dim_per_head, self.value_dim_per_head),
      name="qkv_split")
    if axis == nn.single_step_dim:
      assert causal is None or causal
      assert state
      new_state = nn.LayerState()
      k, new_state.k_accum = nn.cum_concat_step(k, state=state.k_accum, out_spatial_dim=expand_dim)
      v, new_state.v_accum = nn.cum_concat_step(v, state=state.v_accum, out_spatial_dim=expand_dim)
    else:
      new_state = None
      if causal:
        raise NotImplementedError(
          "Causal attention on sequence level not implemented. "
          "We can easily extend CumConcatLayer on RETURNN side for this, to accept any axis argument. "
          "However, normally any causal attention should always be inside a loop and this should never be needed.")
      k = nn.reinterpret_data(k, set_dim_tags={axis: expand_dim}, name="k_new_dim")
      v = nn.reinterpret_data(v, set_dim_tags={axis: expand_dim}, name="v_new_dim")
    att = dot_attention(q, k, v, key_dim=self.key_dim_per_head, axis=expand_dim, att_dropout=self.att_dropout)
    output = nn.merge_dims(
      att, axes=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total, name="output")
    return output, new_state


class SelfAttention(GenericSelfAttention):
  """
  Classic self attention on sequence level
  """
  @nn.scoped
  def __call__(self, source: nn.LayerRef, *, axis: nn.Dim) -> nn.Layer:
    """forward"""
    out, _ = super().__call__(source, axis=axis, causal=False)
    return out


class CausalSelfAttention(GenericSelfAttention):
  """
  Classic causal self attention
  """
  @nn.scoped
  def __call__(self, source: nn.LayerRef, *, axis: nn.Dim) -> nn.Layer:
    """forward"""
    out, _ = super().__call__(source, axis=axis, causal=True)
    return out
