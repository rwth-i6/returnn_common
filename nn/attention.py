"""
Attention, self-attention, auto-regressive self-attention
"""

from typing import Tuple, Union, Optional, Sequence
from .. import nn
from ..py_compat import Protocol


class AttentionFunc(Protocol):
  """Protocol defining a generic attention function"""
  def __call__(self,
               query: nn.Tensor, keys: nn.Tensor, values: nn.Tensor, *,
               key_dim: nn.Dim, axis: nn.Dim, att_dropout: float = 0.1): ...


def dot_attention(query: nn.Tensor, keys: nn.Tensor, values: nn.Tensor, *,
                  key_dim: nn.Dim, axis: nn.Dim, att_dropout: float = 0.1) -> nn.Tensor:
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
               att_dropout: float = 0.1):
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
      v_accum=nn.LayerState(nn.zeros(list(batch_dims) + [expand_dim, self.num_heads, self.value_dim_per_head])))

  def __call__(self, source: nn.Tensor, *, axis: nn.Dim,
               causal: Optional[bool] = None, state: Optional[nn.LayerState] = None
               ) -> Tuple[nn.Tensor, Optional[nn.LayerState]]:
    """forward"""
    qkv = self.qkv(source)
    qkv = nn.split_dims(
      qkv, axis=self.qkv_dim_total, dims=(self.num_heads, self.qkv_dim_per_head), name="qkv_split_dims")
    q, k, v = nn.split(
      qkv, axis=self.qkv_dim_per_head,
      out_dims=(self.key_dim_per_head, self.key_dim_per_head, self.value_dim_per_head),
      name="qkv_split")
    if axis == nn.single_step_dim:
      assert causal is None or causal  # always causal for single step
      assert state
      loop = nn.NameCtx.inner_loop()
      hist_dim = nn.SpatialDim(
        f"{loop.axis.description if loop else nn.NameCtx.current_ctx().get_abs_name()}:kv-history")
      new_state = nn.LayerState()
      k, _, new_state.k_accum = nn.cum_concat_step(k, state=state.k_accum, out_spatial_dim=hist_dim, name="k_accum")
      v, _, new_state.v_accum = nn.cum_concat_step(v, state=state.v_accum, out_spatial_dim=hist_dim, name="v_accum")
    else:
      new_state = None
      assert not state
      if causal:
        raise NotImplementedError(
          "Causal attention on sequence level not implemented. "
          "We can easily extend CumConcatLayer on RETURNN side for this, to accept any axis argument. "
          "However, normally any causal attention should always be inside a loop and this should never be needed.")
      hist_dim = nn.SpatialDim(f"{axis.description}:{'kv-history' if causal else 'kv'}")
      k, _ = nn.reinterpret_new_dim(k, in_dim=axis, out_dim=hist_dim, name="k_new_dim")
      v, _ = nn.reinterpret_new_dim(v, in_dim=axis, out_dim=hist_dim, name="v_new_dim")
    att = dot_attention(q, k, v, key_dim=self.key_dim_per_head, axis=hist_dim, att_dropout=self.att_dropout)
    output, _ = nn.merge_dims(
      att, axes=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total, name="output")
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
  def __call__(self, source: nn.Tensor, *, axis: nn.Dim, state: Optional[nn.LayerState] = None, **_kwargs
               ) -> Tuple[nn.Tensor, nn.LayerState]:
    """forward"""
    assert not _kwargs
    out, state = super().__call__(source, causal=True, axis=axis, state=state)
    return out, state
