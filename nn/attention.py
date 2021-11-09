"""
Attention, self-attention, auto-regressive self-attention
"""

from typing import Tuple
from .. import nn


class SelfAttention(nn.Module):
  """
  Classic self attention
  """
  def __init__(self, *, axis: nn.DimensionTag, key_dim_total, value_dim_total, num_heads: int, att_dropout: float):
    super().__init__()
    self.axis = axis
    self.key_dim_total = key_dim_total
    self.key_dim_per_head = key_dim_total // num_heads
    self.value_dim_total = value_dim_total
    self.value_dim_per_head = value_dim_total // num_heads
    self.num_heads = num_heads
    self.qkv = nn.Linear(key_dim_total * 2 + value_dim_total)
    self.expand_dim = nn.DimensionTag(kind=nn.DimensionTag.Types.Spatial, description="self_att_expand_dim")
    self.att_dropout = att_dropout

  def forward(self, source: nn.LayerRef) -> nn.Layer:
    """forward"""
    # noinspection DuplicatedCode
    qkv = self.qkv(source)
    qkv = nn.split_dims(
      qkv, axis="F", dims=(self.num_heads, self.key_dim_per_head * 2 + self.value_dim_per_head),
      name="qkv_split_dims")
    q, k, v = nn.split(
      qkv, axis="F", size_splits=(self.key_dim_per_head, self.key_dim_per_head, self.value_dim_per_head),
      name="qkv_split")
    q *= self.key_dim_per_head ** -0.5
    k = nn.reinterpret_data(k, set_dim_tags={self.axis: self.expand_dim}, name="k_new_dim")
    v = nn.reinterpret_data(v, set_dim_tags={self.axis: self.expand_dim}, name="v_new_dim")
    energy = nn.dot([q, k], red1="static:-1", red2="static:-1", var1=self.axis, var2=self.expand_dim, name="energy")
    att_weights = nn.softmax(energy, axis=self.expand_dim, name="att_weights")
    att_weights = nn.dropout(att_weights, self.att_dropout)
    att = nn.dot(
      [att_weights, v], red1=self.expand_dim, red2=self.expand_dim, var1=self.axis, var2="static:-1", name="att")
    output = nn.merge_dims(att, axes="static", name="output")
    return output


class SelfAttentionStep(nn.Module):
  """
  Auto-regressive self-attention
  """
  def __init__(self, *, key_dim_total, value_dim_total, num_heads: int, att_dropout: float):
    super().__init__()
    self.key_dim_total = key_dim_total
    self.key_dim_per_head = key_dim_total // num_heads
    self.value_dim_total = value_dim_total
    self.value_dim_per_head = value_dim_total // num_heads
    self.num_heads = num_heads
    self.qkv = nn.Linear(key_dim_total * 2 + value_dim_total)
    self.expand_dim = nn.DimensionTag(kind=nn.DimensionTag.Types.Spatial, description="self_att_expand_dim")
    self.att_dropout = att_dropout

  def forward(self, source: nn.LayerRef, *, state: nn.LayerState) -> Tuple[nn.Layer, nn.LayerState]:
    """forward"""
    new_state = nn.LayerState()
    # noinspection DuplicatedCode
    qkv = self.qkv(source)
    qkv = nn.split_dims(
      qkv, axis="F", dims=(self.num_heads, self.key_dim_per_head * 2 + self.value_dim_per_head),
      name="qkv_split_dims")
    q, k, v = nn.split(
      qkv, axis="F", size_splits=(self.key_dim_per_head, self.key_dim_per_head, self.value_dim_per_head),
      name="qkv_split")
    q *= self.key_dim_per_head ** -0.5
    k_accum, new_state.k_accum = nn.cum_concat_step(k, state=state.k_accum, new_dim=self.expand_dim)
    v_accum, new_state.v_accum = nn.cum_concat_step(v, state=state.v_accum, new_dim=self.expand_dim)
    energy = nn.dot(
      [q, k_accum], red1="static:-1", red2="static:-1", var1=None, var2=self.expand_dim, name="energy")
    att_weights = nn.softmax(energy, axis=self.expand_dim, name="att_weights")
    att_weights = nn.dropout(att_weights, self.att_dropout)
    att = nn.dot(
      [att_weights, v_accum],
      red1=self.expand_dim, red2=self.expand_dim, var1=None, var2="static:-1", name="att")
    output = nn.merge_dims(att, axes="static", name="output")
    return output, new_state
