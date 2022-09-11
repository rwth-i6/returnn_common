"""
Multi layer BLSTm
"""

from typing import Union, Tuple
from ... import nn


class BlstmEncoder(nn.Module):
  """
  multi-layer BLSTM
  """
  def __init__(self,
               num_layers: int = 6, dim: nn.Dim = nn.FeatureDim("lstm-dim", 1024),
               time_reduction: Union[int, Tuple[int, ...]] = 6,
               l2=0.0001, dropout=0.3, rec_weight_dropout=0.0,
               ):
    super(BlstmEncoder, self).__init__()
    self.num_layers = num_layers

    if isinstance(time_reduction, int):
      n = time_reduction
      time_reduction = []
      for i in range(2, n + 1):
        while n % i == 0:
          time_reduction.insert(0, i)
          n //= i
        if n <= 1:
          break
    assert isinstance(time_reduction, (tuple, list))
    assert num_layers > 0
    if num_layers == 1:
      assert not time_reduction, f"time_reduction {time_reduction} not supported for single layer"
    while len(time_reduction) > num_layers - 1:
      time_reduction[:2] = [time_reduction[0] * time_reduction[1]]
    self.time_reduction = time_reduction

    self.dropout = dropout
    self.rec_weight_dropout = rec_weight_dropout

    self.layers = nn.ModuleList([BlstmSingleLayer(dim=dim) for _ in range(num_layers)])

    if l2:
      for param in self.parameters():
        param.weight_decay = l2

    if rec_weight_dropout:
      raise NotImplementedError  # TODO ...

  def __call__(self, x: nn.Tensor, *, spatial_dim: nn.Dim) -> (nn.Tensor, nn.Dim):
    for i, lstm in enumerate(self.layers):
      if i > 0:
        red = self.time_reduction[i - 1] if (i - 1) < len(self.time_reduction) else 1
        if red > 1:
          x, spatial_dim = nn.pool1d(x, mode="max", padding="same", pool_size=red, in_spatial_dim=spatial_dim)
        if self.dropout:
          x = nn.dropout(x, dropout=self.dropout, axis=x.feature_dim)
      assert isinstance(lstm, BlstmSingleLayer)
      x = lstm(x, axis=spatial_dim)
    return x, spatial_dim


class BlstmSingleLayer(nn.Module):
  """
  single-layer BLSTM
  """
  def __init__(self, dim: nn.Dim):
    super(BlstmSingleLayer, self).__init__()
    self.fw = nn.LSTM(out_dim=dim)
    self.bw = nn.LSTM(out_dim=dim)

  def __call__(self, x: nn.Tensor, *, axis: nn.Dim) -> nn.Tensor:
    fw, _ = self.fw(x, axis=axis, direction=1)
    bw, _ = self.bw(x, axis=axis, direction=-1)
    return nn.concat_features(fw, bw)
