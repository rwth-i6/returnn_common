"""
Multi layer BLSTm
"""

from typing import Union, Tuple
from ... import nn
from .base import ISeqDownsamplingEncoder


class BlstmEncoder(ISeqDownsamplingEncoder):
    """
    multi-layer BLSTM
    """

    def __init__(
        self,
        in_dim: nn.Dim,
        dim: nn.Dim = nn.FeatureDim("lstm-dim", 1024),
        *,
        num_layers: int = 6,
        time_reduction: Union[int, Tuple[int, ...]] = 6,
        allow_pool_last: bool = False,
        l2=0.0001,
        dropout=0.3,
        rec_weight_dropout=0.0,
    ):
        super(BlstmEncoder, self).__init__()
        self.in_dim = in_dim
        self.dim = dim
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
        if num_layers == 1 and not allow_pool_last:
            assert not time_reduction, f"time_reduction {time_reduction} not supported for single layer"
        while len(time_reduction) > (num_layers if allow_pool_last else (num_layers - 1)):
            time_reduction[:2] = [time_reduction[0] * time_reduction[1]]
        self.time_reduction = time_reduction

        self.dropout = dropout
        self.rec_weight_dropout = rec_weight_dropout

        in_dims = [in_dim] + [2 * dim] * (num_layers - 1)
        self.layers = nn.ModuleList([BlstmSingleLayer(in_dims[i], dim) for i in range(num_layers)])
        self.out_dim = self.layers[-1].out_dim

        if l2:
            for param in self.parameters():
                param.weight_decay = l2

        if rec_weight_dropout:
            raise NotImplementedError  # TODO ...

    def __call__(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
        feat_dim = self.in_dim
        for i, lstm in enumerate(self.layers):
            if i > 0:
                if self.dropout:
                    source = nn.dropout(source, dropout=self.dropout, axis=feat_dim)
            assert isinstance(lstm, BlstmSingleLayer)
            source = lstm(source, spatial_dim=in_spatial_dim)
            feat_dim = lstm.out_dim
            red = self.time_reduction[i] if i < len(self.time_reduction) else 1
            if red > 1:
                source, in_spatial_dim = nn.pool1d(
                    source, mode="max", padding="same", pool_size=red, in_spatial_dim=in_spatial_dim
                )
        return source, in_spatial_dim


class BlstmSingleLayer(nn.Module):
    """
    single-layer BLSTM
    """

    def __init__(self, in_dim: nn.Dim, out_dim: nn.Dim):
        super(BlstmSingleLayer, self).__init__()
        self.fw = nn.LSTM(in_dim, out_dim)
        self.bw = nn.LSTM(in_dim, out_dim)
        self.out_dim = 2 * out_dim

    def __call__(self, x: nn.Tensor, *, spatial_dim: nn.Dim) -> nn.Tensor:
        fw, _ = self.fw(x, spatial_dim=spatial_dim, direction=1)
        bw, _ = self.bw(x, spatial_dim=spatial_dim, direction=-1)
        return nn.concat_features(fw, bw)
