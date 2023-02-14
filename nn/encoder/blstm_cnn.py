"""
BLSTM with CNN
"""

from typing import Union, Tuple
from ... import nn
from .blstm import BlstmEncoder


class BlstmCnnEncoder(BlstmEncoder):
    """
    PreCNN . BLSTM
    """

    def __init__(
        self,
        in_dim: nn.Dim,
        lstm_dim: nn.Dim = nn.FeatureDim("lstm-dim", 1024),
        *,
        num_layers: int = 6,
        time_reduction: Union[int, Tuple[int, ...]] = 6,
        allow_pool_last: bool = False,
        l2=0.0001,
        dropout=0.3,
        rec_weight_dropout=0.0,
    ):
        self.pre_conv_net = PreConvNet(in_dim=in_dim)
        super(BlstmCnnEncoder, self).__init__(
            in_dim=self.pre_conv_net.out_dim,
            dim=lstm_dim,
            num_layers=num_layers,
            time_reduction=time_reduction,
            allow_pool_last=allow_pool_last,
            l2=l2,
            dropout=dropout,
            rec_weight_dropout=rec_weight_dropout,
        )

    def __call__(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim) -> Tuple[nn.Tensor, nn.Dim]:
        source = self.pre_conv_net(source, spatial_dim=in_spatial_dim)
        source, in_spatial_dim = super(BlstmCnnEncoder, self).__call__(source, in_spatial_dim=in_spatial_dim)
        return source, in_spatial_dim


class PreConvNet(nn.Module):
    """
    2 layer pre conv net, usually used before a BLSTM
    """

    def __init__(self, in_dim: nn.Dim, dim: nn.Dim = nn.FeatureDim("feat", 32), *, filter_size=(3, 3)):
        super(PreConvNet, self).__init__()
        self.in_dim = in_dim
        self._dummy_feat_dim = nn.FeatureDim("dummy-feature", 1)
        self.conv0 = nn.Conv2d(self._dummy_feat_dim, out_dim=dim, padding="same", filter_size=filter_size)
        self.conv1 = nn.Conv2d(dim, dim, padding="same", filter_size=filter_size)
        self._final_extra_spatial_dim = in_dim.ceildiv_right(2).ceildiv_right(2)
        self.out_dim = self._final_extra_spatial_dim * dim

    def __call__(self, x: nn.Tensor, *, spatial_dim: nn.Dim) -> nn.Tensor:
        assert self.in_dim in x.shape
        batch_dims = x.batch_dims_ordered((self.in_dim, spatial_dim))
        extra_spatial_dim = self.in_dim
        x = nn.expand_dim(x, dim=self._dummy_feat_dim)
        x, _ = self.conv0(x, in_spatial_dims=(spatial_dim, extra_spatial_dim))
        feat_dim = x.feature_dim
        x, extra_spatial_dim = nn.pool1d(x, in_spatial_dim=extra_spatial_dim, pool_size=2, mode="max", padding="same")
        x, _ = self.conv1(x, in_spatial_dims=(spatial_dim, extra_spatial_dim))
        x, extra_spatial_dim = nn.pool1d(x, in_spatial_dim=extra_spatial_dim, pool_size=2, mode="max", padding="same")
        x, extra_spatial_dim = nn.replace_dim(x, in_dim=extra_spatial_dim, out_dim=self._final_extra_spatial_dim)
        x, _ = nn.merge_dims(x, axes=(extra_spatial_dim, feat_dim), out_dim=nn.FeatureDim("conv-net-feature", None))
        x.verify_out_shape(set(batch_dims) | {self.out_dim, spatial_dim})
        return x
