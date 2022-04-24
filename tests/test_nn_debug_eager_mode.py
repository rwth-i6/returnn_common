"""
debug eager mode test
"""

from __future__ import annotations

from . import _setup_test_env  # noqa
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


# Enables it globally now.
nn.NameCtx.current_ctx().root.enable_debug_eager_mode()


def test_simple_linear():
  """nn.Linear"""
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, nn.SpatialDim("time"), nn.FeatureDim("in", 5)]))
  assert data.data.placeholder is not None
  lin = nn.Linear(nn.FeatureDim("lin", 10))
  out = lin(data)
  assert lin.weight.data.placeholder is not None
  assert lin.bias.data.placeholder is not None
  assert out.data.placeholder is not None
  assert out.data.placeholder.numpy().size > 0
  assert (out.data.placeholder.numpy() != 0).any()


def test_conformer():
  """nn.Conformer"""
  time_dim = nn.SpatialDim("time")
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, nn.FeatureDim("in", 5)]))
  conformer = nn.ConformerEncoder(nn.FeatureDim("conformer", 10), num_layers=2, num_heads=2)
  out, out_spatial_dim = conformer(data, in_spatial_dim=time_dim)
  assert out.data.placeholder is not None
