"""
Test nn.array_.
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict
import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_concat():
  class _Net(nn.Module):

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      return nn.concat_features(x, x * 2.)

  config, net_dict = dummy_config_net_dict(net=_Net())
  dummy_run_net(config)
