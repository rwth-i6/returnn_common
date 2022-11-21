"""
Test nn.array_.
"""

from __future__ import annotations
from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net_single_custom
import typing
import numpy
import numpy.testing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_concat():
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("in", 3)
  x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)
  x = nn.get_extern_data(x)
  out, dim = nn.concat((x, x.feature_dim), (x * 2., x.feature_dim))
  assert dim == 2 * x.feature_dim
  out.mark_as_default_output()
  out.mark_as_loss("y")
  config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  dummy_run_net_single_custom(config_str, eval_flag=True)


def test_concat_features():
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("in", 3)
  x = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)
  x = nn.get_extern_data(x)
  out = nn.concat_features(x, x * 2.)
  out.mark_as_default_output()
  out.mark_as_loss("y")
  config_str = nn.get_returnn_config().get_complete_py_code_str(nn.Module())
  dummy_run_net_single_custom(config_str, eval_flag=True)


def ceildiv(a, b):
  """ceildiv"""
  return -(-a // b)


def naive_window(source: numpy.ndarray, window: int, stride: int = 1, padding='same'):
  """
  naive implementation of tf_util.windowed

  :param source: (time,...)
  :param window: window size
  :param stride: stride
  :param padding: 'same' or 'valid'
  :return: (time,window,...)
  """
  assert source.ndim >= 1
  if padding == 'same':
    n_time = source.shape[0]
    w_right = window // 2
    w_left = window - w_right - 1
  elif padding == 'valid':
    n_time = source.shape[0] - window + 1
    w_right = 0
    w_left = 0
  else:
    raise Exception("invalid padding %r" % padding)

  dtype = source.dtype
  pad_left = numpy.zeros((w_left,) + source.shape[1:], dtype=dtype)
  pad_right = numpy.zeros((w_right,) + source.shape[1:], dtype=dtype)
  padded = numpy.concatenate([pad_left, source, pad_right], axis=0)
  final = numpy.zeros((ceildiv(n_time, stride), window) + source.shape[1:], dtype=dtype)
  for t in range(final.shape[0]):
    for w in range(final.shape[1]):
      final[t, w] = padded[t * stride + w]
  return final


def test_window():
  for win_size in (3, 2):
    nn.reset_default_root_name_ctx()
    time_dim = nn.SpatialDim("time")
    in_dim = nn.FeatureDim("in", 3)
    x = nn.Data("data", dim_tags=[time_dim, nn.batch_dim, in_dim], available_for_inference=True)
    x = nn.get_extern_data(x)
    win_dim = nn.SpatialDim(f"window{win_size}", win_size)
    out, _ = nn.window(x, spatial_dim=time_dim, window_dim=win_dim)
    out.mark_as_default_output()
    config_str = nn.get_returnn_config().get_config_raw_dict(nn.Module())
    res = dummy_run_net_single_custom(config_str, default_out_dim_tag_order=[time_dim, win_dim, nn.batch_dim, in_dim])
    numpy.testing.assert_array_equal(res["layer:output"], naive_window(res["data:data"], win_dim.dimension))


def test_window_stride():
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("in", 3)
  x = nn.Data("data", dim_tags=[time_dim, nn.batch_dim, in_dim], available_for_inference=True)
  x = nn.get_extern_data(x)
  win_dim = nn.SpatialDim("window", 5)
  out, time_dim_ = nn.window(x, spatial_dim=time_dim, window_dim=win_dim, stride=3)
  out.mark_as_default_output()
  config_str = nn.get_returnn_config().get_config_raw_dict(nn.Module())
  res = dummy_run_net_single_custom(config_str, default_out_dim_tag_order=[time_dim_, win_dim, nn.batch_dim, in_dim])
  numpy.testing.assert_array_equal(res["layer:output"], naive_window(res["data:data"], win_dim.dimension, 3))


def test_window_via_get_network():
  from returnn.config import Config
  from returnn.tf.engine import Engine
  from returnn.datasets import init_dataset

  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("in", 3)
  out_dim = nn.FeatureDim("out", 4)
  data = nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim], available_for_inference=True)

  def _config_get_network(epoch: int, **_kwargs) -> dict:
    print("_config_get_network called")  # it's called multiple times
    # noinspection PyStatementEffect
    epoch  # unused
    nn.reset_default_root_name_ctx()
    x = nn.get_extern_data(data)
    time_dim_ = time_dim
    win_dim = nn.SpatialDim("window", 3)
    out, _ = nn.window(x, spatial_dim=time_dim_, window_dim=win_dim)
    out.verify_out_shape({nn.batch_dim, time_dim_, win_dim, in_dim})
    net = nn.Linear(in_dim, out_dim)  # just to have some params to optimize, for the test
    y = net(x)
    x, _ = nn.window(x, spatial_dim=time_dim_, window_dim=win_dim)
    x = nn.reduce(out, mode="mean", axis=(win_dim, in_dim))
    x.verify_out_shape({nn.batch_dim, time_dim_})
    y = y + x
    y.mark_as_output()
    y = nn.reduce(y, mode="mean", axis=out_dim)
    y.verify_out_shape({nn.batch_dim, time_dim_})
    y.mark_as_loss("dummy")
    print(nn.get_returnn_config().get_complete_py_code_str(net))
    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(net)
    net_dict["#epoch"] = epoch  # trigger reinit
    return net_dict

  config = Config({
    "task": "train", "num_epochs": 2, "start_epoch": 1,
    "get_network": _config_get_network,
    "extern_data": {data.name: {"dim_tags": data.dim_tags, "available_for_inference": True}},
  })
  train_dataset = init_dataset(
    {"class": "DummyDataset", "input_dim": in_dim.dimension, "output_dim": 5, "num_seqs": 3})
  engine = Engine(config)
  engine.init_train_from_config(config, train_data=train_dataset)
  engine.train()
