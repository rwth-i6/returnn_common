"""
Helpers for RETURNN
"""

from __future__ import annotations
import typing
from typing import Dict, Tuple, Sequence, Union, Any, Optional, Callable
import contextlib
import numpy
import tensorflow as tf
import returnn.tf.engine
import returnn.datasets
from .. import nn


@contextlib.contextmanager
def make_scope():
  """
  :rtype: tf.compat.v1.Session
  """
  from returnn.tf import compat as tf_compat
  with tf.Graph().as_default() as graph:
    with tf_compat.v1.Session(graph=graph) as session:
      yield session


def make_feed_dict(data_list, n_batch=3, n_time=7, same_time: bool = False):
  """
  :param returnn.tf.network.ExternData data_list:
  :param int n_batch:
  :param int n_time:
  :param bool same_time:
  :rtype: dict[tf.Tensor,numpy.ndarray|list[int|float|bool]|int|float|bool]
  """
  from returnn.tf.network import ExternData
  if isinstance(data_list, ExternData):
    data_list = [value for (key, value) in sorted(data_list.data.items())]
  assert n_time > 0 and n_batch > 0
  rnd = numpy.random.RandomState(42)
  existing_sizes = {}  # type: typing.Dict[tf.Tensor,int]
  d = {}
  for data in data_list:
    shape = list(data.batch_shape)
    if data.batch_dim_axis is not None:
      shape[data.batch_dim_axis] = n_batch
    for axis, dim in enumerate(shape):
      if dim is None:
        axis_wo_b = data.get_batch_axis_excluding_batch(axis)
        assert axis_wo_b in data.size_placeholder
        dyn_size = data.size_placeholder[axis_wo_b]
        if dyn_size in existing_sizes:
          shape[axis] = existing_sizes[dyn_size]
          continue
        existing_sizes[dyn_size] = n_time
        shape[axis] = n_time
        dyn_size_v = numpy.array([n_time, max(n_time - 2, 1), max(n_time - 3, 1)])
        if dyn_size_v.shape[0] > n_batch:
          dyn_size_v = dyn_size_v[:n_batch]
        elif dyn_size_v.shape[0] < n_batch:
          dyn_size_v = numpy.concatenate(
            [dyn_size_v, rnd.randint(1, n_time + 1, size=(n_batch - dyn_size_v.shape[0],))], axis=0)
        d[dyn_size] = dyn_size_v
        if not same_time:
          n_time += 1
    print("%r %r: shape %r" % (data, data.placeholder, shape))
    if data.sparse:
      d[data.placeholder] = rnd.randint(0, data.dim or 13, size=shape, dtype=data.dtype)
    else:
      d[data.placeholder] = rnd.normal(size=shape).astype(data.dtype)
  return d


def dummy_run_net(config: Dict[str, Any], *, train: bool = False, net: Optional[nn.Module] = None, seq_len: int = 5):
  """
  Runs a couple of training iterations using some dummy dataset on the net dict.
  Use this to validate that the net dict is sane.
  Note that this is somewhat slow. The whole TF session setup and net construction can take 5-30 secs or so.
  It is not recommended to use this for every single test case.

  The dummy dataset might change at some point...

  Maybe this gets extended...

  If net is given, it will be used for additional checks, such as whether params match.
  """
  from returnn.tf.engine import Engine
  from returnn.datasets import init_dataset
  from returnn.config import Config
  extern_data_opts = config["extern_data"]
  n_data_dim = extern_data_opts["data"]["dim_tags"][-1].dimension
  n_classes_dim = extern_data_opts["classes"]["sparse_dim"].dimension if "classes" in extern_data_opts else 7
  config = Config({
    "train": {
      "class": "DummyDataset", "input_dim": n_data_dim, "output_dim": n_classes_dim,
      "num_seqs": 2, "seq_len": seq_len},
    "debug_print_layer_output_template": True,
    "task": "train",  # anyway, to random init the net
    **config
  })
  dataset = init_dataset(config.typed_value("train"))
  engine = Engine(config=config)
  engine.init_train_from_config(train_data=dataset)
  if train:
    engine.train()
  else:
    _dummy_forward_net_returnn(engine=engine, dataset=dataset)

  if net is not None:
    check_params(net, engine)

  return engine


def _dummy_forward_net_returnn(*, engine: returnn.tf.engine.Engine, dataset: returnn.datasets.Dataset):
  from returnn.tf.engine import Runner

  def _extra_fetches_cb(*_args, **_kwargs):
    pass  # just ignore

  output = engine.network.get_default_output_layer().output
  batches = dataset.generate_batches(
    recurrent_net=engine.network.recurrent,
    batch_size=engine.batch_size,
    max_seqs=engine.max_seqs,
    used_data_keys=engine.network.get_used_data_keys())
  extra_fetches = {
    'output': output.placeholder,
    "seq_tag": engine.network.get_seq_tags(),
  }
  for i, seq_len in output.size_placeholder.items():
    extra_fetches["seq_len_%i" % i] = seq_len
  forwarder = Runner(
    engine=engine, dataset=dataset, batches=batches,
    train=False, eval=False,
    extra_fetches=extra_fetches,
    extra_fetches_callback=_extra_fetches_cb)
  forwarder.run(report_prefix=engine.get_epoch_str() + " forward")
  if forwarder.run_exception:
    raise forwarder.run_exception
  assert forwarder.finalized


# noinspection PyShadowingNames
def dummy_run_net_single_custom(config: Union[str, Dict[str, Any]], *,
                                make_feed_dict=make_feed_dict,
                                default_out_dim_tag_order: Optional[Sequence[Union[nn.Dim, str]]] = None,
                                eval_flag: bool = False,
                                train_flag: bool = False,
                                ) -> Dict[str, numpy.ndarray]:
  """
  :param config: e.g. via get_complete_py_code_str() or get_config_raw_dict()
  :param make_feed_dict: func (ExternData) -> feed_dict
  :param default_out_dim_tag_order: if given, for the fetch, will order the dims this way
  :param eval_flag: losses are computed if True
  :param train_flag: use (dynamic) train flag if True
  :return: dict with outputs. e.g. contains "layer:output". also specify default_out_dim_tag_order if possible.
  """
  if isinstance(config, str):
    config_dict, net_dict = config_net_dict_via_serialized(config)
  else:
    assert isinstance(config, dict)
    config_dict = config
    net_dict = config_dict["network"]
  from returnn.config import Config
  from returnn.tf.network import TFNetwork
  config = Config(config_dict)
  with make_scope() as session:
    train_flag_ = False
    if train_flag:
      train_flag_ = tf.compat.v1.placeholder(tf.bool, (), name="dyn_train_flag")
    net = TFNetwork(config=config, train_flag=train_flag_, eval_flag=eval_flag)
    net.construct_from_dict(net_dict)
    net.initialize_params(session)
    feed_dict = make_feed_dict(net.extern_data)
    if train_flag:
      feed_dict[train_flag_] = True
    fetches = net.get_fetches_dict(should_eval=eval_flag)
    have_default_out = False
    for out in net.extern_data.data.values():
      fetches[f"data:{out.name}"] = out.placeholder
    for layer in net.get_output_layers():
      if layer.get_absolute_name() == "output":
        have_default_out = True
        if default_out_dim_tag_order:
          out = layer.output
          out = out.copy_transpose([out.get_axis_from_description(a) for a in default_out_dim_tag_order])
          fetches["layer:output"] = out.placeholder
          continue
      assert f"layer:{layer.name}" not in fetches
      fetches[f"layer:{layer.name}"] = layer.output.placeholder
    if default_out_dim_tag_order:
      assert have_default_out
    return session.run(fetches, feed_dict=feed_dict)


dummy_default_in_dim = nn.FeatureDim("input_dim", 13)


def dummy_config_net_dict(net_maker: Callable[[], nn.Module], *,
                          with_axis=False, in_dim: nn.Dim = dummy_default_in_dim, reset_name_ctx: bool = True
                          ) -> Tuple[Dict[str, Any], Dict[str, Any], nn.Module]:
  """
  :return: config, net_dict, net
  """
  if reset_name_ctx:
    nn.reset_default_root_name_ctx()
  net = net_maker()
  time_dim = nn.SpatialDim("time")
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
  opts = {}
  if with_axis:
    opts["axis"] = time_dim
  out = net(data, **opts)
  if isinstance(out, tuple):
    out = out[0]
  assert isinstance(out, nn.Tensor)
  out.mark_as_default_output()

  config_code = nn.get_returnn_config().get_complete_py_code_str(net)
  return config_net_dict_via_serialized(config_code) + (net,)


def config_net_dict_via_serialized(config_code: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """
  :param str config_code: via get_returnn_config_serialized
  :return: config, net_dict
  """
  from returnn.util import better_exchook
  print(config_code)
  scope = {}
  src_filename = "<config_net_dict_via_serialized>"
  better_exchook.set_linecache(src_filename, config_code)
  code_ = compile(config_code, src_filename, "exec")
  exec(code_, scope, scope)
  for tmp in ["__builtins__", "Dim", "batch_dim", "FeatureDim", "SpatialDim"]:
    scope.pop(tmp)
  config = scope
  net_dict = config["network"]
  return config, net_dict


def check_params(net: nn.Module, engine: returnn.tf.engine.Engine):
  """
  Check that params match as expected.
  """
  # noinspection PyProtectedMember
  from ..nn.naming import _NamePathCache
  rc_naming = _NamePathCache()
  rc_naming.register_module(net, [])
  rc_params = set()
  for param in net.parameters():
    assert isinstance(param, nn.Parameter)
    param_name = "/".join(rc_naming.get_name_path(param))
    print("param:", repr(param_name), param)
    rc_params.add(param_name)

  rc_named_params = set()
  for name, param in net.named_parameters():
    assert isinstance(name, str)
    assert isinstance(param, nn.Parameter)
    rc_named_params.add(name.replace(".", "/"))

  tf_params = set()
  for param in engine.network.get_params_list():
    assert isinstance(param, tf.Variable)
    assert param.name.endswith("/param:0")
    if "random" in param.name and "state_var" in param.name:
      print("Note: Ignoring random state var:", param)
      continue  # be relaxed about these
    tf_params.add(param.name[:-len("/param:0")])

  if rc_params != tf_params:
    print("RETURNN-common params:", rc_params)
    print("TF params:", tf_params)
    raise Exception(f"Mismatch of params RETURNN-common {rc_params} vs TF {tf_params}")

  if rc_named_params != tf_params:
    print("RETURNN-common params:", rc_params)
    print("TF params:", tf_params)
    raise Exception(f"Mismatch of params RETURNN-common {rc_named_params} vs TF {tf_params}")
