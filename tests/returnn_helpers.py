"""
Helpers for RETURNN
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
import tensorflow as tf
import returnn.tf.engine
import returnn.datasets
from .. import nn


def dummy_run_net(config: Dict[str, Any], *, train: bool = False, net: Optional[nn.Module] = None):
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
      "num_seqs": 2, "seq_len": 5},
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


def dummy_config_net_dict(net: nn.Module, *,
                          with_axis=False, in_dim: int = 13
                          ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """
  :return: config, net_dict
  """
  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  in_dim = nn.FeatureDim("input", in_dim)
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
  return config_net_dict_via_serialized(config_code)


def config_net_dict_via_serialized(config_code: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """
  :param str config_code: via get_returnn_config_serialized
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
    tf_params.add(param.name[:-len("/param:0")])

  if rc_params != tf_params:
    print("RETURNN-common params:", rc_params)
    print("TF params:", tf_params)
    raise Exception(f"Mismatch of params RETURNN-common {rc_params} vs TF {tf_params}")

  if rc_named_params != tf_params:
    print("RETURNN-common params:", rc_params)
    print("TF params:", tf_params)
    raise Exception(f"Mismatch of params RETURNN-common {rc_named_params} vs TF {tf_params}")
