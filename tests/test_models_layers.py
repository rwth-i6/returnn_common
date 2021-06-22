"""
Test layers
"""
from __future__ import annotations

from . import _setup_test_env  # noqa

from returnn_common.models.layers import *
from returnn_common.models.base import get_extern_data, NameCtx, Layer, LayerRef
from pprint import pprint
from nose.tools import assert_equal
import returnn.tf.engine
import returnn.datasets


def _test_net_returnn(net_dict: Dict[str, Dict[str, Any]], *, train: bool = False):
  """
  Runs a couple of training iterations using some dummy dataset on the net dict.
  Use this to validate that the net dict is sane.
  Note that this is somewhat slow. The whole TF session setup and net construction can take 5-30 secs or so.
  It is not recommended to use this for every single test case.

  The dummy dataset might change at some point...

  Maybe this gets extended...
  """
  from returnn.tf.engine import Engine
  from returnn.datasets import init_dataset
  from returnn.config import Config
  n_data_dim, n_classes_dim = 5, 7
  config = Config({
    "train": {
      "class": "DummyDataset", "input_dim": n_data_dim, "output_dim": n_classes_dim,
      "num_seqs": 2, "seq_len": 5},
    "extern_data": {"data": {"dim": n_data_dim}, "classes": {"dim": n_classes_dim, "sparse": True}},
    "network": net_dict,
    "debug_print_layer_output_template": True,
    "task": "train",  # anyway, to random init the net
  })
  dataset = init_dataset(config.typed_value("train"))
  engine = Engine(config=config)
  engine.init_train_from_config(train_data=dataset)
  if train:
    engine.train()
  else:
    _dummy_forward_net_returnn(engine=engine, dataset=dataset)


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


def test_simple_net():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.lstm = Lstm(n_out=13)

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x = self.lstm(x)
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert "lstm" in net_dict
  _test_net_returnn(net_dict)


def test_simple_net_share_params():
  class _Net(Module):
    def __init__(self):
      super().__init__()
      self.linear = Linear(n_out=13, activation=None)
      self.lstm = Lstm(n_out=13)

    def forward(self) -> LayerRef:
      """
      Forward
      """
      x = get_extern_data("data")
      x = self.linear(x)
      x = self.lstm(x)
      x = self.lstm(x)
      return x

  net = _Net()
  net_dict = net.make_root_net_dict()
  pprint(net_dict)
  assert "lstm" in net_dict
  assert "lstm_0" in net_dict
  assert_equal(net_dict["lstm_0"]["reuse_params"], "lstm")
  _test_net_returnn(net_dict)


def test_explicit_root_ctx():
  class Net(Module):
    """
    Net
    """
    def __init__(self, l2=1e-07, dropout=0.1, n_out=13):
      super().__init__()
      self.linear = Linear(n_out=n_out, l2=l2, dropout=dropout, with_bias=False, activation=None)

    def forward(self, x: LayerRef) -> LayerRef:
      """
      forward
      """
      x = self.linear(x)
      return x

  with NameCtx.new_root() as name_ctx:
    net = Net()
    out = net(get_extern_data("data"))
    assert isinstance(out, Layer)
    assert_equal(out.get_name(), "Net")

    Copy()(out, name="output")  # make some dummy output layer
    net_dict = name_ctx.make_net_dict()
    pprint(net_dict)

  assert "Net" in net_dict
  sub_net_dict = net_dict["Net"]["subnetwork"]
  assert "linear" in sub_net_dict
  lin_layer_dict = sub_net_dict["linear"]
  assert_equal(lin_layer_dict["class"], "linear")
  assert_equal(lin_layer_dict["from"], "base:data:data")
  _test_net_returnn(net_dict)


def test_root_mod_call_twice():
  class TestBlock(Module):
    """
    Test block
    """
    def __init__(self, l2=1e-07, dropout=0.1, n_out=13):
      super().__init__()
      self.linear = Linear(n_out=n_out, l2=l2, dropout=dropout, with_bias=False, activation=None)

    def forward(self, x: LayerRef) -> LayerRef:
      """
      forward
      """
      x = self.linear(x)
      return x

  with NameCtx.new_root() as name_ctx:
    test_block = TestBlock()
    y = test_block(get_extern_data("input1"))
    z = test_block(get_extern_data("input2"))

    print(y)
    assert isinstance(y, LayerRef)
    assert_equal(y.get_name(), "TestBlock")
    print(z)
    assert isinstance(z, LayerRef)
    assert_equal(z.get_name(), "TestBlock_0")

    net_dict = name_ctx.make_net_dict()
    pprint(net_dict)

  assert "TestBlock" in net_dict and "TestBlock_0" in net_dict
  assert_equal(net_dict["TestBlock_0"]["reuse_params"], "TestBlock")
