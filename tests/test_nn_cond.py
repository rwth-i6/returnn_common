"""
Test nn.cond
"""
from __future__ import annotations

from . import _setup_test_env  # noqa
from .returnn_helpers import dummy_run_net, dummy_config_net_dict, dummy_run_net_single_custom, \
  config_net_dict_via_serialized, dummy_default_in_dim

import typing

if typing.TYPE_CHECKING:
  from .. import nn
else:
  from returnn_common import nn  # noqa


def test_cond():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      out_dim = nn.FeatureDim("linear-out", 13)
      self.linear_true = nn.Linear(dummy_default_in_dim, out_dim)
      self.linear_false = nn.Linear(dummy_default_in_dim, out_dim)

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      return x

  config, net_dict, net = dummy_config_net_dict(_Net)
  dummy_run_net(config, net=net)


def test_cond_shared_params():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = nn.Linear(dummy_default_in_dim, nn.FeatureDim("linear-out", 13))

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear(x)
        cond.false = self.linear(x * 2.)
        x = cond.result
      return x

  config, net_dict, net = dummy_config_net_dict(_Net)
  engine = dummy_run_net(config, net=net)
  params = engine.network.get_params_list()
  print(params)
  assert len(params) == 2
  assert params[0].name == "linear/bias/param:0"


def test_cond_twice_shared_params():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      out_dim = nn.FeatureDim("linear-out", 13)
      self.pre_linear = nn.Linear(dummy_default_in_dim, out_dim)
      self.linear_true = nn.Linear(out_dim, out_dim)
      self.linear_false = nn.Linear(out_dim, out_dim)

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      x = self.pre_linear(x)
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 1) as cond:
        cond.true = self.linear_true(x)
        cond.false = self.linear_false(x)
        x = cond.result
      return x

  config, net_dict, net = dummy_config_net_dict(_Net)
  dummy_run_net(config, net=net)


def test_cond_random():
  class _Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.rnd = nn.Random()

    def __call__(self, x: nn.Tensor) -> nn.Tensor:
      with nn.Cond(nn.length(nn.batch_dim) % 2 == 0) as cond:
        cond.true = x + self.rnd.normal(x.shape_ordered)
        cond.false = x
        x = cond.result
      return x

  config, net_dict, net = dummy_config_net_dict(_Net)
  dummy_run_net(config, net=net)


def test_cond_new_axis():
  # Like in SelfAttention.
  nn.reset_default_root_name_ctx()
  in_dim = nn.FeatureDim("in", 12)
  time_dim = nn.SpatialDim("time")
  x = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, in_dim]))
  net = nn.Linear(in_dim, in_dim)
  axis = time_dim

  with nn.Cond(nn.dim_value(nn.batch_dim) % 2 == 0) as cond:
    x_ = x
    x_ = net(x_)
    new_dim = nn.SpatialDim(f"{axis.description}:new-dim")
    x_, _ = nn.reinterpret_new_dim(x_, in_dim=axis, out_dim=new_dim)
    x_ = net(x_)
    cond.true = nn.reduce(x_, axis=new_dim, mode="max")
    cond.false = nn.reduce(x, axis=axis, mode="max")
  y = cond.result
  y.mark_as_default_output()

  config_str = nn.get_returnn_config().get_complete_py_code_str(net)
  dummy_run_net_single_custom(config_str)


def test_cond_chunking_conformer():
  # This test needs a huge stack size currently, due to the way RETURNN layer construction works currently.
  # On RETURNN side, there is the option flat_net_construction to solve this,
  # however, it's experimental and also does not work for this case.
  # https://github.com/rwth-i6/returnn/issues/957
  # https://stackoverflow.com/a/16248113/133374
  import resource
  import sys
  try:
    resource.setrlimit(resource.RLIMIT_STACK, (2 ** 29, -1))
  except Exception as exc:
    print(f"resource.setrlimit {type(exc).__name__}: {exc}")
  sys.setrecursionlimit(10 ** 6)

  from typing import Optional, Sequence, Dict, Tuple
  import contextlib

  class Model(nn.Module):
    """Model definition"""

    def __init__(self, in_dim: nn.Dim, *,
                 nb_target_dim: nn.Dim,
                 wb_target_dim: nn.Dim,
                 blank_idx: int,
                 bos_idx: int,
                 ):
      super(Model, self).__init__()
      self.in_dim = in_dim
      self.encoder = nn.ConformerEncoder(
        in_dim,
        nn.FeatureDim("enc", 4),
        ff_dim=nn.FeatureDim("enc-ff", 8),
        input_layer=None,
        num_layers=1,
        num_heads=2,
      )

      self.nb_target_dim = nb_target_dim
      self.wb_target_dim = wb_target_dim
      self.blank_idx = blank_idx
      self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

      self.out_label_logits = nn.Linear(self.encoder.out_dim, wb_target_dim)

    def encode(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim,
               ) -> Tuple[nn.Tensor, nn.Dim]:
      """encode, and extend the encoder output for things we need in the decoder"""
      enc, enc_spatial_dim = source, in_spatial_dim
      with nn.Cond(nn.train_flag()) as cond:
        win_dim = nn.SpatialDim("win", 50)
        stride = 50
        enc_chunked, chunk_spatial_dim = nn.window(
          enc, spatial_dim=enc_spatial_dim,
          window_dim=win_dim, stride=stride)
        enc_, _ = self.encoder(enc_chunked, in_spatial_dim=win_dim)
        enc_ = nn.inverse_window(
          enc_, in_spatial_dim=chunk_spatial_dim, out_spatial_dim=enc_spatial_dim,
          window_dim=win_dim, stride=stride)
        cond.true = enc_
        enc_, _ = self.encoder(enc, in_spatial_dim=enc_spatial_dim)
        cond.false = enc_
      enc = cond.result
      return enc, enc_spatial_dim

    def decode(self,
               enc: nn.Tensor,  # single frame if axis is single step, or sequence otherwise ("am" before)
               ) -> nn.Tensor:
      """decoder step, or operating on full seq"""
      logits = self.out_label_logits(enc)
      return logits

  class DecoderLabelSync(nn.Module):
    """
    Often called the (I)LM part, or prediction network.
    Runs label-sync, i.e. only on non-blank labels.
    """

    def __init__(self, in_dim: nn.Dim, *,
                 embed_dim: nn.Dim = nn.FeatureDim("embed", 256),
                 lstm_dim: nn.Dim = nn.FeatureDim("lstm", 1024),
                 dropout: float = 0.2,
                 l2: float = 0.0001,
                 ):
      super(DecoderLabelSync, self).__init__()
      self.embed = nn.Linear(in_dim, embed_dim)
      self.dropout = dropout
      self.lstm = nn.LSTM(self.embed.out_dim, lstm_dim)
      self.out_dim = self.lstm.out_dim
      for p in self.parameters():
        p.weight_decay = l2

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
      """init"""
      return self.lstm.default_initial_state(batch_dims=batch_dims)

    def __call__(self, source: nn.Tensor, *, spatial_dim: nn.Dim, state: nn.LayerState
                 ) -> Tuple[nn.Tensor, nn.LayerState]:
      embed = self.embed(source)
      embed = nn.dropout(embed, self.dropout, axis=embed.feature_dim)
      lstm, state = self.lstm(embed, spatial_dim=spatial_dim, state=state)
      return lstm, state

  def model_recog(*,
                  model: Model,
                  data: nn.Tensor, data_spatial_dim: nn.Dim,
                  targets_dim: nn.Dim,  # noqa
                  ) -> nn.Tensor:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return: recog results including beam
    """
    batch_dims = data.batch_dims_ordered((data_spatial_dim, data.feature_dim))
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 3

    loop = nn.Loop(axis=enc_spatial_dim)  # time-sync transducer
    loop.max_seq_len = nn.dim_value(enc_spatial_dim) * 2  # WHAT? this triggers it?
    loop.state.target = nn.constant(model.blank_idx, shape=batch_dims, sparse_dim=model.wb_target_dim)
    with loop:
      enc = loop.unstack(enc)
      logits = model.decode(enc)
      log_prob = nn.log_softmax(logits, axis=model.wb_target_dim)
      loop.state.target = nn.choice(
        log_prob, input_type="log_prob",
        target=None, search=True, beam_size=beam_size,
        length_normalization=False)
      res = loop.stack(loop.state.target)
    return res

  nn.reset_default_root_name_ctx()
  time_dim = nn.SpatialDim("time")
  input_dim = nn.FeatureDim("input", 10)
  label_dim = nn.FeatureDim("labels", 5)
  data = nn.get_extern_data(nn.Data("data", dim_tags=[nn.batch_dim, time_dim, input_dim]))

  model = Model(
    input_dim,
    nb_target_dim=label_dim,
    wb_target_dim=label_dim + 1,
    blank_idx=label_dim.dimension,
    bos_idx=0,
  )
  model_recog(model=model, data=data, data_spatial_dim=time_dim, targets_dim=label_dim).mark_as_default_output()

  # config = nn.get_returnn_config().get_config_raw_dict(root_module=model)
  config_code = nn.get_returnn_config().get_complete_py_code_str(model)
  config, net_dict = config_net_dict_via_serialized(config_code)
  dummy_run_net(config, net=model)
