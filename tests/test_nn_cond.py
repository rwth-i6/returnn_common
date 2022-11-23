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

  from returnn_common.asr.specaugment import random_mask_v2
  from typing import Optional, Sequence, Dict, Tuple
  import contextlib

  class Model(nn.Module):
    """Model definition"""

    def __init__(self, in_dim: nn.Dim, *,
                 nb_target_dim: nn.Dim,
                 wb_target_dim: nn.Dim,
                 blank_idx: int,
                 bos_idx: int,
                 enc_key_total_dim: nn.Dim = nn.FeatureDim("enc_key_total_dim", 2),
                 att_num_heads: nn.Dim = nn.SpatialDim("att_num_heads", 1),
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

      self.enc_key_total_dim = enc_key_total_dim
      self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
      self.att_num_heads = att_num_heads

      self.enc_ctx = nn.Linear(self.encoder.out_dim, enc_key_total_dim)
      self.enc_ctx_dropout = 0.2
      self.enc_win_dim = nn.SpatialDim("enc_win_dim", 5)
      self.att_query = nn.Linear(self.encoder.out_dim, enc_key_total_dim, with_bias=False)
      self.lm = DecoderLabelSync(nb_target_dim)
      self.readout_in_am = nn.Linear(2 * self.encoder.out_dim, nn.FeatureDim("readout", 1000), with_bias=False)
      self.readout_in_am_dropout = 0.1
      self.readout_in_lm = nn.Linear(self.lm.out_dim, self.readout_in_am.out_dim, with_bias=False)
      self.readout_in_lm_dropout = 0.1
      self.readout_in_bias = nn.Parameter([self.readout_in_am.out_dim])
      self.readout_reduce_num_pieces = 2
      self.readout_dim = self.readout_in_am.out_dim // self.readout_reduce_num_pieces
      self.out_nb_label_logits = nn.Linear(self.readout_dim, nb_target_dim)
      self.label_log_prob_dropout = 0.3
      self.out_emit_logit = nn.Linear(self.readout_dim, nn.FeatureDim("emit", 1))

    def encode(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim,
               ) -> Tuple[Dict[str, nn.Tensor], nn.Dim]:
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
      enc_ctx = self.enc_ctx(nn.dropout(enc, self.enc_ctx_dropout, axis=enc.feature_dim))
      enc_ctx_win, _ = nn.window(enc_ctx, spatial_dim=enc_spatial_dim, window_dim=self.enc_win_dim)
      enc_val_win, _ = nn.window(enc, spatial_dim=enc_spatial_dim, window_dim=self.enc_win_dim)
      return dict(enc=enc, enc_ctx_win=enc_ctx_win, enc_val_win=enc_val_win), enc_spatial_dim

    @staticmethod
    def encoder_unstack(ext: Dict[str, nn.Tensor]) -> Dict[str, nn.Tensor]:
      """
      prepare the encoder output for the loop (full-sum or time-sync)
      """
      # We might improve or generalize the interface later...
      # https://github.com/rwth-i6/returnn_common/issues/202
      loop = nn.NameCtx.inner_loop()
      return {k: loop.unstack(v) for k, v in ext.items()}

    def decoder_default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
      """Default initial state"""
      return nn.LayerState(lm=self.lm.default_initial_state(batch_dims=batch_dims))

    def decode(self, *,
               enc: nn.Tensor,  # single frame if axis is single step, or sequence otherwise ("am" before)
               enc_spatial_dim: nn.Dim,  # single step or time axis,
               enc_ctx_win: nn.Tensor,  # like enc
               enc_val_win: nn.Tensor,  # like enc
               all_combinations_out: bool = False,  # [...,prev_nb_target_spatial_dim,axis] out
               prev_nb_target: Optional[nn.Tensor] = None,  # non-blank
               prev_nb_target_spatial_dim: Optional[nn.Dim] = None,  # one longer than target_spatial_dim, due to BOS
               prev_wb_target: Optional[nn.Tensor] = None,  # with blank
               wb_target_spatial_dim: Optional[nn.Dim] = None,  # single step or align-label spatial axis
               state: Optional[nn.LayerState] = None,
               ) -> (ProbsFromReadout, nn.LayerState):
      """decoder step, or operating on full seq"""
      if state is None:
        assert enc_spatial_dim != nn.single_step_dim, "state should be explicit, to avoid mistakes"
        batch_dims = enc.batch_dims_ordered(
          remove=(enc.feature_dim, enc_spatial_dim)
          if enc_spatial_dim != nn.single_step_dim
          else (enc.feature_dim,))
        state = self.decoder_default_initial_state(batch_dims=batch_dims)
      state_ = nn.LayerState()

      att_query = self.att_query(enc)
      att_energy = nn.dot(enc_ctx_win, att_query, reduce=att_query.feature_dim)
      att_energy = att_energy * (att_energy.feature_dim.dimension ** -0.5)
      att_weights = nn.softmax(att_energy, axis=self.enc_win_dim)
      att = nn.dot(att_weights, enc_val_win, reduce=self.enc_win_dim)

      if all_combinations_out:
        assert prev_nb_target is not None and prev_nb_target_spatial_dim is not None
        assert prev_nb_target_spatial_dim in prev_nb_target.shape
        assert enc_spatial_dim != nn.single_step_dim
        lm_scope = contextlib.nullcontext()
        lm_input = prev_nb_target
        lm_axis = prev_nb_target_spatial_dim
      else:
        assert prev_wb_target is not None and wb_target_spatial_dim is not None
        assert wb_target_spatial_dim in {enc_spatial_dim, nn.single_step_dim}
        prev_out_emit = prev_wb_target != self.blank_idx
        lm_scope = nn.MaskedComputation(mask=prev_out_emit)
        lm_input = nn.reinterpret_set_sparse_dim(prev_wb_target, out_dim=self.nb_target_dim)
        lm_axis = wb_target_spatial_dim

      with lm_scope:
        lm, state_.lm = self.lm(lm_input, spatial_dim=lm_axis, state=state.lm)

        # We could have simpler code by directly concatenating the readout inputs.
        # However, for better efficiency, keep am/lm path separate initially.
        readout_in_lm_in = nn.dropout(lm, self.readout_in_lm_dropout, axis=lm.feature_dim)
        readout_in_lm = self.readout_in_lm(readout_in_lm_in)

      readout_in_am_in = nn.concat_features(enc, att)
      readout_in_am_in = nn.dropout(readout_in_am_in, self.readout_in_am_dropout, axis=readout_in_am_in.feature_dim)
      readout_in_am = self.readout_in_am(readout_in_am_in)
      readout_in = nn.combine_bc(readout_in_am, "+", readout_in_lm)
      readout_in += self.readout_in_bias
      readout = nn.reduce_out(
        readout_in, mode="max", num_pieces=self.readout_reduce_num_pieces, out_dim=self.readout_dim)

      return ProbsFromReadout(model=self, readout=readout), state_

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

  class ProbsFromReadout:
    """
    functions to calculate the probabilities from the readout
    """

    def __init__(self, *, model: Model, readout: nn.Tensor):
      self.model = model
      self.readout = readout

    def get_label_logits(self) -> nn.Tensor:
      """label log probs"""
      label_logits_in = nn.dropout(self.readout, self.model.label_log_prob_dropout, axis=self.readout.feature_dim)
      label_logits = self.model.out_nb_label_logits(label_logits_in)
      return label_logits

    def get_label_log_probs(self) -> nn.Tensor:
      """label log probs"""
      label_logits = self.get_label_logits()
      label_log_prob = nn.log_softmax(label_logits, axis=label_logits.feature_dim)
      return label_log_prob

    def get_emit_logit(self) -> nn.Tensor:
      """emit logit"""
      emit_logit = self.model.out_emit_logit(self.readout)
      return emit_logit

    def get_wb_label_log_probs(self) -> nn.Tensor:
      """align label log probs"""
      label_log_prob = self.get_label_log_probs()
      label_log_prob = nn.label_smoothed_log_prob_gradient(label_log_prob, 0.1)
      emit_logit = self.get_emit_logit()
      emit_log_prob = nn.log_sigmoid(emit_logit)
      blank_log_prob = nn.log_sigmoid(-emit_logit)
      label_emit_log_prob = label_log_prob + nn.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
      assert self.model.blank_idx == label_log_prob.feature_dim.dimension  # not implemented otherwise
      output_log_prob = nn.concat_features(label_emit_log_prob, blank_log_prob)
      return output_log_prob

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
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    loop = nn.Loop(axis=enc_spatial_dim)  # time-sync transducer
    loop.max_seq_len = nn.dim_value(enc_spatial_dim) * 2
    loop.state.decoder = model.decoder_default_initial_state(batch_dims=batch_dims)
    loop.state.target = nn.constant(model.blank_idx, shape=batch_dims, sparse_dim=model.wb_target_dim)
    with loop:
      enc = model.encoder_unstack(enc_args)
      probs, loop.state.decoder = model.decode(
        **enc,
        enc_spatial_dim=nn.single_step_dim,
        wb_target_spatial_dim=nn.single_step_dim,
        prev_wb_target=loop.state.target,
        state=loop.state.decoder)
      log_prob = probs.get_wb_label_log_probs()
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
