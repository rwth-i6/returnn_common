"""
Loop. Provides :class:`Loop`.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, Iterable
from tensorflow.python.util import nest
from returnn.util.basic import NotSpecified
from .. import nn


class Loop:
  """
  This represents a RecLayer subnetwork in RETURNN,
  i.e. where the calculation per step is defined explicitly.

  (For RecLayer with a predefined unit, see :class:`Rec`.
   Or for example :class:`Lstm`.)

  To define a loop like this pseudo Python code::

    x  # given, shape (batch, time, dim)
    h = Zeros([batch,dim])()  # initial state, shape (batch,dim)
    out = []
    for t in range(x.max_seq_len):
      x_lin = Linear(dim)(x[t])
      h_prev = h
      h = Linear(dim)(x_lin + h_prev)
      out.append(h)

    h  # final state
    out  # shape (time, batch, h_dim)

  You would write::

    dim = nn.FeatureDim(...)
    loop = nn.Loop(axis=...)
    loop.state.h = nn.zeros([batch_dim,dim])  # initial state
    with loop:
      x_t = loop.unstack(x)
      x_lin = Linear(dim)(x_t)
      loop.state.h = Linear(dim)(x_lin + loop.state.h)
      out = loop.stack(loop.state.h)

  ``state`` is :class:`Loop._StateHolder` and manages the recurrent state.

  This code must be run within a :func:`Module.forward`
  or with some active global name context (:class:`NameCtx`).

  This API is currently in development, and might change.
  See: https://github.com/rwth-i6/returnn_common/issues/16
  """

  def __init__(self, *,
               max_seq_len: Optional[Union[str, int, callable]] = NotSpecified,
               optimize_move_layers_out: Optional[bool] = NotSpecified,
               unroll: bool = NotSpecified,
               axis: Optional[nn.Dim] = NotSpecified,
               debug: Optional[bool] = NotSpecified,
               name: str = "loop"
               ):
    super(Loop, self).__init__()
    if not axis or axis is NotSpecified:
      axis = nn.SpatialDim(f"{name}-dim")
    self.extra_opts = {
      key: value for (key, value) in locals().items()
      if value is not NotSpecified and key not in {"self", "__class__", "name"}}
    self.layer_module = LoopModule(loop=self)
    self.name_ctx = nn.NameCtx(module=self.layer_module, suggested_name=name, parent=nn.NameCtx.current_ctx())
    self.name_ctx.is_subnet_ctx = True
    self.name_ctx.extend_reserved_names({"output", "end"})
    self._entered_scope = False
    self._exited_scope = False
    self._state = _LoopStateHolder(loop=self)
    self.unstacked_refs = []  # type: List[nn.Tensor]
    self.outputs = []  # type: List[nn.Tensor]
    self._has_given_axis = bool(axis)
    self.axis = axis
    self.end_ref = None  # type: Optional[nn.Tensor]

  def __repr__(self):
    return f"<{self.__class__.__name__} {self.name_ctx.get_abs_name_repr()}>"

  def __enter__(self) -> Loop:
    assert not self._entered_scope, f"{self}: cannot enter twice"
    self._entered_scope = True
    self.name_ctx.__enter__()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert not self._exited_scope, f"{self}: cannot exit twice"
    self._exited_scope = True
    try:
      if not exc_type:
        if not self.outputs:  # stack or last was called at least once, so we have some output
          raise Exception(f"{self}: call `stack` or `last` at least once to define some output")
        if not self.end_ref and not self.unstacked_refs:
          raise Exception(f"{self}: call `unstack` or `end` at least once to define the loop length")
        # Make sure there is an "output" layer. (Similar as for Module with subnetwork.)
        if "output" not in self.name_ctx.children:
          from . import copy
          copy(self.outputs[0], name=self.name_ctx.get_child("output"))
    finally:
      self.name_ctx.__exit__(exc_type, exc_val, exc_tb)
    if not exc_type:
      self.layer_module()  # create the rec layer itself

  @property
  def has_entered_scope(self) -> bool:
    """
    :return: whether we have entered the scope, i.e. we define the per-step calculation.
    """
    return self._entered_scope

  @property
  def state(self) -> Union[_LoopStateHolder, nn.LayerState]:
    """state holder inside the loop"""
    if not self._exited_scope:
      return self._state
    # noinspection PyProtectedMember
    return self._state._get_last()

  @state.setter
  def state(self, initial_state: nn.LayerState):
    assert len(self._state) == 0, f"can only assign {self}.state once for the initial state"
    for key, value in initial_state.items():
      self._state[key] = value

  def unstack(self, source: nn.Tensor, *,
              name: Optional[str] = None
              ) -> nn.Tensor:
    """
    Unrolls over the specified axis, and provides each frame in each loop iteration.
    The axis can be specified globally for the :class:`Loop` instance (recommended)
    or locally here (not recommended).
    """
    from . import rec_unstack
    assert self._has_given_axis, "%s: unstack() requires a given axis" % self
    res = rec_unstack(source, axis=self.axis, name=name)
    self.unstacked_refs.append(res)
    return res

  def stack(self, source: nn.Tensor, *, name: Optional[str] = None) -> nn.Tensor:
    """
    Accumulates the frames of source within the loop,
    to make it accessible outside the loop.
    """
    from . import copy
    # We don't need to do anything special because RETURNN RecLayer will automatically accumulate the frames
    # when we marked a layer with is_output_layer, and we access it from outside the loop.
    if not name and "output" not in self.name_ctx.children:
      name = self.name_ctx.get_child("output")
    res = copy(source, name=name)
    assert isinstance(res, nn.Tensor)
    if res.name_ctx.name != "output":
      res.layer_dict["is_output_layer"] = True
    # We access the returned layer-ref from outside, thus fix the data template.
    res.data = res.data.copy_add_dim_by_tag(dim_tag=self.axis, unbroadcast=True, axis=0)
    res.data.time_dim_axis = 0
    self.outputs.append(res)
    return res

  def last(self, source: nn.Tensor, *, name: Optional[str] = None) -> nn.Tensor:
    """
    Gets the last value from source.
    """
    assert isinstance(source, nn.Tensor)
    source.layer_dict["need_last"] = True
    sub_layer_name = source.name_ctx.get_name_in_ctx(self.name_ctx)
    with self.name_ctx.parent:  # need to be outside the loop
      return nn.make_layer(
        {"class": "rec_last_output", "rec_layer": self.name_ctx.layer_ref, "sub_layer_name": sub_layer_name},
        name=name or sub_layer_name.replace("/", "_"))

  def end(self, source: nn.Tensor, *, include_eos: bool) -> nn.Tensor:
    """
    For loops with dynamic ending condition (which might not use unstack),
    this defines the ending condition.

    :param source: the ending condition
    :param include_eos: if True, the last() and stack() function include the current ending frame, otherwise not
    """
    assert not self.end_ref, f"{self}.end() can only be called once"
    self.extra_opts["include_eos"] = include_eos
    from . import copy
    self.end_ref = copy(source, name=self.name_ctx.get_child("end"))
    return self.end_ref


class LoopModule(nn.Module):
  """
  This module is used internally by :class:`Loop` to create the RETURNN :class:`RecLayer` for the loop.
  This module would not be directly used by the user.
  """
  layer_name_scope = ""

  def __init__(self, loop: Loop):
    super(LoopModule, self).__init__()
    self.loop = loop

  def __call__(self) -> nn.Tensor:
    """
    Makes layer dict for this loop, i.e. a RecLayer.
    """
    name_ctx = self.loop.name_ctx
    out = name_ctx.children["output"].layer_ref
    return nn.make_layer(
      {"class": "rec", "from": [], "name_scope": "", "unit": name_ctx.make_net(), **self.loop.extra_opts},
      name=name_ctx,
      predefined_out_data=out.data.copy_add_dim_by_tag(self.loop.axis, unbroadcast=True, axis=0))


class PrevTensorRef(nn.Tensor):
  """
  Refers to a layer from the previous loop iteration.
  """
  @classmethod
  def get_prev_ref(cls, *, cur_layer_name_ctx: nn.NameCtx, initial: nn.Tensor) -> PrevTensorRef:
    """
    Create prev ref.
    """
    parent_name_ctx = cur_layer_name_ctx.parent
    prev_layer_name_ctx = parent_name_ctx.get_child(f"prev:{cur_layer_name_ctx.name}")
    if prev_layer_name_ctx.layer_ref:
      prev_layer_ref = prev_layer_name_ctx.layer_ref
      assert isinstance(prev_layer_ref, PrevTensorRef)
      assert prev_layer_ref.cur_layer_name_ctx is cur_layer_name_ctx
    else:
      prev_layer_ref = PrevTensorRef(
        name_ctx=prev_layer_name_ctx, cur_layer_name_ctx=cur_layer_name_ctx, data=initial.data)
      assert prev_layer_name_ctx.layer_ref is prev_layer_ref
    return prev_layer_ref

  def __init__(self, *, name_ctx: nn.NameCtx, cur_layer_name_ctx: nn.NameCtx, data: nn.Data):
    # At the time we instantiate this, cur_layer_name_ctx.layer probably does not exist yet.
    super().__init__(name_ctx=name_ctx, data=data, is_ref=True)
    self.cur_layer_name_ctx = cur_layer_name_ctx


class _LoopStateHolder:
  def __init__(self, loop: Loop):
    self._loop = loop
    self._state = {}  # type: Dict[str, _LoopState]

  def __repr__(self):
    return f"{self._loop}.state"

  def _get_state(self, name: str) -> _LoopState:
    if name in self._state:
      return self._state[name]
    raise AttributeError(f"{self}: Unknown state attrib {name!r}. Assign the initial state first.")

  def _get_last(self) -> nn.LayerState:
    return nn.LayerState({key: value.get_last() for (key, value) in self._state.items()})

  def __getitem__(self, item):
    return self._get_state(item).get()

  def __setitem__(self, key, value):
    if not self._loop.has_entered_scope:
      assert key not in self._state, f"{self} already has state {key!r}"
      self._state[key] = _LoopState(name=key, loop=self._loop, initial=value)
      return
    self._get_state(key).assign(value)

  def __getattr__(self, item):
    return self[item]

  def __setattr__(self, key, value):
    if key in {"_state", "_loop"}:
      return super().__setattr__(key, value)
    self[key] = value

  def keys(self) -> Iterable[str]:
    """keys"""
    return self._state.keys()

  def __len__(self):
    return len(self._state)


class _LoopState:
  """
  Represents some recurrent state, to be used with :class:`Loop`.
  It can also represent some nested hierarchy of states.
  """

  def __init__(self, *, name: str, loop: Loop, initial: Union[nn.Tensor, Any]):
    """
    :param name:
    :param loop:
    :param initial: some layer-ref, or any kind of nested structure of layers.
    """
    super(_LoopState, self).__init__()
    assert initial is not None
    initial = nest.map_structure(nn.convert_to_layer_ref, initial)
    self.initial = initial
    self.loop = loop
    self.name = name
    self.assigned_value = None
    self.name_ctx = nest.map_structure_with_tuple_paths(
      lambda path, ref: nn.NameCtx(
        suggested_name='.'.join(str(key) for key in ('state', name) + path),
        parent=loop.name_ctx),
      self.initial)

  def assign(self, value):
    """
    Assign the new value for the current iteration.
    """
    assert self.name_ctx is not None
    assert value is not None
    assert self.assigned_value is None, (
      f"Cannot assign the rec state {self.loop}/{self.name} multiple times, "
      f"assigned previously to {self.assigned_value}, now to {value}")
    nest.assert_same_structure(self.initial, value)
    nest.assert_same_structure(self.name_ctx, value)
    self.assigned_value = value

    def _map_ref_to_name_ctx(layer_ref: nn.Tensor, name_ctx: nn.NameCtx, initial: nn.Tensor):
      assert isinstance(layer_ref, nn.Tensor)
      assert isinstance(name_ctx, nn.NameCtx)

      # Potential optimization for RETURNN layers.
      # See ReturnnWrappedLayerBase._get_recurrent_state.
      if isinstance(layer_ref, nn.Tensor):
        if layer_ref.layer_dict["class"] == "get_last_hidden_state":
          used_state_eliminate_optimization = False
          key = layer_ref.layer_dict.get("key", "state")
          src = layer_ref.layer_dict["from"]
          assert isinstance(src, nn.Tensor)
          src_state_opt = src.layer_dict.get("state")
          if isinstance(src_state_opt, nn.LayerState):
            src_state_for_key = src_state_opt.get(key)
            if isinstance(src_state_for_key, PrevTensorRef):
              if src_state_for_key.cur_layer_name_ctx is name_ctx:
                # The 'state' argument of the rec layer refers to "prev:..." of the state.
                # So we don't need to pass it now.
                used_state_eliminate_optimization = True
                src_state_opt[key] = None
                if all(opt is None for opt in nest.flatten(src_state_opt)):
                  del src.layer_dict["state"]
                # We need to pass the initial_state instead though.
                src_initial_state_opt = src.layer_dict.setdefault("initial_state", nn.LayerState())
                src_initial_state_opt[key] = initial
                # If there is any other code which refers to this state, it can access the passed layer.
                # So anyway pass through.

          if not used_state_eliminate_optimization:
            raise NotImplementedError(
              f"{self}.assign to {layer_ref} on {src}:"
              f" We need https://github.com/rwth-i6/returnn_common/issues/31"
              f" and https://github.com/rwth-i6/returnn/issues/732.")

        else:  # class != get_last_hidden_state
          assert "initial_state" not in layer_ref.layer_dict
          assert "initial_output" not in layer_ref.layer_dict
          layer_ref.layer_dict["initial_output"] = initial

      else:  # layer_ref not Layer
        raise NotImplementedError(f"{self}.assign to {layer_ref} but layer expected")

      name_ctx.move_layer_ref_here(layer_ref)

    nest.map_structure(_map_ref_to_name_ctx, value, self.name_ctx, self.initial)

  @staticmethod
  def _map_name_ctx_to_prev_layer_ref(name_ctx: nn.NameCtx, initial: nn.Tensor) -> PrevTensorRef:
    assert isinstance(name_ctx, nn.NameCtx)
    return PrevTensorRef.get_prev_ref(cur_layer_name_ctx=name_ctx, initial=initial)

  def get(self):
    """
    Return prev or current value
    """
    assert self.name_ctx is not None
    if self.assigned_value is None:  # not yet assigned
      # Return prev value
      return nest.map_structure(self._map_name_ctx_to_prev_layer_ref, self.name_ctx, self.initial)
    return self.assigned_value

  def _map_name_ctx_to_last_layer_ref(self, name_ctx: nn.NameCtx) -> nn.Tensor:
    assert isinstance(name_ctx, nn.NameCtx)
    assert name_ctx.layer_ref, f"{self.loop} state {name_ctx} not assigned?"
    assert self.loop.name_ctx.layer_ref, f"{self.loop} not yet exited?"
    return self.loop.last(name_ctx.layer_ref)

  def get_last(self):
    """
    Outside the loop, get the last instance.
    """
    assert self.name_ctx is not None
    assert self.assigned_value is not None
    return nest.map_structure(self._map_name_ctx_to_last_layer_ref, self.name_ctx)
