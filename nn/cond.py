"""
Conditional logic

https://github.com/rwth-i6/returnn_common/issues/24
"""

from tensorflow.python.util import nest
from .. import nn


class Cond:
  """
  Conditional branching. Basically behaves like ``if ... else ...``.
  Only one branch will be executed, and the condition needs to be a bool scalar.
  This wraps to :class:`CondLayer` in RETURNN and to ``tf.cond`` in TensorFlow.

  Example::

      with Cond(cond) as cond_obj:
        cond_obj.true = mod_true_case(x)
        cond_obj.false = mod_false_case(x)
        y = cond_obj.result

  Corresponds to::

      if cond:
        y = mod_true_case(x)
      else:
        y = mod_false_case(x)

  The context scope has two states corresponding to the True and False computation branch.
  The initial state is the True branch.
  Assigning ``cond_obj.true`` has the side effect of switching the computation to the False branch.
  """

  def __init__(self, condition: nn.Tensor, *, name: str = "cond"):
    self.condition = condition
    self._entered = False
    self._entered_state = True
    self._true_value = None
    self._false_value = None
    self._result_value = None
    self.layer_module = CondModule(cond=self)
    self.name_ctx = nn.NameCtx(
      module=self.layer_module, suggested_name=name, parent=nn.NameCtx.current_ctx(), can_access_children=False)
    self.name_ctx.custom_layer_name_scope = ""
    self.true_branch_name_ctx = nn.NameCtx(
      module=self.layer_module, suggested_name="true", parent=self.name_ctx, virtual=True, can_access_children=False)
    self.true_branch_name_ctx.is_subnet_ctx = True
    self.true_branch_name_ctx.extend_reserved_names({"output"})
    self.true_branch_control_flow_ctx = nn.ControlFlowContext(
      kind=nn.ControlFlowContext.Types.Cond, outer_ctx=nn.NameCtx.inner_control_flow())
    self.false_branch_name_ctx = nn.NameCtx(
      module=self.layer_module, suggested_name="false", parent=self.name_ctx, virtual=True, can_access_children=False)
    self.false_branch_name_ctx.is_subnet_ctx = True
    self.false_branch_name_ctx.extend_reserved_names({"output"})
    self.false_branch_control_flow_ctx = nn.ControlFlowContext(
      kind=nn.ControlFlowContext.Types.Cond, outer_ctx=nn.NameCtx.inner_control_flow())

  def __repr__(self):
    return f"Cond{self.name_ctx}"

  def __enter__(self):
    assert not self._entered, f"{self} cannot enter twice"
    self._entered = True
    self._entered_state = True
    self.true_branch_name_ctx.__enter__()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # First exit any scopes and do cleanup without throwing any exceptions.
    if self._entered:
      if self._true_value is None:
        self.true_branch_name_ctx.__exit__(exc_type, exc_val, exc_tb)
      elif self._false_value is None:
        self.false_branch_name_ctx.__exit__(exc_type, exc_val, exc_tb)
    if not exc_type:  # only do error checking if there was no other exception
      assert self._entered
      assert self._true_value is not None, f"{self} you need to call else_()"
      assert self._false_value is not None, f"{self} you need to call end()"
    self._entered = False

  @property
  def true(self):
    """
    The getter usually would not be used.
    """
    return self._true_value

  @true.setter
  def true(self, true_value):
    """
    Enter the False branch.
    """
    assert self._entered, f"{self} you need to be in the context scope"
    assert self._entered_state is True, f"{self} you cannot enter the else branch twice"
    assert true_value is not None
    assert self._true_value is None
    assert isinstance(true_value, nn.Tensor)  # not implemented otherwise
    nn.copy(true_value, name=self.true_branch_name_ctx.get_child("output"))
    self.true_branch_name_ctx.__exit__(None, None, None)
    self.false_branch_name_ctx.__enter__()
    self._true_value = true_value
    self._entered_state = False

  @property
  def false(self):
    """
    The getter usually would not be used.
    """
    return self._false_value

  @false.setter
  def false(self, false_value):
    """
    Define the False branch value,
    and
    :return: cond(condition, true_value, false_value).
    """
    assert self._entered, f"{self} you need to be in the context scope"
    assert self._entered_state is False, f"{self} you need to be in the False branch, have assigned :func:`true` before"
    assert false_value is not None
    assert self._false_value is None
    nest.assert_same_structure(self._true_value, self._false_value)
    assert isinstance(false_value, nn.Tensor)  # not implemented otherwise
    nn.copy(false_value, name=self.false_branch_name_ctx.get_child("output"))
    self.false_branch_name_ctx.__exit__(None, None, None)
    self._false_value = false_value
    self._result_value = self.layer_module()

  @property
  def result(self) -> nn.Tensor:
    """
    :return: the result, after you assigned :func:`true` and :func:`false`.
    """
    assert self._true_value is not None, f"{self} you need to have defined the true value"
    assert self._false_value is not None, f"{self} you need to have defined the false value"
    assert self._result_value is not None
    return self._result_value

  @property
  def control_flow_ctx(self) -> nn.ControlFlowContext:
    """
    :return: the control flow context.
    """
    assert self._entered, f"{self}: you need to be in the context scope"
    if self._entered_state is True:
      return self.true_branch_control_flow_ctx
    elif self._entered_state is False:
      return self.false_branch_control_flow_ctx
    else:
      assert False, f"{self}: invalid state {self._entered_state!r}"


class CondModule(nn.Module):
  """
  This module is used internally by :class:`Cond` to create the RETURNN :class:`CondLayer` for the conditional code.
  This module would not be directly used by the user.
  """

  def __init__(self, cond: Cond):
    super(CondModule, self).__init__()
    self.cond = cond

  def __call__(self) -> nn.Tensor:
    """
    Makes layer dict for this loop, i.e. a RecLayer.
    """
    name_ctx = self.cond.name_ctx
    # noinspection PyProtectedMember
    true_value = self.cond._true_value
    assert isinstance(true_value, nn.Tensor)  # not implemented otherwise
    return nn.make_layer(
      {
        "class": "cond", "from": [],
        "condition": self.cond.condition,
        "true_layer": {
          "class": "subnetwork", "from": [], "subnetwork": self.cond.true_branch_name_ctx.make_net()},
        "false_layer": {
          "class": "subnetwork", "from": [], "subnetwork": self.cond.false_branch_name_ctx.make_net()},
      },
      name=name_ctx,
      predefined_out_data=true_value.data)
