"""
Base interfaces.

The core interfaces for the user are:

* :class:`Module` and using :func:`make_layer` to directly create a RETURNN layer via dict.
  We recommend using this only for directly wrapping RETURNN layers
  and not for any higher-level logic,
  which should be done as a :class:`Module`.

* :class:`Module`, to write PyTorch-style code, which acts like a subnetwork.
  We recommend using this as the base interface
  for any higher-level interfaces
  (such as a generic decoder interface).
  Use :func:`scoped` as a decorator for the ``__call__`` method.

Instances of both objects can be called directly,
and return instances of type :class:`Tensor`,
which can be thought of as analogue to :class:`torch.Tensor` or :class:`tf.Tensor`.

Use ``x.mark_as_loss()`` to mark some output (layer ref) as a loss.

The root network should be a :class:`Module`,
and then you can use ``make_root_net_dict()``
to get the network dict.
Code example::

    class Network(nn.Module):
      def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(nn.FeatureDim("lstm-out", 1024))

      @nn.scoped
      def __call__(self, x: nn.Tensor) -> nn.Tensor:
        y = self.lstm(x)
        return y

    net = Network()
    net_dict = make_root_net_dict(net, "data")

---

Code conventions:

- Usual, as in RETURNN, PEP8, 2-space indents, 120 char line limit.
- Pure interface classes are prefixed with `I`.
  (`Module` is an exception because this is made analogue to PyTorch).

"""

from __future__ import annotations
import numpy
from typing import Dict, Any, Optional, List, Tuple, Union, Set, Sequence, Callable, Type
from returnn.tf.util.data import *  # Dim, Data, and others
# noinspection PyProtectedMember
from returnn.tf.util.data import _MarkedDim
from tensorflow.python.util import nest
from .. import nn


LayerDictRaw = Dict[str, Any]
TensorRefRaw = str
NetDictRaw = Dict[str, LayerDictRaw]
RawTensorTypes = Union[int, float, complex, numpy.number, numpy.ndarray, bool, str]
OutShapeType = Union[Set[Union[Dim, _MarkedDim]], tuple, list]

min_returnn_behavior_version = 12


class Tensor:
  """
  Refers to a layer in RETURNN.

  An instance of this class can be treated very much like a tensor.
  It supports all the common unary and binary math operations such as addition.
  This is the intended view point for the user,
  to treat instances of this class like a tensor.

  You do not create instances of this object explicitly
  but they are created via any of the standard functions
  like :func:`zeros` etc. or any :func:`Module`,
  or via :func:`make_layer` for directly wrapping some RETURNN layer,
  or via :func:`get_extern_data` for external data.
  """
  require_global_access = False

  def __init__(self, *,
               name_ctx: nn.NameCtx,
               data: Optional[Data] = None,
               layer_dict: Optional[LayerDictRaw] = None,
               is_ref: bool = False,
               ):
    """
    :param name_ctx: this defines the name of the layer itself
    :param data: Data template describing the shape and dtype and other meta information on the tensor (layer output)
    :param is_ref: in RETURNN, there can be references to special layers, like "data:..." or "prev:...",
      which are not layers themselves, i.e. we do not have a layer dict for them.
    """
    self.parent_modules = []  # type: List[Tuple[nn.Module, str]]  # with attr
    self.name_ctx = name_ctx
    # Do not assign name_ctx.layer{_ref} yet because we potentially could raise exceptions later.
    assert name_ctx.layer_ref is None
    assert name_ctx.layer is None
    self.debug_layer = None

    if is_ref:
      assert layer_dict is None
    else:  # not is_ref (default)
      assert layer_dict is not None
      # Note that the following code can potentially raise user errors.
      if not data:
        data = _data_from_layer_dict(layer_dict, tensor=self)
      else:
        data = data.copy()
      data.control_flow_ctx = nn.NameCtx.inner_control_flow()
      if data.have_batch_axis() and not data.batch:
        # You could say this is a bug of RETURNN. Or at least RETURNN is just incomplete here.
        # RETURNN usually would fix that later when the layer is actually created,
        # but we don't do that here.
        # We can still try to look at dependencies and use those batch info.
        batches = []
        for dep in self.get_dependencies(_extra_layer_dict=layer_dict):
          if dep.data.batch and dep.data.batch not in batches:
            batches.append(dep.data.batch)
        if batches:
          data.batch = nn.BatchInfo.get_common_batch_info(batches)
        elif name_ctx.root.global_batch:
          data.batch = name_ctx.root.global_batch

    self.data = data
    self.layer_dict = layer_dict
    name_ctx.layer_ref = self
    if not is_ref:
      name_ctx.layer = self
    self.is_ref = is_ref
    self.extra_dependencies = []  # type: List[Tensor]
    self.remove_unused_cleanup_hooks = []  # type: List[Callable[[nn.Tensor], None]]

  def __repr__(self):
    parts = [self.__class__.__name__, self.name_ctx.get_abs_name_repr()]
    if not hasattr(self, "data"):
      return f"<{' '.join(parts)} uninitialized>"
    if self.data:
      parts.append("[%s]" % ",".join(self.data.get_batch_axes_short_description()))
    if nn.is_debug_eager_mode_enabled():
      if self.data.placeholder is None:
        parts.append("<tf.Tensor: None>")
      else:
        parts.append(repr(self.data.placeholder))
    if not self.is_ref:
      parts.append(f"via {self.name_ctx.module if self.name_ctx.module else self.layer_dict.get('class', '?')!r}")
    return f"<{' '.join(parts)}>"

  @property
  def shape(self) -> Set[Dim]:
    """
    :return: shape (set of dims)
    """
    return self.data.dim_tags_set_implicit

  @property
  def shape_ordered(self) -> Tuple[Dim]:
    """
    :return: ordered dims.
      Note that usually the order should never matter.
      For some functions like nn.constant or nn.random_...,
      we currently need a specific order,
      and often we want to copy the order from some other tensor.
      This property shape_ordered is supposed to be used for such functions.
      Note that the rtype here could potentially change at some point
      to a ref-type which just indicates to reuse the same order of this tensor.
      So you should not rely on the rtype here
      and make any direct use of the returned value,
      except of passing it to functions like nn.constant.
      https://github.com/rwth-i6/returnn_common/issues/138
    """
    return self.data.dim_tags

  @property
  def dtype(self) -> str:
    """
    :return: data type (e.g. "float32")
    """
    return self.data.dtype

  @property
  def feature_dim(self) -> Optional[Dim]:
    """
    :return: feature dim
    """
    dim = self.data.feature_dim_or_sparse_dim
    if dim and dim.kind == Dim.Types.Feature:
      # Make sure it is unique.
      feature_dims = [dim_ for dim_ in self.data.dim_tags_set_implicit if dim_.kind == Dim.Types.Feature]
      if feature_dims == [dim]:
        return dim
    return None

  @property
  def batch_dim(self) -> Optional[Dim]:
    """
    :return: batch dim
    """
    if self.data.have_batch_axis():
      return self.data.dim_tags[self.data.batch_dim_axis]
    return None

  def verify_out_shape(self, out_shape: OutShapeType):
    """
    Verify out_shape via :func:`Data.verify_out_shape`.

    This does not add out_shape to the layer dict as we already have that automatically.
    Thus, this is purely for verification here on returnn-common side.

    :return: self, such that you can write this as a chained op
    :rtype: Tensor
    """
    self.data.verify_out_shape(out_shape)
    return self

  def _get_name_in_current_ctx(self) -> str:
    """
    :return: RETURNN layer name, valid in the current active name context.
    """
    return self._get_name_in_ctx(ctx=nn.NameCtx.current_ctx())

  def _assign_parent_name_ctx(self, *, ref_ctx: nn.NameCtx):
    """
    :param ref_ctx: where this comes from
    """
    assert not self.name_ctx.parent
    assert self.parent_modules  # cannot assign parent without parent modules
    #   (Although we could loosen this by checking some module from the stack trace of the __init__ call,
    #    when the actual name ctx parent is not so relevant.)
    sub_name = None
    for parent_module, attr in self.parent_modules:
      if getattr(parent_module, attr, None) is not self:
        continue  # might have been reset later...
      # This code could be extended by further heuristics.
      # The actual logic is not so important
      # as the final name_scope is always fixed in any case.
      # https://github.com/rwth-i6/returnn_common/issues/125
      if parent_module.calls:
        parent_name_ctx = parent_module.calls[0]
        sub_name = attr
        if self.require_global_access and not parent_name_ctx.can_access_children_from_root:
          sub_name = parent_name_ctx.name + "_" + sub_name
          while not parent_name_ctx.can_access_children_from_root:
            parent_name_ctx = parent_name_ctx.parent
        self.name_ctx.assign_parent(parent_name_ctx, sub_name)
        break
    if not self.name_ctx.parent:
      # None found. Just assign to the root.
      self.name_ctx.assign_parent(ref_ctx.root, sub_name or "unnamed_param")

  def _get_name_in_ctx(self, ctx: nn.NameCtx) -> str:
    """
    :return: RETURNN layer name in the given name context.
    """
    assert self.name_ctx.parent or ctx == self.name_ctx
    return self.name_ctx.get_name_in_ctx(ctx=ctx)

  def get_abs_name(self) -> str:
    """
    :return: absolute RETURNN layer name starting from root context.
    """
    return self.name_ctx.get_abs_name()

  def mark_as_loss(self, *,
                   scale: Optional[float] = 1.0,
                   as_error: bool = False,
                   use_normalized_loss: bool = False,
                   use_flatten_frames: bool = True,
                   custom_inv_norm_factor: Optional[nn.Tensor] = None,
                   ):
    """
    Mark this as a loss.
    This has the effect that it is specially handled by RETURNN.
    Specifically, the optimizer can use it in training,
    and it is used for reporting per batch or per epoch,
    and for learning rate scheduling.

    This currently uses :class:`AsIsLoss` in RETURNN
    but this is an implementation detail and might change.

    :param scale: scale the loss by this factor for the training optimizer
      (but not for any reporting). setting to 0.0 has the effect that this loss is not used by the optimizer.
    :param as_error: if True, this loss is reported as an error instead of a loss,
      and not used by the training optimizer.
      This is by convention sth like the frame-error or edit-distance, and usually not differentiable anyway.
    :param bool use_flatten_frames: If True, will use :func:`returnn.tf.util.basic.flatten_with_seq_len_mask`,
      i.e. a "packed" sequence with the padded frames removed, and accumulates over that.
      This can be more efficient, also because it will further optimize incoming computations
      and e.g. skip softmax computations right before on the padded frames.
      This can also avoid issues with inf/nan in some cases.
      If False, it will mask the loss to 0 in the padded frames and accumulate over that.
      Typically, setting this to True (default) is both more efficient and better.
    :param bool use_normalized_loss: the loss used in optimization will be normalized.
      E.g. if the overall normalization is sum(loss)/sum(num_frames), this is also what the optimizer will use,
      otherwise the optimizer will just use sum(loss).
    :param custom_inv_norm_factor:
      The standard norm factor is 1/sum(target_seq_len) if the target has a time-axis,
      or 1/sum(output_seq_len) if there is no target and the output has a time-axis,
      or 1 otherwise. (See :func:`Loss.init` for details.)
      This is used for proper normalization of accumulated loss/error per epoch
      and also proper normalization per batch for reporting,
      no matter if use_normalized_loss is True or False.
      If you want to change this norm factor, you can set this.
      As a function, it takes (self=self, output=output, layer=layer) and returns a float scalar.
      This here is the inverse of the norm factor.
      Here we also allow to pass any shape, and it will automatically be reduced via sum.
      So you could simply pass target_seq_len directly here.
      Basically, for all reporting, it uses sum(loss) * sum(custom_inv_norm_factor).
    """
    assert not self.is_ref, f"mark_as_loss can only be called on a layer, not a layer-ref {self}."
    assert "loss" not in self.layer_dict
    self.layer_dict["loss"] = "as_is"
    loss_opts = {}
    if scale is not None and scale != 1:
      assert "loss_scale" not in self.layer_dict
      loss_opts["scale"] = scale
    if as_error:
      loss_opts["as_error"] = True
    if use_normalized_loss:
      loss_opts["use_normalized_loss"] = True
    if not use_flatten_frames:
      loss_opts["use_flatten_frames"] = False
    if custom_inv_norm_factor is not None:
      loss_opts["custom_inv_norm_factor"] = custom_inv_norm_factor
    if loss_opts:
      self.layer_dict["loss_opts"] = loss_opts
    # Add it to the root name scope marked_losses list.
    # Note that this logic might change.
    scope = nn.NameCtx.current_ctx().root
    scope.marked_losses.append(self)

  def mark_as_output(self):
    """
    Mark this as an output.
    This has the effect that RETURNN will in any case construct the corresponding layer.
    Also see :func:`mark_as_default_output`.
    """
    assert not self.is_ref, f"mark_as_output can only be called on a layer, not a layer-ref {self}."
    scope = nn.NameCtx.current_ctx()
    assert scope.parent is None, f"{self} mark_as_output only makes sense at the top level"
    if self.name_ctx is scope.children.get("output"):
      pass  # not needed
    else:
      self.layer_dict["is_output_layer"] = True
    scope.marked_outputs.append(self)

  def mark_as_default_output(self) -> Tensor:
    """
    Mark this as the default output, i.e. create the "output" layer with a reference to this.
    This has the effect that RETURNN will in any case construct the corresponding layer,
    and it is the default output layer for forwarding and potential other tasks.

    :return: the "output" layer.
    """
    res = nn.NameCtx.current_ctx().make_default_output(self)
    res.mark_as_output()
    return res

  def get_dependencies(self, *, _extra_layer_dict=None) -> List[nn.Tensor]:
    """
    :return: list of tensors this tensor depends on
    """
    dep_list = []
    dep_name_set = set()

    def _maybe_add_dep(x):
      if isinstance(x, nn.Tensor):
        if x.name_ctx in dep_name_set:
          return
        dep_list.append(x)
        dep_name_set.add(x.name_ctx)
        return
      if isinstance(x, nn.Net):
        _maybe_add_dep(x.name_ctx.children["output"].layer_ref)

    if _extra_layer_dict:
      nest.map_structure(_maybe_add_dep, _extra_layer_dict)
    if hasattr(self, "layer_dict") and self.layer_dict:  # hasattr to be able to run this function early
      nest.map_structure(_maybe_add_dep, self.layer_dict)
    if self.name_ctx.children and "output" in self.name_ctx.children:
      _maybe_add_dep(self.name_ctx.children["output"].layer_ref)
    if self.name_ctx.parent and self.name_ctx.parent.layer_ref:
      _maybe_add_dep(self.name_ctx.parent.layer_ref)
    if getattr(self, "extra_dependencies", None):
      dep_list.extend(self.extra_dependencies)
    return dep_list

  def _replace_by(self, tensor: nn.Tensor):
    """
    Replace this tensor by the given tensor.
    This is a workaround in case other refs point to this tensor object.
    """
    assert isinstance(tensor, nn.Tensor)
    self.parent_modules = tensor.parent_modules
    self.name_ctx = tensor.name_ctx
    self.data = tensor.data
    self.layer_dict = tensor.layer_dict
    self.is_ref = tensor.is_ref
    self.extra_dependencies = tensor.extra_dependencies
    self.remove_unused_cleanup_hooks.clear()

  def _sis_hash(self):
    from sisyphus.hash import sis_hash_helper  # noqa
    if self.is_ref:
      return sis_hash_helper(self.name_ctx.get_abs_name())
    return sis_hash_helper(self.layer_dict)

  def __add__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 0:
      return self
    return nn.combine(self, other, kind="add", name="add")

  def __sub__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 0:
      return self
    return nn.combine(self, other, kind="sub", name="sub")

  def __mul__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 1:
      return self
    return nn.combine(self, other, kind="mul", name="mul")

  def __truediv__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 1:
      return self
    return nn.combine(self, other, kind="truediv", name="truediv")

  def __floordiv__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 1:
      return self
    return nn.combine(self, other, kind="floordiv", name="floordiv")

  def __mod__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    return nn.combine(self, other, kind="mod", name="mod")

  def __radd__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 0:
      return self
    return nn.combine(other, self, kind="add", name="add")

  def __rsub__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 0:
      return self
    return nn.combine(other, self, kind="sub", name="sub")

  def __rmul__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 1:
      return self
    return nn.combine(other, self, kind="mul", name="mul")

  def __rtruediv__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 1:
      return self
    return nn.combine(other, self, kind="truediv", name="truediv")

  def __rfloordiv__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, (int, float, numpy.number)) and other == 1:
      return self
    return nn.combine(other, self, kind="floordiv", name="floordiv")

  def __rmod__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    return nn.combine(other, self, kind="mod", name="mod")

  def __neg__(self) -> Tensor:
    return nn.neg(self)

  def __invert__(self) -> Tensor:
    return nn.logical_not(self)

  def __pow__(self, other: Union[RawTensorTypes, Tensor], modulo=None) -> Tensor:
    assert modulo is None
    if isinstance(other, (int, float, numpy.number)) and other == 1:
      return self
    return nn.combine(self, other, kind="pow", name="pow")

  def __rpow__(self, other: Union[RawTensorTypes, Tensor], modulo=None) -> Tensor:
    assert modulo is None
    return nn.combine(other, self, kind="pow", name="pow")

  def __and__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, bool) and other is True:
      return self
    if isinstance(other, bool) and other is False:
      return nn.zeros_like(self)
    return nn.combine(self, other, kind="logical_and", name="logical_and")

  def __or__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    if isinstance(other, bool) and other is True:
      return nn.ones_like(self)
    if isinstance(other, bool) and other is False:
      return self
    return nn.combine(self, other, kind="logical_or", name="logical_or")

  def __abs__(self) -> Tensor:
    return nn.abs(self)

  def __ceil__(self) -> Tensor:
    return nn.ceil(self)

  def __floor__(self) -> Tensor:
    return nn.floor(self)

  def __eq__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    return nn.compare(self, nn.convert_to_tensor(other), kind="equal")

  def __ne__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    return nn.compare(self, nn.convert_to_tensor(other), kind="not_equal")

  def __lt__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    return nn.compare(self, nn.convert_to_tensor(other), kind="less")

  def __le__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    return nn.compare(self, nn.convert_to_tensor(other), kind="less_equal")

  def __gt__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    return nn.compare(self, nn.convert_to_tensor(other), kind="greater")

  def __ge__(self, other: Union[RawTensorTypes, Tensor]) -> Tensor:
    return nn.compare(self, nn.convert_to_tensor(other), kind="greater_equal")


class Parameter(Tensor):
  """
  This represents a (potential trainable) parameter,
  aka ``tf.Variable`` in TensorFlow,
  wrapping to ``VariableLayer`` in RETURNN.
  """
  require_global_access = True

  def __init__(self, shape: Sequence[Dim], dtype: Optional[str] = None,
               *,
               trainable: Optional[bool] = None,
               auxiliary: bool = False):
    """
    :param shape:
    :param dtype:
    :param trainable: if True, and optimizer would do updates to this parameter in training mode
    :param auxiliary: if True, this indicates that this parameter should not be transformed by transformations
      such as weight normalization. One example are running statistics, as used for batch normalization.
      This usually implies that the parameter is not trainable, i.e. not to be updated by the optimizer,
      but usually has some custom update.
      This flag is not passed on to RETURNN but just used here for returnn-common logic.
    """
    if not all(isinstance(dim, Dim) for dim in shape):
      raise TypeError(f"shape {shape} must be a sequence of Dim")
    if not all(isinstance(dim.dimension, int) for dim in shape):
      raise ValueError(f"shape {shape} must be static")
    if len(shape) != len(set((d, d.match_priority) for d in shape)):
      raise ValueError(f"shape {shape} dims must be unique")
    # Note: At creation time, we don't know the name yet.
    # The name will be inferred by the parent modules and the attribute chain.
    # The name_ctx object will be completed by this information later.
    # See Tensor.get_name_in_ctx().
    name_ctx = nn.NameCtx(name="parameter", parent=None)
    data = Data("parameter", dim_tags=list(shape), dtype=dtype)
    layer_dict = {"class": "variable", "shape": list(shape), "param_name": "param"}
    if dtype is not None:
      layer_dict["dtype"] = dtype
    if auxiliary and trainable is None:
      trainable = False
    if trainable is not None:
      layer_dict["trainable"] = trainable
    super(Parameter, self).__init__(
      layer_dict=layer_dict,
      data=data,
      name_ctx=name_ctx)
    self.auxiliary = auxiliary

  @property
  def initial(self) -> Optional[Union[nn.Tensor, RawTensorTypes]]:
    """initial value of the parameter"""
    if "init" in self.layer_dict:
      return self.layer_dict["init"]
    return self.layer_dict.get("init_by_layer")

  @initial.setter
  def initial(self, value: Optional[Union[nn.Tensor, RawTensorTypes, nn.init.VarianceScaling]]):
    if isinstance(value, nn.init.VarianceScaling):
      value = value(self.shape_ordered)
    if value is None or isinstance(value, nn.Tensor):
      self.layer_dict.pop("init", None)
      if isinstance(value, nn.Tensor) and not value.name_ctx.parent.can_access_children_from_root:
        accessible_parent = value.name_ctx.parent
        while not accessible_parent.can_access_children_from_root:
          accessible_parent = accessible_parent.parent
        value.name_ctx.assign_parent(accessible_parent)
        # We could also maybe move out all the dependencies. However, it's not clear whether this is always safe.
        for dep in value.get_dependencies():
          assert dep.name_ctx.parent.can_access_children_from_root, (
            f"dep {dep} of moved value {value} is not accessible")
      self.layer_dict["init_by_layer"] = value
    else:
      self.layer_dict.pop("init_by_layer", None)
      self.layer_dict["init"] = value
    if nn.is_debug_eager_mode_enabled():
      shape = [d.get_dim_value() for d in self.shape_ordered]
      if isinstance(value, nn.Tensor):
        assert value.data.placeholder is not None
        value_tf = value.data.placeholder
      else:
        value_tf = tf.broadcast_to(tf.convert_to_tensor(value), shape)
      if self.data.placeholder is None:
        var = tf.Variable(
          value_tf, shape=[d.get_dim_value() for d in self.shape_ordered], dtype=self.data.dtype)
        self.data.placeholder = var
      else:
        var = self.data.placeholder
        assert isinstance(var, tf.Variable)
        var.assign(value_tf)

  @property
  def weight_decay(self) -> float:
    """
    Weight decay, which is equivalent to L2 loss on the parameters for SGD.
    On RETURNN side, whether this is handled separately or is part of the main loss,
    can be controlled via the ``decouple_constraints`` config option.
    https://github.com/rwth-i6/returnn_common/issues/59#issuecomment-1073913421
    """
    return self.layer_dict.get("L2", 0.0)

  @weight_decay.setter
  def weight_decay(self, value: Optional[float]):
    if value:
      self.layer_dict["L2"] = value
    else:
      self.layer_dict.pop("L2", None)


class LayerState(dict):
  """
  Covers all the state of a layer,
  i.e. exactly what needs to be stored and passed into the module or module
  next time you call it as initial state.

  This behaves somewhat like a namedtuple, although we derive from dict.
  """
  def __init__(self, *args, **kwargs):
    if kwargs:
      assert not args
      super().__init__(**kwargs)
    elif args:
      assert len(args) == 1
      if isinstance(args[0], dict):
        super().__init__(**args[0])
      else:
        super().__init__(state=args[0])
    else:
      super().__init__()

  def __repr__(self):
    return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for (k, v) in self.items())})"

  def __getattr__(self, item):
    if item in self:
      return self[item]
    raise AttributeError(f"{self}.{item}")

  def __setattr__(self, key, value):
    self[key] = value


def make_layer(layer_dict: LayerDictRaw, *,
               name: Optional[Union[str, nn.NameCtx]] = None,
               module: Optional[nn.Module] = None,
               predefined_out_data: Optional[Data] = None,
               ) -> Tensor:
  """
  Creates the layer. This also registers the layer instance in the top name ctx.
  When no name is given, this assumes that the top name ctx corresponds to this module.

  If a layer has params, and you want the param sharing logic,
  you should instead derive a new class from :class:`Module`.
  Usually, you do not need either of these,
  as all standard layers should already be wrapped,
  and it should be possible to define any possible logic
  using that.
  (If this is not the case, please report an issue.)

  :param LayerDictRaw layer_dict: can contain :class:`Tensor` instances
  :param str|NameCtx|None name:
    if str: (suggested) layer name. if given, will create a new :class:`NameCtx`
    if NameCtx, will use this.
  :param Module|None module: if given, will create new name scope with this module
  :param Data|None predefined_out_data: normally we can derive the out data automatically.
    If this should be skipped, you can pass this explicitly.
  """
  if isinstance(name, str) or module:
    assert not name or isinstance(name, str)
    name_ctx = nn.NameCtx.get_from_call(module=module, name=name)
    created_name_ctx = True
  elif isinstance(name, nn.NameCtx):
    name_ctx = name
    created_name_ctx = False
  else:
    raise TypeError(f"name must be str or NameCtx, not {type(name)}; or you should pass a module")
  assert not name_ctx.layer_ref and not name_ctx.layer  # not yet assigned
  layer_dict = layer_dict.copy()

  name_ctx.is_subnet_ctx = False
  try:
    layer = Tensor(
      layer_dict=layer_dict, name_ctx=name_ctx,
      data=predefined_out_data)
  except Exception as exc:
    # Just forward the exception.
    # However, if we already created a new name_ctx for it, we can clean this up now.
    if created_name_ctx:
      assert name_ctx.parent
      name_ctx.parent.children.pop(name_ctx.name)
    raise exc
  if name_ctx.module:
    name_ctx.module.calls.append(name_ctx)
  return layer


def get_extern_data(data: Data) -> Tensor:
  """
  Get extern data from root ctx.
  As a side effect, it registers the given data as extern data,
  and this will be included when creating the RETURNN config,
  via :func:`NameCtx.get_returnn_config`.
  """
  assert isinstance(data, Data)  # the usage was different before. make sure we get this correct
  scope = nn.NameCtx.top()  # must exist
  assert not scope.parent  # get_extern_data only allowed (only makes sense) in root name ctx
  if data.name not in scope.extern_data:
    scope.extern_data[data.name] = data
  else:
    assert scope.extern_data[data.name] is data
  if data.have_batch_axis():
    if not scope.global_batch:
      if data.batch:
        scope.global_batch = data.batch
      elif nn.is_debug_eager_mode_enabled():
        scope.global_batch = nn.BatchInfo.make_global_batch_info(
          tf.constant(3, name="global_batch"))  # https://xkcd.com/221/, but prime
      else:
        scope.global_batch = nn.BatchInfo.make_global_batch_info(-1)
    if not data.batch:
      data.batch = scope.global_batch
  root_layer_name = f"data:{data.name}"
  out = _get_raw_layer_by_name(root_layer_name, scope=scope, data=data)
  for tag in data.dim_tags:
    if not tag.is_batch_dim() and tag.dimension is None and not tag.dyn_size_ext:
      # Undefined dynamic dim tag. Set default data template.
      tag.dyn_size_ext = Data(
        name=f"{data.name}_default_dyn_size_ext", dim_tags=[nn.batch_dim], dtype=data.size_dtype, batch=data.batch)
  if nn.is_debug_eager_mode_enabled():
    out.data.placeholder = _make_random_tf_tensor_for_returnn_data(out.data)
  return out


def _make_random_tf_tensor_for_returnn_data(data: Data) -> tf.Tensor:
  shape = []
  for dim in data.dim_tags:
    if dim.is_batch_dim():
      assert data.batch
      shape.append(data.batch.dim)
    elif dim.dimension is not None:
      shape.append(dim.dimension)
    else:
      dim.complete_dyn_size()
      if dim.dyn_size_ext is None:
        assert data.batch
        dim.dyn_size_ext = Data(
          name=f"{data.name}_dummy_dyn_size_ext", dim_tags=[nn.batch_dim], dtype=data.size_dtype, batch=data.batch)
      if dim.dyn_size_ext.placeholder is None:
        dim.dyn_size_ext.placeholder = _make_random_tf_tensor_for_returnn_data(dim.dyn_size_ext)
      shape.append(tf.reduce_max(dim.dyn_size_ext.placeholder))
  dtype = tf.as_dtype(data.dtype)
  if dtype.is_integer:
    if data.sparse:
      return tf.random.uniform(shape=shape, dtype=dtype, minval=0, maxval=data.dim)
    else:
      import binascii
      c = abs(binascii.crc32(data.name.encode("utf8"))) % 21 + 3
      shape = tf.convert_to_tensor(shape)
      c_tf = tf.constant(c, name="dummy_random_const", dtype=dtype)
      rnd = tf.broadcast_to(c_tf, shape)
      rnd_diff = tf.random.uniform(shape=shape, minval=0, maxval=2 ** 31 - 1, dtype=dtype)
      rnd_diff = rnd_diff % tf.reshape(tf.minimum(tf.range(0, tf.size(rnd), dtype=dtype) + 1, c_tf - 2), shape)
      rnd = tf.clip_by_value(rnd - rnd_diff, 1, c_tf)
      return rnd
  assert dtype.is_floating  # not implemented otherwise
  return tf.random.normal(shape=shape, dtype=dtype)


def _get_raw_layer_by_name(name: str, *, scope: Optional[nn.NameCtx] = None, data: Data) -> Tensor:
  """
  Special layer can be "data:..." or whatever.
  """
  if not scope:
    scope = nn.NameCtx.current_ctx()  # must exist
  return scope.get_child_layer_ref(name, data=data)


def _get_sub_layer(layer: Tensor, name: str, *, data: Data) -> Tensor:
  """
  Like the "{layer}/{name}" syntax in RETURNN.
  Normally this should only be needed for internal usage.
  """
  out = layer.name_ctx.get_child_layer_ref(name, data=data)
  if nn.is_debug_eager_mode_enabled():
    assert layer.debug_layer
    import returnn.tf.layers.base
    assert isinstance(layer.debug_layer, returnn.tf.layers.base.LayerBase)
    sub_layer = layer.debug_layer.get_sub_layer(name)
    assert sub_layer and sub_layer.output.dim_tags == out.data.dim_tags
    out.debug_layer = sub_layer
    out.data = sub_layer.output
  return out


class ReturnnConstructTemplateException(Exception):
  """
  In :func:`_data_from_layer_dict`, when we call layer_class.get_out_data_from_opts,
  we potentially can get errors, often due to user mistakes.
  We wrap those errors in this exception for better reporting.
  """


def _data_from_layer_dict(layer_dict: LayerDictRaw, *, tensor: Tensor) -> Data:
  """
  Use RETURNN layer_class.get_out_data_from_opts to get the :class:`Data`.
  For this function, we need to set up some dummy network and dummy source layers.
  """
  from returnn.tf.network import TFNetwork, ExternData
  from returnn.tf.layers.base import InternalLayer, LayerBase
  from returnn.util import BehaviorVersion
  from returnn.config import Config
  config = Config({
    "behavior_version": min_returnn_behavior_version,
  })
  BehaviorVersion.set(min_returnn_behavior_version)
  loop = nn.NameCtx.inner_loop()  # Note: for control_flow_ctx, we should also check Cond
  net = TFNetwork(
    config=config, extern_data=ExternData(), name="dummy_net",
    train_flag=True,  # should not have an effect usually for templates, except maybe in debug-eager-mode
    inside_rec_time_dim=loop.axis if loop else None,
    control_flow_ctx=nn.NameCtx.inner_control_flow())

  ref_to_layer_name = {}  # type: Dict[nn.NameCtx, str]

  def _get_unique_name(name) -> str:
    reserved_names = set(net.layers.keys()) | {"data"}
    if name not in reserved_names:
      return name
    i = 0
    while True:
      name_ = f"{name}_{i}"
      if name_ not in reserved_names:
        return name_
      i += 1

  def _get_layer_name(ref: Tensor) -> str:
    if ref.name_ctx in ref_to_layer_name:
      return ref_to_layer_name[ref.name_ctx]
    name = _get_unique_name(ref.name_ctx.name)
    ref_to_layer_name[ref.name_ctx] = name
    assert name not in net.layers
    data = ref.data.copy()
    net.layers[name] = InternalLayer(name=name, network=net, output=data)
    return name

  def _map_layer_dict_elem(value):
    if isinstance(value, Tensor):
      return _get_layer_name(value)
    return value

  layer_dict = nest.map_structure(_map_layer_dict_elem, layer_dict)
  out_name = _get_unique_name(tensor.name_ctx.name)
  net_dict = {out_name: layer_dict}

  if nn.is_debug_eager_mode_enabled():
    _add_layer = None  # implies to really construct the layer
  else:
    # Creates only a template layer.
    def _add_layer(name: str, layer_class: Type[LayerBase], **layer_desc) -> LayerBase:
      # noinspection PyProtectedMember
      layer_desc = net._create_layer_layer_desc(name=out_name, layer_desc=layer_desc, template=True)
      try:
        out_data = layer_class.get_out_data_from_opts(**layer_desc)
      except Exception as exc:
        msgs = [
          "The RETURNN call\n",
          f"  {layer_class.__name__}.get_out_data_from_opts(\n"]
        for key, v in layer_desc.items():
          msgs.append(f"    {key}={v!r},\n")
        msgs += [
          "  )\n",
          "raised the exception:\n",
          f"  {type(exc).__name__} {exc!s}\n",
          "(See above for the RETURNN exception traceback.)"]
        raise ReturnnConstructTemplateException("".join(msgs)) from exc
      return InternalLayer(name=name, network=net, output=out_data)

  # Use construct_layer to automatically handle more complex logic such as subnetworks.
  layer = net.construct_layer(net_dict=net_dict, name=out_name, add_layer=_add_layer)

  if nn.is_debug_eager_mode_enabled():
    tensor.debug_layer = layer

  return layer.output
