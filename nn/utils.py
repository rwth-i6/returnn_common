"""
Some generic utils (which doesn't fit into math_, array_, etc)
"""

from typing import Optional, Union, Sequence, Tuple, Callable, Any
from .. import nn


def convert_to_tensor(x: Union[nn.Tensor, int, float, complex, bool, str]) -> nn.Tensor:
  """
  In case it is not a :class:`Tensor` yet, it will make some constant.
  """
  if isinstance(x, nn.Tensor):
    return x
  return nn.constant(value=x)


def constant_value(x: nn.Tensor) -> Optional[Union[int, float, complex, bool, str]]:
  """
  If the tensor is a constant, return its value.
  """
  if x.layer_dict and x.layer_dict["class"] == "constant":
    return x.layer_dict["value"]
  return None


def where(cond: nn.Tensor,
          true_: Union[nn.Tensor, float, int],
          false_: Union[nn.Tensor, float, int],
          *, name: Optional[str] = None) -> nn.Tensor:
  """
  Wraps tf.where, which is SwitchLayer in RETURNN.

  :return: true_ if cond else false_, elemwise.
  """
  return nn.make_layer({
    'class': 'switch', "condition": cond, "true_from": true_, "false_from": false_}, name=name or 'where')


# noinspection PyShadowingNames
def dropout(source: nn.Tensor,
            dropout: float,
            *,
            axis: Union[nn.Dim, Sequence[nn.Dim]],
            on_forward: bool = False,
            name: Optional[str] = None
            ) -> nn.Tensor:
  """
  Applies dropout.

  Dropout will only be applied during training (unless you set on_forward=True).

  When dropout is applied, the output will be scaled by 1/dropout.

  :param nn.Tensor source:
  :param float dropout: 0.0 means to apply no dropout. 100% would mask everything.
    For every value in the tensor, the probability of it being dropped is drawn independently given this probability.
    The broadcasted axes are those not specified in ``axis``.
  :param axis: axis to apply dropout on. multiple axes can be specified.
    This defines the set of axes where the dropout mask is not broadcasted to.
    (RETURNN also has the ``noise_shape`` option but the ``axis`` option provides the same functionality.)
  :param bool on_forward: apply dropout during inference
  :param str|None name:
  """
  assert isinstance(source, nn.Tensor)
  if not dropout:
    return source
  opts = {"dropout": dropout, "dropout_axis": axis}
  if on_forward:
    opts["dropout_on_forward"] = True
  from .base import make_layer
  return make_layer(
    {"class": "dropout", "from": source, **opts},
    name=name or "dropout", name_ctx_ignore_top_stack_frames=1)


def stop_gradient(source: nn.Tensor, name: Optional[str] = None) -> nn.Tensor:
  """wraps tf.stop_gradient"""
  return nn.scaled_gradient(source, scale=0, name=name)


# noinspection PyShadowingBuiltins
def top_k(source: nn.Tensor,
          *,
          axis: Union[nn.Dim, Sequence[nn.Dim]],
          k: Union[int, nn.Tensor],
          k_dim: Optional[nn.Dim] = None,
          sorted: bool = True,
          name: Optional[str] = None
          ) -> Tuple[nn.Tensor, Union[nn.Tensor, Sequence[nn.Tensor]], nn.Dim]:
  """
  Basically wraps tf.nn.top_k.

  Directly returns the top_k values.
  The indices are accessible via the "indices" sub-layer.

  For an input [B,D] with axis=D, the output and indices values are shape [B,K].

  It's somewhat similar to :class:`ReduceLayer` with max and argmax.
  The axis dim is reduced and then a new dim for K is added.

  Axis can also cover multiple axes, such as [beam,classes].
  In that cases, there is not a single "indices" sub-layer,
  but sub-layers "indices0" .. "indices{N-1}"
  corresponding to each axis, in the same order.

  All other axes are treated as batch dims.

  :param source:
  :param axis: the axis to do the top_k on, which is reduced, or a sequence of axes
  :param k: the "K" in "TopK"
  :param k_dim: the new axis dim for K. if not provided, will be automatically created.
  :param sorted:
  :param name:
  :return: values, indices (multiple if axis is a sequence), k_dim
  """
  from ._generated_layers import _top_k
  from .base import _get_sub_layer
  values, k_dim = _top_k(source, axis=axis, k=k, k_dim=k_dim, sorted=sorted, name=name)
  if isinstance(axis, (tuple, list)):
    axes = axis
    single_axis = False
  else:
    assert isinstance(axis, nn.Dim)
    axes = [axis]
    single_axis = True
  indices = []
  for i, a in enumerate(axes):
    assert isinstance(a, nn.Dim)
    sub_name = "indices" if single_axis else f"indices{i}"
    indices_data = values.data.copy_template(name=f"{values.data.name}_{sub_name}_{a.description}")
    indices_data.dtype = "int32"
    indices_data.sparse_dim = a
    indices.append(_get_sub_layer(values, sub_name, data=indices_data))
  if single_axis:
    indices = indices[0]
  return values, indices, k_dim


def reinterpret_new_dim(source: nn.Tensor, *, in_dim: nn.Dim, out_dim: Optional[nn.Dim] = None,
                        name: Optional[str] = None) -> Tuple[nn.Tensor, nn.Dim]:
  """
  :return: source with in_dim replaced by out_dim.
    this does not work for the sparse_dim. see :func:`reinterpret_set_sparse_dim` for that case.
  """
  if not out_dim:
    out_dim = in_dim.copy(same_as_self=False, description="new-dim")
  out = nn.make_layer(
    {"class": "reinterpret_data", "set_dim_tags": {in_dim: out_dim}, "from": source},
    name=name or "new_dim")
  return out, out_dim


def reinterpret_set_sparse_dim(source: nn.Tensor, out_dim: nn.Dim, *, name: str = "set_sparse_dim") -> nn.Tensor:
  """
  :return: source with sparse_dim set to out_dim
  """
  return nn.make_layer(
    {"class": "reinterpret_data", "set_sparse_dim": out_dim, "from": source},
    name=name)


def check_in_feature_dim_lazy_init(
      source: nn.Tensor, in_dim: Optional[nn.Dim], mod_in_dim: Optional[nn.Dim],
      lazy_init: Callable[[nn.Dim], Any]) -> nn.Tensor:
  """
  This is a helper function for modules which want to lazily support assigning the in_dim.
  """
  if mod_in_dim:
    if in_dim:
      if in_dim != mod_in_dim:
        raise ValueError(f"in_dim {in_dim} does not match module in_dim {mod_in_dim}")
    if mod_in_dim in source.shape:
      return source  # all fine
    raise ValueError(f"{source} does not have feature dim {mod_in_dim}")
  # Not yet initialized.
  if in_dim:
    if in_dim not in source.shape:
      raise ValueError(f"invalid in_dim {in_dim} for {source}")
    lazy_init(in_dim)
    return source
  if not source.feature_dim:
    raise ValueError(f"{source} has no feature dim. define the in_dim explicitly")
  lazy_init(source.feature_dim)
  return source


def dim_match_priority_when_needed(dim: nn.Dim, *other_dims: nn.Dim) -> nn.Dim:
  """
  :return: maybe copy of dim with higher match_priority if needed to distinguish from other_dims
  """
  if dim in other_dims:
    return dim.copy(match_priority=1)
  return dim


def range_over_dim(dim: nn.Dim,
                   *,
                   dtype: str = nn.NotSpecified,
                   sparse: bool = False,
                   ) -> nn.Tensor:
  """
  Creates a tensor with shape [dim] with values 0,1,2,...,dim-1.
  In RETURNN, this is the range_in_axis layer.

  :param nn.Dim dim:
  :param str dtype: default is int32
  :param bool sparse:
  :return: layer
  """
  args = {}
  if sparse:
    args["sparse"] = True
  if dtype is not nn.NotSpecified:
    args["dtype"] = dtype
  return nn.make_layer({
    'class': 'range_in_axis',
    'from': nn.get_dim_deps(dim),
    'axis': dim,
    **args}, name='range_over_dim')


def sparse_to_dense(source: nn.Tensor, *,
                    label_value: Union[nn.Tensor, int, float],
                    other_value: Union[nn.Tensor, int, float]) -> nn.Tensor:
  """
  Converts a sparse tensor to a dense one.

  This is a more generic variant of "one_hot".

  Note that usually this is not needed as most other functions should handle sparse tensors just fine
  and much more efficiently than they would be with dense tensors.
  """
  assert source.data.sparse
  axis = source.data.sparse_dim
  indices = range_over_dim(axis, sparse=True)
  return nn.where(source == indices, label_value, other_value)


def one_hot(source: nn.Tensor) -> nn.Tensor:
  """
  one_hot. special case of :func:`sparse_to_dense`.

  Note that usually this is not needed as most other functions should handle sparse tensors just fine
  and much more efficiently than they would be with dense tensors.
  """
  return sparse_to_dense(source, label_value=1., other_value=0.)


def smooth_one_hot(source: nn.Tensor, *, label_prob: Union[nn.Tensor, float]) -> nn.Tensor:
  """
  Smooth variant of :func:`one_hot`.
  Uses ``label_prob`` for the labels and ``(1 - label_prob) / (dim - 1)`` for the remaining values.
  This is used for label smoothing.
  """
  assert source.data.sparse
  if source.data.sparse_dim.dimension is None:
    raise NotImplementedError(f"smooth_one_hot({source}) not implemented for dynamic dims")
  return sparse_to_dense(
    source, label_value=label_prob, other_value=(1. - label_prob) / (source.data.sparse_dim.dimension - 1))


def label_smoothing(source: nn.Tensor, smoothing: Union[nn.Tensor, float],
                    *, axis: Optional[nn.Dim] = None) -> nn.Tensor:
  """
  Label smoothing, often used for cross entropy.

  In case of sparse data, it will become dense (via :func:`smooth_one_hot`)
  and the target label will get probability (1 - smoothing).
  """
  if not axis:
    assert source.feature_dim
    axis = source.feature_dim
  if source.data.sparse:
    assert source.data.sparse_dim == axis
    return smooth_one_hot(source, label_prob=1. - smoothing)
  else:
    assert axis in source.shape
    # Make it consistent to the sparse case.
    # Value of 1.0 should result in (1 - smoothing).
    # Value of 0.0 should result in smoothing / (dim - 1).
    # Sum over all should still remain 1.0.
    dim = source.data.sparse_dim.dimension
    floor_prob = smoothing / (dim - 1)
    factor = 1. - dim * floor_prob
    return source * factor + floor_prob


def stochastic_depth(
      func: Callable[[], nn.Tensor],
      p: float, mode: str = "batch") -> nn.Tensor:
  """
  Implements Stochastic Depth (sometimes also called "layer drop")
  for randomly dropping residual branches of residual architectures.

  Code adopted from here: https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py

  Only applied when in training.

  For some further discussion, also see: https://github.com/rwth-i6/returnn_common/issues/99
  Relevant papers:
  - `"Deep Networks with Stochastic Depth" <https://arxiv.org/abs/1603.09382>`__
  - `"Very Deep Self-Attention Networks for End-to-End Speech Recognition" <https://arxiv.org/abs/1904.13377>`__
  - `"Reducing Transformer Depth on Demand with Structured Dropout" <https://arxiv.org/abs/1909.11556>`__
  - `"Intermediate Loss Regularization for CTC-based Speech Recognition" <https://arxiv.org/abs/1904.09751>`__
  - `"Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" <https://arxiv.org/abs/2103.14030>`__

  Args:
      func (() -> Tensor[...]): Module or function for input tensor or arbitrary dimensions
      p (float): probability of the input to be zeroed.
      mode (str): ``"batch"`` or ``"row"``.
                  ``"batch"`` randomly zeroes the entire input and performs the computation only when necessary,
                  ``"row"`` zeroes randomly selected rows (batch indices) from the batch.
  Returns:
      Tensor[...]: The randomly zeroed tensor.
  """
  if p < 0.0 or p > 1.0:
    raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
  if mode not in ["batch", "row"]:
    raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
  if p == 0.0:
    return func()

  training = nn.train_flag()

  survival_rate = 1.0 - p
  if mode == "row":
    true_value = func()
    with nn.Cond(training) as cond:
      # Not efficient.
      noise = nn.random_bernoulli([true_value.batch_dim], p=survival_rate)
      if survival_rate > 0.0:
        noise /= survival_rate
      cond.true = true_value * noise
      cond.false = true_value
    return cond.result
  elif mode == "batch":
    with nn.Cond(training) as cond_train:
      noise = nn.random_bernoulli((), p=survival_rate)
      with nn.Cond(noise) as cond_noise:
        true_value = func()
        if survival_rate > 0.0:
          true_value /= survival_rate
        cond_noise.true = true_value
        cond_noise.false = nn.zeros_like(true_value)
      cond_train.true = cond_noise.result
      cond_train.false = func()
    return cond_train.result
  else:
    raise ValueError(f"mode {mode!r} invalid")


def gather_by_mask(x: nn.Tensor, *,
                   mask: nn.Tensor,
                   in_spatial_dim: nn.Dim,
                   name: str = "masked"
                   ) -> Tuple[nn.Tensor, nn.Dim]:
  """
  Like tf.boolean_mask.

  :param x: apply mask to this tensor. for example [B,T,D]
  :param mask: boolean tensor. has a subset of the same dims as x. for example [B,T]
  :param in_spatial_dim: dim to mask/compress
  :param name:
  :return: (masked_tensor, out_spatial_dim), for example [B,T',D], where T' is potentially shorter than T.
  """
  out_spatial_dim = nn.Dim(description=(in_spatial_dim.description or "unknown") + ":masked", kind=in_spatial_dim.kind)
  return nn.make_layer({
    "class": "masked_computation", "from": x, "mask": mask,
    "in_spatial_dim": in_spatial_dim, "out_spatial_dim": out_spatial_dim,
    "unit": {"class": "copy", "from": "data"}}, name=name), out_spatial_dim


def ctc_greedy_decode(logits: nn.Tensor, *,
                      in_spatial_dim: nn.Dim,
                      feature_dim: Optional[nn.Dim] = None,
                      blank_index: int = -1
                      ) -> Tuple[nn.Tensor, nn.Dim]:
  """
  Also see :func:`nn.ctc_loss`.

  :param logits: non-normalized (or actually does not matter, as we will take argmax). for example [B,T,D]
  :param in_spatial_dim:
  :param feature_dim:
  :param blank_index:
  :return: (greedy_decoded, out_spatial_dim). for example [B,T'] -> D.
  """
  if feature_dim is None:
    feature_dim = logits.feature_dim
  if blank_index < 0:
    blank_index += feature_dim.dimension
  assert 0 <= blank_index < feature_dim.dimension
  argmax = nn.reduce(logits, axis=feature_dim, mode="argmax")
  shift_right = nn.shift_axis(argmax, axis=in_spatial_dim, amount=1, pad_value=-1, adjust_size_info=False)
  unique_mask = argmax != shift_right
  non_blank_mask = argmax != blank_index
  mask = unique_mask & non_blank_mask
  decoded, out_spatial_dim = nn.gather_by_mask(argmax, mask=mask, in_spatial_dim=in_spatial_dim)
  decoded_sparse_dim = feature_dim.sub_left(1) if blank_index == 0 else feature_dim - 1
  decoded = nn.reinterpret_set_sparse_dim(decoded, decoded_sparse_dim)
  return decoded, out_spatial_dim


def prev_target_seq(targets: nn.Tensor, *, spatial_dim: nn.Dim, bos_idx: int) -> nn.Tensor:
  """
  shift by one
  """
  y, dim_ = nn.slice(targets, axis=spatial_dim, slice_end=-1)
  pad_dim = nn.SpatialDim("dummy", 1)
  pad_value = nn.constant(value=bos_idx, shape=[pad_dim], dtype=targets.dtype, sparse_dim=targets.feature_dim)
  y = nn.concat((pad_value, pad_dim), (y, dim_), allow_broadcast=True)
  dim_ = pad_dim + dim_
  y, _ = nn.reinterpret_new_dim(y, in_dim=dim_, out_dim=spatial_dim)
  return y
