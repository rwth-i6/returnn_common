
"""
SpecAugment.
"""

from typing import Union, Collection
from .. import nn


@nn.scoped
def specaugment_v2(x: nn.Tensor, *,
                   spatial_dim: nn.Dim,
                   feature_dim: nn.Dim = nn.NotSpecified,
                   global_train_step_dependent: bool = True,
                   only_on_train: bool = True,
                   ) -> nn.Tensor:
  """
  SpecAugment reimplementation of :func:`specaugment_v1`
  """
  if feature_dim is nn.NotSpecified:
    assert x.feature_dim
    feature_dim = x.feature_dim
  if global_train_step_dependent:
    step = nn.global_train_step()
    step1 = nn.where(step >= 1000, 1, 0)
    step2 = nn.where(step >= 2000, 1, 0)
  else:
    step1 = step2 = 1
  time_factor = 1

  with nn.Cond(nn.train_flag() | (not only_on_train)) as cond:
    x_masked = x
    spatial_len = nn.dim_value(x, axis=spatial_dim)
    x_masked = random_mask_v2(
      x_masked, mask_axis=spatial_dim, broadcast_axis=feature_dim, name="time_masking",
      min_num=nn.minimum(step1 + step2, spatial_len),
      max_num=nn.minimum(nn.maximum(spatial_len // 100, 2) * (1 + step1 + step2 * 2), spatial_len),
      max_dims=20 // time_factor)
    x_masked = random_mask_v2(
      x_masked, mask_axis=feature_dim, broadcast_axis=spatial_dim, name="feature_masking",
      min_num=step1 + step2, max_num=2 + step1 + step2 * 2,
      max_dims=feature_dim.dimension // 5)
    cond.true = x_masked
    cond.false = x
  return cond.result


@nn.scoped
def random_mask_v2(x: nn.Tensor, *,
                   mask_axis: nn.Dim,
                   broadcast_axis: Union[nn.Dim, Collection[nn.Dim]],
                   min_num: Union[int, nn.Tensor],
                   max_num: Union[int, nn.Tensor],
                   max_dims: Union[int, nn.Tensor],
                   mask_value: Union[int, float] = 0.
                   ) -> nn.Tensor:
  """
  :param x: (batch,time,feature)
  :param mask_axis: axis to mask
  :param broadcast_axis: one or multiple, which should be broadcasted over.
    The remaining axes not specified by mask_axis and broadcast_axis are not broadcasted over
    and treated as batch dims.
    E.g. in [B,T,D], with mask_axis=F, broadcast_axis=T, it creates masks [B,F].
  :param min_num:
  :param max_num: inclusive
  :param max_dims: inclusive
  :param mask_value:
  :rtype: tf.Tensor
  """
  batch_dims = list(x.shape_ordered)
  batch_dims.remove(mask_axis)
  if isinstance(broadcast_axis, nn.Dim):
    batch_dims.remove(broadcast_axis)
  else:
    for a in broadcast_axis:
      batch_dims.remove(a)
  if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
    num = min_num
  else:
    num = nn.random_uniform(batch_dims, minval=min_num, maxval=max_num + 1, dtype="int32")
  _, indices, k_dim = nn.top_k(
    nn.random_uniform(batch_dims + [mask_axis], minval=0., maxval=1.),
    axis=mask_axis,
    k=num if isinstance(num, int) else nn.reduce(num, mode="max", axis=num.shape_ordered))
  # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
  if isinstance(num, int):
    for i in range(num):
      x = _mask_v2(
        x, mask_axis=mask_axis,
        pos=nn.gather(indices, axis=k_dim, position=i), max_amount=max_dims, mask_value=mask_value)
  else:
    loop = nn.Loop(axis=k_dim)
    k_dim_indices = nn.range_in_axis(indices, axis=k_dim)
    loop.state.x = x
    with loop:
      i = loop.unstack(k_dim_indices)
      loop.state.x = _mask_v2(
        loop.state.x, mask_axis=mask_axis,
        pos=nn.gather(indices, axis=k_dim, position=i), max_amount=max_dims, mask_value=mask_value)
      loop.stack(i)  # loop needs some dummy output currently...
    x = loop.state.x
  return x


@nn.scoped
def _mask_v2(x: nn.Tensor, *,
             mask_axis: nn.Dim,
             pos: nn.Tensor,
             max_amount: Union[int, nn.Tensor],
             mask_value: Union[int, float] = 0.
             ) -> nn.Tensor:
  """
  :param x: (batch,time,[feature]). any dim not mask_axis or in pos.shape will be broadcasted over
  :param mask_axis:
  :param pos: (batch,) (or multiple batch dims)
  :param max_amount: inclusive
  :param mask_value:
  """
  dim = nn.length(x, axis=mask_axis)
  amount = nn.random_uniform(shape=pos.shape_ordered, minval=1, maxval=max_amount + 1, dtype="int32")
  pos2 = nn.minimum(pos + amount, dim)
  idxs = nn.range_in_axis(x, axis=mask_axis)  # (dim,)
  cond = nn.compare_bc(idxs, ">=", pos) & nn.compare_bc(idxs, "<", pos2)  # (batch,dim)
  x = nn.where(cond, mask_value, x)
  return x


def specaugment_v1(x: nn.Tensor, **kwargs) -> nn.Tensor:
  """
  SpecAugment, wrapping the old-style code
  """
  return nn.make_layer({
    "class": "eval", "from": x,
    "eval": specaugment_v1_eval_func, "eval_locals": kwargs}, name="specaugment")


# Use this for an EvalLayer
def specaugment_v1_eval_func(*, source,
                             global_train_step_dependent: bool = True,
                             only_on_train: bool = True,
                             **kwargs):
  """
  :rtype: tf.Tensor
  """
  from returnn.tf.compat import v1 as tf
  data = source(0, as_data=True)
  time_factor = 1  # for switchout == 6
  x = data.placeholder
  network = kwargs["self"].network
  if global_train_step_dependent:
    step = network.global_train_step
    step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
    step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)
  else:
    step1 = step2 = 1

  def get_masked():
    """
    :return: masked tensor
    """
    x_masked = x
    x_masked = random_mask_v1(
      x_masked, batch_axis=data.batch_dim_axis, axis=data.time_dim_axis,
      min_num=step1 + step2,
      max_num=tf.maximum(
        tf.maximum(tf.shape(x)[data.time_dim_axis] // 100, 2) * (1 + step1 + step2 * 2),
        tf.shape(x)[data.time_dim_axis]),
      max_dims=20 // time_factor)
    x_masked = random_mask_v1(
      x_masked, batch_axis=data.batch_dim_axis, axis=data.feature_dim_axis,
      min_num=step1 + step2, max_num=2 + step1 + step2 * 2,
      max_dims=data.dim // 5)
    return x_masked

  if only_on_train:
    x = network.cond_on_train(get_masked, lambda: x)
  else:
    x = get_masked()
  return x


def random_mask_v1(x, *, batch_axis, axis, min_num, max_num, max_dims, mask_value=0.):
  """
  :param tf.Tensor x: (batch,time,feature)
  :param int batch_axis:
  :param int axis:
  :param int|tf.Tensor min_num:
  :param int|tf.Tensor max_num: inclusive
  :param int|tf.Tensor max_dims: inclusive
  :param float|int mask_value:
  :rtype: tf.Tensor
  """
  from returnn.tf.compat import v1 as tf
  n_batch = tf.shape(x)[batch_axis]
  if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
    num = min_num
  else:
    num = tf.random_uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
  # https://github.com/tensorflow/tensorflow/issues/9260
  # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
  z = -tf.log(-tf.log(tf.random_uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
  _, indices = tf.nn.top_k(z, num if isinstance(num, int) else tf.reduce_max(num))
  # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
  # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
  if isinstance(num, int):
    for i in range(num):
      x = _mask_v1(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims, mask_value=mask_value)
  else:
    _, x = tf.while_loop(
      cond=lambda i_, _: tf.less(i_, tf.reduce_max(num)),
      body=lambda i_, x_: (
        i_ + 1,
        tf.where(
          tf.less(i_, num),
          _mask_v1(
            x_, batch_axis=batch_axis, axis=axis, pos=indices[:, i_], max_amount=max_dims, mask_value=mask_value),
          x_)),
      loop_vars=(0, x))
  return x


def _mask_v1(x, *, batch_axis, axis, pos, max_amount, mask_value=0.):
  """
  :param tf.Tensor x: (batch,time,[feature])
  :param int batch_axis:
  :param int axis:
  :param tf.Tensor pos: (batch,)
  :param int|tf.Tensor max_amount: inclusive
  :param float|int mask_value:
  """
  from returnn.tf.compat import v1 as tf
  from returnn.tf.util.basic import where_bc
  ndim = x.get_shape().ndims
  n_batch = tf.shape(x)[batch_axis]
  dim = tf.shape(x)[axis]
  amount = tf.random_uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
  pos2 = tf.minimum(pos + amount, dim)
  idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
  pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
  pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
  cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
  if batch_axis > axis:
    cond = tf.transpose(cond)  # (dim,batch)
  cond = tf.reshape(cond, [tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)])
  x = where_bc(cond, mask_value, x)
  return x
