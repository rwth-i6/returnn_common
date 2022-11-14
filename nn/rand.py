"""
Common functions for random number generation.
"""

from __future__ import annotations
from typing import Union, Optional, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
  import tensorflow as tf
  import numpy
from .. import nn


class Random(nn.Module):
  """
  Random number generator with state.

  Note that there is ongoing discussion whether this module can be anonymous (not being an attribute of a module)
  to allow nn.random_uniform and co, or not allowing this:
  https://github.com/rwth-i6/returnn_common/issues/147
  """

  def __init__(self):
    super(Random, self).__init__()
    self._call_counter = 0

  _state_dim = nn.FeatureDim("random-state", None)

  def __call__(self, **kwargs) -> nn.Tensor:
    # For every call, we create a new state var to make sure there is no non-determinism.
    # https://github.com/rwth-i6/returnn_common/issues/148
    # No explicit seed, so RETURNN uses its global seed.
    init_state, _ = nn.random_state_init(out_dim=self._state_dim)
    state_var = nn.Parameter(init_state.shape_ordered, init_state.dtype, auxiliary=True, non_critical_for_restore=True)
    setattr(self, f"state_var{self._call_counter}", state_var)
    state_var.initial = init_state
    self._call_counter += 1
    return nn.random(
      explicit_state=state_var, auto_update_state=True,
      **kwargs)

  def uniform(self,
              shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
              *,
              minval: Union[int, float, nn.Tensor] = 0,
              maxval: Union[int, float, nn.Tensor]
              ) -> nn.Tensor:
    """
    Random uniform.

    :param shape:
    :param dtype:
    :param minval: inclusive
    :param maxval: exclusive
    """
    return self(distribution="uniform", shape=shape, dtype=dtype, minval=minval, maxval=maxval)

  def normal(self,
             shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
             *,
             mean: Union[int, float, nn.Tensor] = 0.,
             stddev: Union[int, float, nn.Tensor] = 1.,
             ) -> nn.Tensor:
    """normal"""
    return self(distribution="normal", shape=shape, dtype=dtype, mean=mean, stddev=stddev)

  def truncated_normal(self,
                       shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
                       *,
                       mean: Union[int, float, nn.Tensor] = 0.,
                       stddev: Union[int, float, nn.Tensor] = 1.,
                       ) -> nn.Tensor:
    """truncated normal"""
    return self(distribution="truncated_normal", shape=shape, dtype=dtype, mean=mean, stddev=stddev)

  def bernoulli(self, shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
                *,
                p: Union[float, nn.Tensor]
                ) -> nn.Tensor:
    """bernoulli. 0 or 1. p: prob for 1. uniform(1) <= p -> 1, else 0"""
    x = self.uniform(shape=shape, dtype=dtype, minval=0., maxval=1.)
    return nn.where(x <= p, nn.ones(shape=(), dtype=dtype), nn.zeros(shape=(), dtype=dtype))


def random_uniform(shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
                   *,
                   minval: Union[int, float, nn.Tensor] = 0,
                   maxval: Union[int, float, nn.Tensor]
                   ) -> nn.Tensor:
  """
  Random uniform.

  :param shape:
  :param dtype:
  :param minval: inclusive
  :param maxval: exclusive
  """
  return Random().uniform(shape=shape, dtype=dtype, minval=minval, maxval=maxval)


def random_normal(shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
                  *,
                  mean: Union[int, float, nn.Tensor] = 0.,
                  stddev: Union[int, float, nn.Tensor] = 1.,
                  ) -> nn.Tensor:
  """Random normal"""
  return Random().normal(shape=shape, dtype=dtype, mean=mean, stddev=stddev)


def random_truncated_normal(
      shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
      *,
      mean: Union[int, float, nn.Tensor] = 0.,
      stddev: Union[int, float, nn.Tensor] = 1.,
      ) -> nn.Tensor:
  """Random truncated normal"""
  return Random().truncated_normal(shape=shape, dtype=dtype, mean=mean, stddev=stddev)


def random_bernoulli(
      shape: Sequence[nn.Dim], dtype: str = nn.NotSpecified,
      *,
      p: Union[float, nn.Tensor]
      ) -> nn.Tensor:
  """Random Bernoulli. 0 or 1. p: prob for 1. uniform(1) <= p -> 1, else 0"""
  return Random().bernoulli(shape=shape, dtype=dtype, p=p)


def random_label(shape: Sequence[nn.Dim], sparse_dim: nn.Dim, dtype: str = "int32") -> nn.Tensor:
  """
  Random label
  """
  res = random_uniform(shape=shape, dtype=dtype, minval=0, maxval=sparse_dim.dimension)
  res = nn.reinterpret_set_sparse_dim(res, sparse_dim)
  return res


def random(
           *,
           shape: Sequence[nn.Dim],
           distribution: str,
           mean: Optional[Union[int, float, nn.Tensor]] = nn.NotSpecified,
           stddev: Optional[Union[int, float, nn.Tensor]] = nn.NotSpecified,
           bound: Optional[Union[int, float, nn.Tensor]] = nn.NotSpecified,
           minval: Optional[Union[int, float, nn.Tensor]] = nn.NotSpecified,
           maxval: Optional[Union[int, float, nn.Tensor]] = nn.NotSpecified,
           dtype: str = nn.NotSpecified,
           seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = nn.NotSpecified,
           algorithm: Optional[Union[str, tf.random.Algorithm]] = nn.NotSpecified,
           explicit_state: Optional[nn.Tensor] = nn.NotSpecified,
           auto_update_state: Optional[bool] = nn.NotSpecified,
           static: Optional[bool] = nn.NotSpecified,
           name: Optional[Union[str, nn.NameCtx]] = None) -> nn.Tensor:
  """
  Generates random numbers from uniform or normal or truncated normal distribution.

  This uses the TensorFlow stateless random ops internally, i.e. all the state handling is explicit.
  The state var can be explicitly provided and initialized via :class:`RandomStateInitLayer`,
  or when not provided it will be automatically created.

  There are two possible distinct use cases:

  - For any randomness in the model, e.g. dropout. So each ``session.run`` step will produce a new random number
    and advance the random state.
  - To initialize parameters via the config, using :class:`VariableLayer` with the ``init_by_layer`` option.
    This will only be called once when initializing the parameters.
    For this use case, we do not want to keep a random state var.
    You can just pass ``static=False``.
    Alternatively you could also pass the output of a :class:`RandomStateInitLayer` as ``state``.

  :param Sequence[nn.Dim] shape:
  :param str distribution: "uniform", "normal" or "truncated_normal"
  :param int|float|nn.Tensor|None mean:
  :param int|float|nn.Tensor|None stddev:
  :param int|float|nn.Tensor|None bound: for uniform, defining the range [-bound, bound)
  :param int|float|nn.Tensor|None minval: for uniform
  :param int|float|nn.Tensor|None maxval: for uniform
  :param str dtype:
  :param int|list[int]|numpy.ndarray|None seed: If not given, uses self.network.random.randint,
    i.e. then it is controlled by the global seed setting, and every layer would get its own seed.
    If you specify it explicitly, make sure every :class:`RandomLayer` uses a different seed,
    otherwise you would get the same random numbers everywhere.
  :param str|tf.random.Algorithm|None algorithm: see :class:`RandomStateInitLayer`
  :param nn.Tensor|None explicit_state: You can pass the state explicitly here.
    If not given, will be created automatically, and updated automatically.
    You could pass a :class:`VariableLayer` with initial value via :class:`RandomStateInitLayer`,
    or directly a :class:`RandomStateInitLayer`.
    If auto_update_state is True, it must be a variable,
    and every time a new random number is created, this variable is updated.
    Otherwise (default) it will not be updated automatically.
  :param bool|None auto_update_state: only used when you pass an explicit state
  :param bool|None static: if no state at all should be used. it just relies on the seed then.
  :param str|nn.NameCtx|None name:
  :return: layer
  """
  args = {
    'shape': shape,
    'distribution': distribution,
    'mean': mean,
    'stddev': stddev,
    'bound': bound,
    'minval': minval,
    'maxval': maxval,
    'dtype': dtype,
    'seed': seed,
    'algorithm': algorithm,
    'explicit_state': explicit_state,
    'auto_update_state': auto_update_state,
    'static': static,
    'shape_deps': nn.get_dim_deps(shape),
    }
  args = {key: value for (key, value) in args.items() if value is not nn.NotSpecified}
  return nn.make_layer({
    'class': 'random',
    **args}, name=name or 'random')
