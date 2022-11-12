r"""
Weight Normalization from https://arxiv.org/abs/1602.07868

Code adapted from PyTorch implementation.

See :func:`weight_norm` to apply weight normalization to one parameter of a module.

It's usually not a good idea to just apply it everywhere,
thus this is rather selective.
For :class:`nn.Linear`, as described in the original paper,
you want to apply it on the ``weight`` parameter only.
Additionally, it is recommended to combine it with a special variant of batch-norm,
only mean-normalizing the input.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, TypeVar
import numpy
from ... import nn


T_module = TypeVar('T_module', bound=nn.Module)


def weight_norm(module: T_module, name: str = "weight", dim: Optional[nn.Dim] = nn.NotSpecified) -> T_module:
  r"""Applies weight normalization to a parameter in the given module.

  .. math::
       \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

  Weight normalization is a reparameterization that decouples the magnitude
  of a weight tensor from its direction. This replaces the parameter specified
  by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
  (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
  Weight normalization is implemented via a hook that recomputes the weight
  tensor from the magnitude and direction before every :meth:`~Module.forward`
  call.

  By default, with ``dim=weight.feature_dim``, the norm is computed independently per output
  channel/plane. To compute a norm over the entire weight tensor, use
  ``dim=None``.

  See https://arxiv.org/abs/1602.07868

  Args:
      module (Module): containing module
      name (str, optional): name of weight parameter
      dim (int, optional): dimension over which to compute the norm

  Returns:
      The original module with the weight norm hook
  """
  weight = getattr(module, name)
  if isinstance(weight, WeightNorm):
    raise RuntimeError("Cannot register two weight_norm hooks on the same parameter {}".format(name))
  assert isinstance(weight, nn.Parameter)

  fn = WeightNorm(weight, dim)

  delattr(module, name)  # remove w from parameter list
  assert not hasattr(module, f"{name}_normalized")
  setattr(module, f"{name}_normalized", fn)  # add weight norm functions
  setattr(module, name, fn.compute_weight())  # set it to calculated weight

  return fn


def remove_weight_norm(module: T_module, name: str = 'weight') -> T_module:
  r"""Removes the weight normalization reparameterization from a module.

  Args:
      module (Module): containing module
      name (str, optional): name of weight parameter
  """
  fn = getattr(module, f"_{name}_weight_normalized")
  assert isinstance(fn, WeightNorm)
  delattr(module, name)
  delattr(module, f"{name}_normalized")

  p = nn.Parameter(fn.v.shape_ordered, fn.v.dtype)
  p.initial = fn.weight_init()
  setattr(module, name, p)
  return module


class WeightNorm(nn.Module):
  """
  Encapsulates a weight-normalized parameter.
  """

  def __init__(self, weight: nn.Parameter, dim: Optional[nn.Dim], eps=1e-6) -> None:
    self.dim = dim
    self.eps = eps

    # add g and v as new parameters and express w as g/||v|| * v
    g = nn.Parameter([dim] if dim else [], weight.dtype)
    v = nn.Parameter(weight.shape_ordered, weight.dtype)
    self.g = g
    self.v = v

    self.norm_axes = v.batch_dims_ordered(dim)
    if isinstance(weight, nn.Parameter) and weight.initial is not None:
      # Custom ParamInit such that any deepcopy will make individual random inits.
      v.initial = WeightNormDirectionParamInit(weight.initial)
      g.initial = WeightNormScaleParamInit(self)
    else:
      g.initial = 1.

  def compute_weight(self) -> nn.Tensor:
    """computes the actual weight from g and v"""
    g = self.g
    v = self.v
    # See _weight_norm in PyTorch.
    # https://github.com/pytorch/pytorch/blob/324ac93a43a93f671bb34b835926b22d13442735/aten/src/ATen/native/WeightNorm.cpp#L107
    # v*(g/at::norm_except_dim(v, 2, dim));
    # Tensor norm_except_dim(const Tensor & v, int64_t pow, int64_t dim) {
    #    if (dim == -1)
    #      return v.norm(pow);
    #    else if (dim == 0) {
    #      std::vector<int64_t> output_size(v.dim(), 1);
    #      output_size[0] = v.size(0);
    #      return v.contiguous().view({v.size(0), -1}).norm(pow, 1).view(output_size);
    #    } ...
    assert isinstance(v, nn.Tensor)
    return v * (g * nn.rsqrt(nn.reduce(nn.square(v), mode="sum", axis=self.norm_axes) + self.eps))

  def g_init(self, weight_init: Union[nn.Tensor, nn.RawTensorTypes]) -> Union[nn.Tensor, nn.RawTensorTypes]:
    """
    given specific weight_init, calculate g_init
    """
    if not isinstance(weight_init, nn.Tensor):
      return numpy.sqrt(numpy.square(weight_init) + self.eps)  # assume scalar
    return nn.sqrt(nn.reduce(nn.square(weight_init), mode="sum", axis=self.norm_axes) + self.eps)

  def weight_init(self) -> Optional[nn.init.ParamInitType]:
    """
    from the original weight, or wrapped
    """
    if self.v.initial is None:
      return None
    init = self.v.initial
    if isinstance(init, WeightNormDirectionParamInit):
      return init.weight_init
    return None


class WeightNormDirectionParamInit(nn.init.ParamInit):
  """
  Param init weight norm
  """

  def __init__(self, weight_init: nn.init.ParamInitType):
    self.weight_init = weight_init
    self.weight_init_value = None  # type: Optional[Union[nn.Tensor, nn.RawTensorTypes]]

  def __call__(self, shape: Sequence[nn.Dim], dtype: str) -> Union[nn.Tensor, nn.RawTensorTypes]:
    if isinstance(self.weight_init, nn.init.ParamInit):
      if self.weight_init_value is None:
        self.weight_init_value = self.weight_init(shape, dtype)
        return self.weight_init_value
      raise Exception(f"{self}: Don't call this twice. You probably miss a deepcopy.")
    return self.weight_init

  def __copy__(self):
    return WeightNormDirectionParamInit(self.weight_init)

  def get_weight_init_value(self) -> Union[nn.Tensor, nn.RawTensorTypes]:
    """get value"""
    if isinstance(self.weight_init, nn.init.ParamInit):
      assert self.weight_init_value is not None, f"{self}: Expected to be called before."
      return self.weight_init_value
    return self.weight_init


class WeightNormScaleParamInit(nn.init.ParamInit):
  """
  Param init weight norm
  """
  def __init__(self, parent: WeightNorm):
    self.parent = parent

  def __call__(self, shape: Sequence[nn.Dim], dtype: str) -> Union[nn.Tensor, nn.RawTensorTypes]:
    v_init = self.parent.v.initial
    if isinstance(v_init, WeightNormDirectionParamInit):
      return self.parent.g_init(v_init.get_weight_init_value())
    return 1.
