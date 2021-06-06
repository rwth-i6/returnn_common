"""
This file is auto-generated.
"""

from .base import ILayerMaker, LayerRef, LayerDictRaw


class _Base(ILayerMaker):
  """
  This is the base class for all layers.
  Every layer by default has a list of source layers `sources` and defines `self.output` which is of type :class:`Data`.
  It shares some common functionality across all layers, such as explicitly defining the output format,
  some parameter regularization, and more.

  If you want to implement your own layer::

      class YourOwnLayer(_ConcatInputLayer):  # e.g. either _ConcatInputLayer or LayerBase as a base
          " some docstring "
          layer_class = "your_layer_name"

          def __init__(self, your_kwarg1, your_opt_kwarg2=None, **kwargs):
              " docstring, document the args! "
              super(YourOwnLayer, self).__init__(**kwargs)
              # Now we need to set self.output, which must be of type :class:`Data`.
              # It is set at this point to whatever we got from `self.get_out_data_from_opts()`,
              # so it is enough if we set self.output.placeholder and self.output.size_placeholder,
              # but we could also reset self.output.
              self.output.placeholder = self.input_data.placeholder + 42  # whatever you want to do
              # If you don't modify the sizes (e.g. sequence-length), just copy the input sizes.
              self.output.size_placeholder = self.input_data.size_placeholder.copy()

          @classmethod
          def get_out_data_from_opts(cls, **kwargs):
              " This is supposed to return a :class:`Data` instance as a template, given the arguments. "
              # example, just the same as the input:
              return get_concat_sources_data_template(kwargs["sources"], name="%s_output" % kwargs["name"])

  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': None,
      'from': source.get_name()}


class Source(_Base):
  """
  This gives access to some entry from network.extern_data (:class:`ExternData`).
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'source',
      'from': source.get_name()}


class _ConcatInput(_Base):
  """
  Base layer which concatenates all incoming source layers in the feature dimension,
  and provides that as `self.input_data`, which is of type :class:`Data`.
  This is the most common thing what many layers do with the input sources.
  If there is only a single source, will not do anything.
  This layer also optionally can do dropout on the input.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': None,
      'from': source.get_name()}


class Copy(_ConcatInput):
  """
  This layer does nothing, it copies its input.
  If multiple sources are provided, they are concatenated in the feature-dim.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'copy',
      'from': source.get_name()}


class Dropout(Copy):
  """
  Just the same as :class:`CopyLayer`, because that one already supports dropout.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'dropout',
      'from': source.get_name()}


class ScaledGradient(Copy):
  """
  Just tf.identity in the forward pass.
  Scales the gradient by some factor in backprop.
  Can be used as gradient reversal layer (with negative factor).
  Uses :func:`TFUtil.scaled_gradient`, or :func:`tf.stop_gradient`
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'scaled_grad',
      'from': source.get_name()}


class Activation(Copy):
  """
  This layer just applies an activation function.
  See :func:`TFUtil.get_activation_function` about supported functions.
  Also see :class:`EvalLayer` and :class:`CombineLayer` for similar layers.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'activation',
      'from': source.get_name()}


class BatchNorm(Copy):
  """
  Implements batch-normalization (http://arxiv.org/abs/1502.03167) as a separate layer.

  Also see :class:`NormLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'batch_norm',
      'from': source.get_name()}


class LayerNorm(_ConcatInput):
  """
  Applies `layer-normalization <https://arxiv.org/abs/1607.06450>`__.

  Note that we *just* normalize over the feature-dim axis here.
  This is consistent to the default behavior of :class:`tf.keras.layers.LayerNormalization`
  and also how it is commonly used in many models, including Transformer.

  However, there are cases where it would be common to normalize over all axes except batch-dim,
  or all axes except batch and time.
  For a more generic variant, see :class:`NormLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'layer_norm',
      'from': source.get_name()}


class Norm(_ConcatInput):
  """
  Normalize over specified axes, e.g. time and/or feature axis.

  Note: For calculating a norm, see :class:`MathNormLayer` instead.

  In case of just feature (``axes="F"``),
  this corresponds to `layer normalization <https://arxiv.org/abs/1607.06450>`__ (see :class:`LayerNormLayer`).
  In case of time and feature (``axes="TF"``) for a 3D input,
  or more general all except batch (``axes="except_batch"``),
  this corresponds to `group normalization <https://arxiv.org/abs/1803.08494>`__ with G=1,
  or non-standard layer normalization.
  (The definition of layer-normalization is not clear on what axes should be normalized over.
  In many other frameworks, the default axis is just the last axis,
  which is usually the feature axis.
  However, in certain implementations and models,
  it is also common to normalize over all axes except batch.)

  The statistics are calculated just on the input.
  There are no running statistics (in contrast to batch normalization, see :class:`BatchNormLayer`).

  For some discussion on the definition of layer-norm vs group-norm,
  also see
  `here <https://stats.stackexchange.com/questions/485550/is-group-norm-with-g-1-equiv-to-layer-norm>`__
  and `here <https://github.com/tensorflow/addons/issues/2143>`__.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'norm',
      'from': source.get_name()}


class MathNorm(_ConcatInput):
  """
  Calculates sum(abs(x) ** p) ** (1./p).
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'math_norm',
      'from': source.get_name()}


class Slice(_ConcatInput):
  """
  Slicing on the input, i.e. x[start:end:step] in some axis.
  See also :class:`SliceNdLayer`, for variable start.
  See also :class:`GatherLayer`, for one single position.

  Note that __getitem__ on a TF tensor (or also Numpy ND array) is more generic,
  and supports slices in multiple axes, as well as adding new dimensions, etc.
  It even allows to get boolean values, and then applies a boolean mask.
  See TF _slice_helper (== tf.Tensor.__getitem__) for a generic implementation,
  which calls tf.strided_slice.
  If we ever need such more generic support, we might consider adding a new layer,
  like ``GenericSliceLayer``, which gets a ``splice_spec``,
  just like ``_slice_helper`` (argument to ``__getitem__``).
  But any such a slice can already be constructed with multiple individual layers,
  which perform individual slices (per axis).

  We just support slicing in a single axis here, with optional striding (slice_step).
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'slice',
      'from': source.get_name()}


class SliceNd(_ConcatInput):
  """
  This takes out a slice-range from some axis,
  e.g. ``x[start:start + size]``.
  This layers allows a different start slice point for each batch,
  in contrast to :class:`SliceLayer`, and the start is variable.
  See also :class:`GatherNdLayer`.
  :class:`PrefixInTimeLayer` can recover the original shape (by zero-padding).
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'slice_nd',
      'from': source.get_name()}


class Gather(_ConcatInput):
  """
  Gathers slices on a specified axis from the input layer using indices from a ``position`` layer.
  If the input is a layer of the shape ``[B,D,F1]``, and position of shape ``[B,F2]``, this will yield output of the
  shape ``[B,F2,F1]`` where

  ``output[b,f2,f1] = input[b,position[b,f2],f1]``

  (if ``D`` is the axis to gather from).
  In general, all shared axes of the input and the positions will be considered as batch-axes.

  The ``position`` argument can also be an ``int``.
  In this case, this simply gives ``input[position]`` one the specified ``axis``.

  It's basically a wrapper around ``tf.gather``.
  It provides the same functionality as the deprecated ``GatherNdLayer``, but is more generic.
  See also :class:`GatherNdLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'gather',
      'from': source.get_name()}


class GatherNd(_ConcatInput):
  """
  Warning: This layer is deprecated, use the more general :class:`GatherLayer` instead.
  :class:`GatherLayer` should be equivalent, but is more general (supports multiple batch dimensions, can specify gather
   axis) and its name is less misleading.

  This takes out a position from some axis, e.g. ``x[pos]``.
  This layers allows a different position for each batch.
  It's basically a wrapper around ``tf.gather`` (the name of this layer is misleading).
  See also :class:`GatherLayer` instead, which will replace this layer in the future.
  See also :class:`SliceNdLayer`.
  See also :class:`ScatterNdLayer`, which is the inverse operation.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'gather_nd',
      'from': source.get_name()}


class ScatterNd(_ConcatInput):
  """
  The inverse of :class:`GatherNdLayer`.
  Mostly a wrapper for ``tf.scatter_nd``.

  The input to the layer are the ``updates``, the ``indices`` are via the ``position`` argument.
  The indices are into the newly constructed output dimension.
  The output shape is constructed via the common shape of the input, the position,
  and the the unique common axis (if not unique, we would need to introduce an option to specify it)
  is replaced by the given output dimension (currently via ``output_dim_via_time_from``).

  Examples::

    position (indices): (B,eTs)
    input (updates): (eTs,D) or (B,eTs,D) -> expanded to (B,eTs,D)
    output shape: (B,eT,D)

    position (indices): (B,dT,eTs)
    input (updates): (eTs,D) -> expanded to (B,dT,eTs,D)
    output shape: (B,dT,eT,D)

    position (indices): (dT,eTs)
    input (updates): (eTs,D) -> expanded to (dT,eTs,D)
    output shape: (dT,eTs,D)

    position (indices): (dT,eTs)
    input (updates): (B,eTs,D) -> expanded to (dT,eTs,B,D)
    output shape: (dT,eT,B,D)

  In all these examples, output_dim_via_time_from is (B,eT,F), and eTs gets replaced by eT.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'scatter_nd',
      'from': source.get_name()}


class Linear(_ConcatInput):
  """
  Linear/forward/fully-connected/1x1-conv layer.
  Does a linear transformation on the feature-dimension of the input
  with an optional bias term and an optional activation function.
  See also :class:`DotLayer`, :class:`ElemwiseProdLayer`, :class:`WeightedSumLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'linear',
      'from': source.get_name()}


class Softmax(Linear):
  """
  Just a LinearLayer with activation="softmax" by default.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'softmax',
      'from': source.get_name()}


class Length(_Base):
  """
  Returns the length of sources as (B,), via input size_placeholder.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'length',
      'from': source.get_name()}


class SoftmaxOverSpatial(_ConcatInput):
  """
  This applies a softmax over spatial axis/axes (currently only time axis supported).
  E.g. when the input is of shape (B,T,dim), the output will be (B,T,dim).
  It automatically masks the frames outside the seq defined by the seq-len.
  In contrast to :class:`SoftmaxLayer`, this will not do a linear transformation.
  See :class:`SeqLenMaskLayer` if you just want to apply a masking.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'softmax_over_spatial',
      'from': source.get_name()}


class SeqLenMask(_ConcatInput):
  """
  Masks some values away given the seq_len_source with mask_value.
  Also see :class:`SoftmaxOverSpatialLayer`.
  Also see :class:`SwitchLayer`, which can be used to apply a generic mask.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'seq_len_mask',
      'from': source.get_name()}


class RandInt(_Base):
  """
  Generates random numbers using ``tf.random.uniform``
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'rand_int',
      'from': source.get_name()}


class Range(_Base):
  """
  Generic wrapper around ``tf.range``.
  See also :class:`RangeInAxisLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'range',
      'from': source.get_name()}


class RangeInAxis(_Base):
  """
  Assume that the input is e.g. (B,T,D), and you specify axis="T", you will get (B=1,T,D=1),
  where the specified axis is filled with ``tf.range``.
  See also :class:`RangeLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'range_in_axis',
      'from': source.get_name()}


class BatchSoftmax(_ConcatInput):
  """
  Softmax over spacial and feature axis
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'batch_softmax',
      'from': source.get_name()}


class Constant(_Base):
  """
  Output is a constant value.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'constant',
      'from': source.get_name()}


class Gating(_ConcatInput):
  """
  Splits the output into two equal parts, applies the gate_activation (sigmoid by default)
  on the one part, some other activation (e.g. tanh) on the other part and then
  element-wise multiplies them.
  Thus, the output dimension is input-dimension / 2.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'gating',
      'from': source.get_name()}


class Window(_ConcatInput):
  """
  Adds a window dimension.
  By default, uses the time axis and goes over it with a sliding window.
  The new axis for the window is created right after the time axis.
  Will always return as batch major mode.
  E.g. if the input is (batch, time, dim), the output is (batch, time, window_size, dim).
  If you want to merge the (window_size, dim) together to (window_size * dim,),
  you can use the MergeDimsLayer, e.g. {"class": "merge_dims", "axes": "except_time"}.
  Use stride==window_size and window_right=window_size - 1 in combination with a
  MergeDimsLayer to achieve feature stacking with right-hand zero padding.

  This is not to take out a window from the time-dimension.
  See :class:`SliceLayer` or :class:`SliceNdLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'window',
      'from': source.get_name()}


class Cumsum(_ConcatInput):
  """
  Basically wraps tf.cumsum. Also supports that in the RecLayer.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'cumsum',
      'from': source.get_name()}


class Pad(_ConcatInput):
  """
  Adds (e.g. zero) padding in some axis or axes.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'pad',
      'from': source.get_name()}


class MergeDims(_ConcatInput):
  """
  Merges a list of axes into a single one. (Flatten the dims.)
  E.g. input is (batch, width, height, dim) and axes=(1,2), then we get (batch, width*height, dim).
  Or input is (batch, time, height, dim) and axes="except_time", then we get (batch, time, height*dim).
  See also :class:`CombineDimsLayer`.
  When batch and time got merged, :class:`SplitBatchTimeLayer` can undo this.
  When you want to merge batch and time, but remove the padding efficiently, i.e. flatten it,
  see :class:`FlattenBatchLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'merge_dims',
      'from': source.get_name()}


class Split(_ConcatInput):
  """
  Splits one axis into multiple parts, via tf.split.
  self.output is simply the input copied.
  Each part can be accessed via the sublayers "/%i".
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'split',
      'from': source.get_name()}


class SplitDims(_ConcatInput):
  """
  Splits one axis into multiple axes.
  E.g. if you know that your feature-dim is composed by a window,
  i.e. the input is (batch, time, window * feature),
  you can set axis="F", dims=(window, -1),
  and you will get the output (batch, time, window, feature).

  If the split axis has a dynamic length,
  exactly one of the axes that we split into need to also have a dynamic length.
  You can e.g. use this to split the input dimension into smaller "chunks" of a fixed window size.
  E.g. you could have input (batch, time, feature) and set axis="T", dims=(-1, window),
  to get output (batch, split_time, window, feature).
  In this case, the exact sequence lengths are lost and everything is padded to multiples of the window size using
  the given padding value.
  Use :class:`ReinterpretDataLayer` to receive back the original sequence lengths after merging.

  Also see :class:`SplitBatchTimeLayer`.
  Also see :class:`MergeDimsLayer` which can undo this operation.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'split_dims',
      'from': source.get_name()}


class SplitBatchTime(_ConcatInput):
  """
  A very specific layer which expects to get input of shape (batch * time, ...)
  and converts it into (batch, time, ...), where it recovers the seq-lens from some other layer.
  See :class:`SplitDimsLayer` for a more generic layer.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'split_batch_time',
      'from': source.get_name()}


class FlattenBatch(_ConcatInput):
  """
  Merges one axis into the batch axis.
  If the axis has dynamic lengths, this would use flattening,
  i.e. recalculate the padding, i.e. the size changes.
  This basically wraps :func:`flatten_with_seq_len_mask` or :func:`flatten_with_seq_len_mask_time_major`.
  See also :class:`MergeDimsLayer`, which does not do flattening,
  i.e. the size stays the same.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'flatten_batch',
      'from': source.get_name()}


class UnflattenNd(_ConcatInput):
  """
  This keeps the batch axis as-is, i.e. the flattening/unflattening did not happen on the batch axis.

  Example:

    Assumes that the input is of shape (B,T,<Ds>) which represents flattened images,
    where each image is of size width * height.
    We additionally provide these image sizes (shape (B,2)), i.e. (width,height) tuples.
    We return the unflattened images of shape (B,W,H,<Ds>), where W/H are the max width/height.

  This basically wraps :func:`TFUtil.unflatten_nd`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'unflatten_nd',
      'from': source.get_name()}


class ExpandDims(_ConcatInput):
  """
  Adds some axis.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'expand_dims',
      'from': source.get_name()}


class Repeat(_ConcatInput):
  """
  A wrapper around tf.repeat, but supports an additional batch axis for the durations
  The sum of the repetitions has to be non-zero for each sequence in the batch.

  This layer can only be used with Tensorflow 1.15.0 or newer.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'repeat',
      'from': source.get_name()}


class Tile(_ConcatInput):
  """
  A wrapper around tf.tile
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'tile',
      'from': source.get_name()}


class Cast(Copy):
  """
  Cast to some other dtype.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'cast',
      'from': source.get_name()}


class SwapAxes(_ConcatInput):
  """
  Swaps two axes. Basically a wrapper around :func:`TFUtil.swapaxes`.
  Note that usually, this should not be needed, and it is recommended not to be used,
  as this will be unnecessarily inefficient.
  Normally, all RETURNN layers will automatically transpose the input data into whatever format they need.

  All axes always have a special meaning (e.g. feature dim or time dim)
  or dimension tag (e.g. for time axes, including dyn seq lengths).
  If you need to change the meaning (and not actually transpose / swap axes),
  you need to use :class:`ReinterpretDataLayer`.

  See also :class:`TransposeLayer` for a more generic variant.

  See also :class:`ReinterpretDataLayer`, which does not swap/transpose axes,
  but allows to reinterpret their meaning / dim tags.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'swap_axes',
      'from': source.get_name()}


class Transpose(_ConcatInput):
  """
  Basically a wrapper around :func:`tf.transpose`.
  Note that usually, this should not be needed, and it is recommended not to be used,
  as this will be unnecessarily inefficient.
  Normally, all RETURNN layers will automatically transpose the input data into whatever format they need.

  All axes always have a special meaning (e.g. feature dim or time dim)
  or dimension tag (e.g. for time axes, including dyn seq lengths).
  If you need to change the meaning (and not actually transpose / swap axes),
  you need to use :class:`ReinterpretDataLayer`.

  See also :class:`ReinterpretDataLayer`, which does not transpose axes,
  but allows to reinterpret their meaning / dim tags.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'transpose',
      'from': source.get_name()}


class ReinterpretData(_ConcatInput):
  """
  Acts like the :class:`CopyLayer` but reinterprets the role of some axes or data.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'reinterpret_data',
      'from': source.get_name()}


class Conv(_ConcatInput):
  """
  A generic convolution layer which supports 1D, 2D and 3D convolution.
  Pooling can be done in the separate "pool" layer.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'conv',
      'from': source.get_name()}


class Pool(_ConcatInput):
  """
  A generic N-D pooling layer.
  This would usually be done after a convolution for down-sampling.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'pool',
      'from': source.get_name()}


class Dct(_ConcatInput):
  """
  Layer to perform DCT
  Wraps :func:`tf.signal.dct`. For further documentation on the input arguments, refer to
  https://www.tensorflow.org/api_docs/python/tf/signal/dct
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'dct',
      'from': source.get_name()}


class TransposedConv(_ConcatInput):
  """
  Transposed convolution, sometimes also called deconvolution.
  See :func:`tf.nn.conv2d_transpose` (currently we support 1D/2D).
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'transposed_conv',
      'from': source.get_name()}


class Reduce(_ConcatInput):
  """
  This reduces some axis by using "sum" or "max".
  It's basically a wrapper around tf.reduce_sum or tf.reduce_max.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'reduce',
      'from': source.get_name()}


class ReduceOut(_ConcatInput):
  """
  Combination of :class:`SplitDimsLayer` applied to the feature dim
  and :class:`ReduceLayer` applied to the resulting feature dim.
  This can e.g. be used to do maxout.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'reduce_out',
      'from': source.get_name()}


class Squeeze(_ConcatInput):
  """
  Removes an axis with dimension 1.
  This is basically a wrapper around tf.squeeze.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'squeeze',
      'from': source.get_name()}


class Stack(_Base):
  """
  Stacks multiple inputs together using :func:`tf.stack`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'stack',
      'from': source.get_name()}


class WeightedSum(_ConcatInput):
  """
  Calculates a weighted sum, either over a complete axis of fixed dimension, or over some window.
  Can also do that for multiple axes.
  The weights are a trainable parameter matrix.
  Similar would be to use :class:`ElemwiseProdLayer` and :class:`ReduceLayer`,
  or just a :class:`DotLayer` with a :class:`VariableLayer`.
  See also :class:`LinearLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'weighted_sum',
      'from': source.get_name()}


class ElemwiseProd(_ConcatInput):
  """
  Element-wise product in some axes.
  Microsoft calls this "static attention", in Deep Conv. NN with Layer-wise Context Expansion and Attention (LACE).
  The matrix/tensor to be used for the product are given as a trainable parameter.
  See also :class:`LinearLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'elemwise_prod',
      'from': source.get_name()}


class PrefixInTime(_ConcatInput):
  """
  Adds some prefix in time dimension.
  This is kind of the reverse of :class:`SliceNdLayer` does.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'prefix_in_time',
      'from': source.get_name()}


class PostfixInTime(_ConcatInput):
  """
  Adds some postfix in time dimension.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'postfix_in_time',
      'from': source.get_name()}


class TimeChunking(_ConcatInput):
  """
  Performs chunking in time. See :func:`TFNativeOp.chunk`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'time_chunking',
      'from': source.get_name()}


class TimeUnChunking(_ConcatInput):
  """
  Performs chunking in time. See :func:`TFNativeOp.chunk`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'time_unchunking',
      'from': source.get_name()}


class Dot(_Base):
  """
  This performs a dot-product of two sources.
  The underlying matmul expects shapes (shared..., I, J) * (shared..., J, K) -> (shared..., I, K).
  We say that J is the axis to be reduced,
  I is the var-dim of source 1, and K is the var-dim of source 2.
  I, J, K can also be multiple axes from the sources.
  The var-dims don't need to exist.
  All other axes (shared...) are expected to match.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'dot',
      'from': source.get_name()}


class ShiftAxis(_ConcatInput):
  """
  Shifts the dimensions in an axis around.
  This layer may change the axis-dimension.

  This name might be confusing. No axis will be shifted here. See :class:`SwapAxesLayer` for that.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'shift_axis',
      'from': source.get_name()}


class Resize(_ConcatInput):
  """
  Resizes the input, i.e. upsampling or downsampling.
  Supports different kinds, such as linear interpolation or nearest-neighbor.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'resize',
      'from': source.get_name()}


class CombineDims(_ConcatInput):
  """
  Combines multiple dimensions.
  See also :class:`MergeDimsLayer`. This is deprecated in favor of :class:`MergeDimsLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'combine_dims',
      'from': source.get_name()}


class Remove(_Base):
  """
  Currently, assumes sparse data, and removes a specific symbol from the data.

  It is recommended to use :class:`MaskedComputationLayer` in combination with e.g.
  a :class:CompareLayer` instead, as this provides more flexibility.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'remove',
      'from': source.get_name()}


class Combine(_Base):
  """
  Applies a binary operation, such as addition, to all sources while accumulating the partial results.
  In the first step, the binary operation is performed on the first two sources.
  After the first step, the previous results is always the left-hand operator.

  Its basic working is similar to the `reduce` function used in functional programming.
  Also see :class:`ActivationLayer`, or :class:`CompareLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'combine',
      'from': source.get_name()}


class Eval(Combine):
  """
  Evaluates some string.
  The :class:`CombineLayer` provides this functionality, thus this is just a special case of it.
  Also see :class:`ActivationLayer`, or :class:`CompareLayer`.

  The output type is defined as a broadcasted extension of all sources.
  You can overwrite it by (partially) specifying `out_type`.
  `out_type` can also be a generic Python function, returning a `Data` instance.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'eval',
      'from': source.get_name()}


class Compare(_Base):
  """
  Compares element-wise the tokens of all input sequences among themselves and/or with a specified given value.
  The comparisons are performed in a chain according to the order in which they are listed.

  Example::

      {"class": "compare", "from": ["i1", "i2"], "value": val, "kind": "less"}

  computes i1 < i2 < val and it is true only if the whole chain of operations is true.
  The final result is the logical "and" of all comparisons. Note that `value` is the last element to be compared to.

  A common example usage is the `end` layer in a rec subnetwork to specify the stopping criterion,
  e.g. the last generated token is equal to the end-of-sentence token::

      "output": {"class": "rec", "from": [], "unit": {
          .
          .
          .
          "end": {"class": "compare", "from": "output", "value": end_of_sentence_id}
      }, "target": "classes0"}

  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'compare',
      'from': source.get_name()}


class Switch(_Base):
  """
  Wrapper around ``tf.where()`` (or more generically :func:`TFUtil.where_bc`),
  or statically choose a single source if the condition is a callable (...)->bool.
  (``tf.cond`` is not useful here, as the sources would have been already constructed and computed.)

  This layer is also useful for applying any kind of generic masking to the frames.
  E.g. one could have a layer called "mask" computing a boolean mask for the values stored in another layer "input".
  Then use this layer with condition="mask", true_from="input", false_from=mask_value,
  to mask out all frames where the mask is false with the mask_value.

  See also :class:`CondLayer`.
  See also :class:`SeqLenMaskLayer` if you just want to mask using the sequence lengths.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'switch',
      'from': source.get_name()}


class Cond(_Base):
  """
  See also :class:`SwitchLayer`, which uses :func:`tf.where`.
  Here, we use `tf.cond` instead. I.e. the condition has to be a scalar bool,
  and only the corresponding true/false branch is computed.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'cond',
      'from': source.get_name()}


class SearchSorted(_Base):
  """
  Basically wraps :func:`tf.searchsorted`.

  Takes a tensor `sorted_sequence` that is sorted along one axis, and a tensor `values`.
  Will compute an output tensor with the same axes as `values`,
  where each entry is the index of the value within the sorted sequence.
  All (batch) axes of `sorted_sequence` except for the axis it is sorted along must be present in `values`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'search_sorted',
      'from': source.get_name()}


class Subnetwork(_Base):
  """
  You can define a whole subnetwork as a single layer by this class.

  The subnetwork will be specified by a ``dict[str,dict[str]]``, just like
  a normal network is specified in the config.

  The ``"output"`` layer of the subnetwork will be the output of this
  subnetwork-layer.

  With ``concat_sources=True`` (default),
    the input to this layer will be represented as the ``"data:data"`` or simply ``"data"``
    in the subnetwork,
  otherwise with ``concat_sources=False``,
    the input to this layer will be represented as ``"data:input_layer_name"``
    and also ``"data:0"`` to ``"data:<n-1>"`` for n inputs,
    for each input, in the subnetwork.
    The first input will also be simply available as ``"data:data"``/``"data"`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'subnetwork',
      'from': source.get_name()}


class Variable(_Base):
  """
  Represents a variable. Can add batch/time dimension if wanted. Can be trainable.
  See defaults.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'variable',
      'from': source.get_name()}


class AccumulateMean(Reduce):
  """
  Accumulates the mean of the input (in training) (over batch-dim and time-dim by default).
  It's similar to :class:`ReduceLayer`
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'accumulate_mean',
      'from': source.get_name()}


class Loss(_Base):
  """
  This layers wraps a :class:`Loss` calculation as a layer.
  I.e. the loss will be calculated and returned by the layer.
  But this loss will not be used as a loss by the updater.
  If you want to use it as a loss, you can use the :class:`AsIsLoss`,
  i.e. write ``"loss": "as_is"``.

  Note that the loss options for the wrapped loss need to be provided via ``loss_opts_``,
  and it does not apply any reduce function.

  .. note::

    The ``LossLayer`` might be deprecated in the future in favor of implementing the losses as actual layers.

    If you want to define a loss inside the network, it is recommended to define it explicitly.
    An example could be:

    ``"se_loss": {"class": "eval", "eval": "(source(0) - source(1)) ** 2", "from": ["output", "data:classes"]}``

    Followed by an e.g. mean reduce if needed:

    ``"mse_loss": {"class": "reduce", "mode": "mean", "axis": "F", "from": "se_loss"}``


  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'loss',
      'from': source.get_name()}


class ForcedAlignment(_ConcatInput):
  """
  Calculates a forced alignment, via Viterbi algorithm.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'forced_align',
      'from': source.get_name()}


class FastBaumWelch(_ConcatInput):
  """
  Calls :func:`fast_baum_welch` or :func:`fast_baum_welch_by_sprint_automata`.
  We expect that our input are +log scores, e.g. use log-softmax.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'fast_bw',
      'from': source.get_name()}


class SyntheticGradient(_ConcatInput):
  """
  This is a generalized way to be able to replace the true gradient with any kind of predicted gradient.
  This enabled to implement the idea from here:
    Decoupled Neural Interfaces using Synthetic Gradients, https://arxiv.org/abs/1608.05343
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'synthetic_gradient',
      'from': source.get_name()}


class TikhonovRegularization(Copy):
  """
  Adds the Tikhonov regularization as a meta-loss (see :class:`TFUtil.MetaLosses`).
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'tikhonov_regularization',
      'from': source.get_name()}


class AllophoneStateIdxParser(_Base):
  """
  This is very much Sprint/RASR specific.
  We get allophone state indices and return (center, left_1, right_1, ..., state, boundary).
  The index is defined by NoTyingDense (ClassicStateTying.cc).
  In the Sprint config, this is via option --*.state-tying.type=no-tying-dense.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'allophone_state_idx_parser',
      'from': source.get_name()}


class FramewiseStatistics(_Base):
  """
  Collects various statistics (such as FER, etc) on the sources.
  The tensors will get stored in self.stats which will be collected by TFEngine.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'framewise_statistics',
      'from': source.get_name()}


class Print(_Base):
  """
  Prints the sources to console/log, via :func:`TFUtil.py_print`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'print',
      'from': source.get_name()}


class HDFDump(_Base):
  """
  Dumps into HDF file, compatible to :class:`HDFDataset`.

  The HDF will be written to disk under the specified filename, if there was no error,
  by default at graph reset, via :func:`TFNetwork.register_graph_reset_callback`.
  Or after the dataset iteration run loop, with dump_per_run,
  via :func:`TFNetwork.register_run_finished_callback`.

  Common usage would be to add this to your network with "is_output_layer": True,
  such that you don't need to make other layers depend on it.

  It currently uses :class:`SimpleHDFWriter` internally.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'hdf_dump',
      'from': source.get_name()}


class ImageSummary(_Base):
  """
  Creates image summaries which can be viewed in TensorBoard.
  This layer expects the source to be in (T-decoder, T-encoder, B, 1).
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'image_summary',
      'from': source.get_name()}


class OfficialResNet(_ConcatInput):
  """
  Wrapper around extern/official_tf_resnet.

  This operates on NHWC (batch, height, width, channel) data, and returns ND, where D = num_classes.
  If you have (batch, time, width, channel) as input,
  you probably want to use :class:`WindowLayer` to get (batch,time,window,width,channel),
  and then :class:`MergeDimsLayer` to get (batch*time,window,width,channel),
  such that we would interpret window = height here.
  Then the output is (batch*time,D),
  and you can use :class:`SplitBatchTimeLayer` to get (batch,time,D).
  As you get logits, you can then use :class:`ActivationLayer` with softmax.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'official_resnet',
      'from': source.get_name()}


class Rec(_ConcatInput):
  """
  Recurrent layer, has support for several implementations of LSTMs (via ``unit`` argument),
  see :ref:`tf_lstm_benchmark` (http://returnn.readthedocs.io/en/latest/tf_lstm_benchmark.html),
  and also GRU, or simple RNN.
  Via `unit` parameter, you specify the operation/model performed in the recurrence.
  It can be a string and specify a RNN cell, where all TF cells can be used,
  and the `"Cell"` suffix can be omitted; and case is ignored.
  Some possible LSTM implementations are (in all cases for both CPU and GPU):

   * BasicLSTM (the cell), via official TF, pure TF implementation
   * LSTMBlock (the cell), via tf.contrib.rnn.
   * LSTMBlockFused, via tf.contrib.rnn. should be much faster than BasicLSTM
   * CudnnLSTM, via tf.contrib.cudnn_rnn. This is experimental yet.
   * NativeLSTM, our own native LSTM. should be faster than LSTMBlockFused.
   * NativeLstm2, improved own native LSTM, should be the fastest and most powerful.

  We default to the current tested fastest one, i.e. NativeLSTM.
  Note that they are currently not compatible to each other, i.e. the way the parameters are represented.

  A subnetwork can also be given which will be evaluated step-by-step,
  which can use attention over some separate input,
  which can be used to implement a decoder in a sequence-to-sequence scenario.
  The subnetwork will get the extern data from the parent net as templates,
  and if there is input to the RecLayer,
  then it will be available as the "source" data key in the subnetwork.
  The subnetwork is specified as a `dict` for the `unit` parameter.
  In the subnetwork, you can access outputs from layers from the previous time step when they
  are referred to with the "prev:" prefix.

  Example::

      {
          "class": "rec",
          "from": "input",
          "unit": {
            # Recurrent subnet here, operate on a single time-step:
            "output": {
              "class": "linear",
              "from": ["prev:output", "data:source"],
              "activation": "relu",
              "n_out": n_out},
          },
          "n_out": n_out},
      }

  More examples can be seen in :mod:`test_TFNetworkRecLayer` and :mod:`test_TFEngine`.

  The subnetwork can automatically optimize the inner recurrent loop
  by moving layers out of the loop if possible.
  It will try to do that greedily. This can be disabled via the option `optimize_move_layers_out`.
  It assumes that those layers behave the same with time-dimension or without time-dimension and used per-step.
  Examples for such layers are :class:`LinearLayer`, :class:`RnnCellLayer`
  or :class:`SelfAttentionLayer` with option `attention_left_only`.

  This layer can also be inside another RecLayer. In that case, it behaves similar to :class:`RnnCellLayer`.
  (This support is somewhat incomplete yet. It should work for the native units such as NativeLstm.)

  Also see :ref:`recurrency`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'rec',
      'from': source.get_name()}


class _Template(_Base):
  """
  Used by _SubnetworkRecCell.
  In a first pass, it creates template layers with only the meta information about the Data.
  All "prev:" layers also stay instances of _TemplateLayer in the real computation graph.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': None,
      'from': source.get_name()}


class RecStepInfo(_Base):
  """
  Used by _SubnetworkRecCell.
  Represents the current step number.
  Usually via :func:`TFNetwork.set_rec_step_info`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': ':i',
      'from': source.get_name()}


class RnnCell(_ConcatInput):
  """
  Wrapper around tf.contrib.rnn.RNNCell.
  This will operate a single step, i.e. there is no time dimension,
  i.e. we expect a (batch,n_in) input, and our output is (batch,n_out).
  This is expected to be used inside a RecLayer.
  (But it can also handle the case to be optimized out of the rec loop,
   i.e. outside a RecLayer, with a time dimension.)
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'rnn_cell',
      'from': source.get_name()}


class GetLastHiddenState(_Base):
  """
  Will combine (concat or add or so) all the last hidden states from all sources.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'get_last_hidden_state',
      'from': source.get_name()}


class GetRecAccumulatedOutput(_Base):
  """
  For :class:`RecLayer` with a subnet.
  If some layer is explicitly marked as an additional output layer (via 'is_output_layer': True),
  you can get that subnet layer output via this accessor.
  Retrieves the accumulated output.

  Note that this functionality is obsolete now. You can simply access such an sub layer
  via the generic sub layer access mechanism. I.e. instead of::

    "sub_layer": {"class": "get_rec_accumulated", "from": "rec_layer", "sub_layer": "hidden"}

  You can do::

    "sub_layer": {"class": "copy", "from": "rec_layer/hidden"}
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'get_rec_accumulated',
      'from': source.get_name()}


class BaseChoice(_Base):
  """
  This is a base-class for any layer which defines a new search choice,
  i.e. which defines ``self.search_choices``.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': None,
      'from': source.get_name()}


class Choice(BaseChoice):
  """
  This layer represents a choice to be made in search during inference,
  such as choosing the top-k outputs from a log-softmax for beam search.
  During training, this layer can return the true label.
  This is supposed to be used inside the rec layer.
  This can be extended in various ways.

  We present the scores in +log space, and we will add them up along the path.
  Assume that we get input (batch,dim) from a (log-)softmax.
  Assume that each batch is already a choice via search.
  In search with a beam size of N, we would output
  sparse (batch=N,) and scores for each.

  In case of multiple sources, this layer computes the top-k combinations of choices. The score of such a combination
  is determined by adding up the (log-space) scores of the choices for the individual sources. In this case, the
  'target' parameter of the layer has to be set to a list of targets corresponding to the sources respectively. Because
  computing all possible combinations of source scores is costly, the sources are pruned beforehand using the beam
  sizes set by the 'source_beam_sizes' parameter. The choices made for the different sources can be accessed via the
  sublayers '<choice layer name>/out_0', '<choice layer name>/out_1' and so on.
  Note, that the way scores are combined assumes the sources to be independent. If you want to model a dependency,
  use separate ChoiceLayers and let the input of one depend on the output of the other.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'choice',
      'from': source.get_name()}


class Decide(BaseChoice):
  """
  This is kind of the counter-part to the choice layer.
  This only has an effect in search mode.
  E.g. assume that the input is of shape (batch * beam, time, dim)
  and has search_sources set.
  Then this will output (batch, time, dim) where the beam with the highest score is selected.
  Thus, this will do a decision based on the scores.
  In will convert the data to batch-major mode.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'decide',
      'from': source.get_name()}


class DecideKeepBeam(BaseChoice):
  """
  This just marks the search choices as decided, but does not change them (in contrast to :class:`DecideLayer`).
  You can use this to get out some values as-is, without having them resolved to the final choices.

  For internal usage only.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'decide_keep_beam',
      'from': source.get_name()}


class ChoiceGetBeamScores(_Base):
  """
  Gets beam scores from :class:`SearchChoices`.
  This requires that the source has search choices.

  .. note::

    This layer might be deprecated in the future.

  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'choice_get_beam_scores',
      'from': source.get_name()}


class ChoiceGetSrcBeams(_Base):
  """
  Gets source beam indices from :class:`SearchChoices`.
  This requires that the source has search choices.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'choice_get_src_beams',
      'from': source.get_name()}


class AttentionBase(_ConcatInput):
  """
  This is the base class for attention.
  This layer would get constructed in the context of one single decoder step.
  We get the whole encoder output over all encoder frames (the base), e.g. (batch,enc_time,enc_dim),
  and some current decoder context, e.g. (batch,dec_att_dim),
  and we are supposed to return the attention output, e.g. (batch,att_dim).

  Some sources:
  * Bahdanau, Bengio, Montreal, Neural Machine Translation by Jointly Learning to Align and Translate, 2015,
    https://arxiv.org/abs/1409.0473
  * Luong, Stanford, Effective Approaches to Attention-based Neural Machine Translation, 2015,
    https://arxiv.org/abs/1508.04025
    -> dot, general, concat, location attention; comparison to Bahdanau
  * https://github.com/ufal/neuralmonkey/blob/master/neuralmonkey/decoders/decoder.py
  * https://google.github.io/seq2seq/
    https://github.com/google/seq2seq/blob/master/seq2seq/contrib/seq2seq/decoder.py
    https://github.com/google/seq2seq/blob/master/seq2seq/decoders/attention_decoder.py
  * https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/attention.py
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': None,
      'from': source.get_name()}


class GlobalAttentionContextBase(AttentionBase):
  """
  Base class for other attention types, which use a global context.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': None,
      'from': source.get_name()}


class GenericAttention(AttentionBase):
  """
  The weighting for the base is specified explicitly here.
  This can e.g. be used together with :class:`SoftmaxOverSpatialLayer`.
  Note that we do not do any masking here. E.g. :class:`SoftmaxOverSpatialLayer` does that.

  Note that :class:`DotLayer` is similar, just using a different terminology.
  Reduce axis: weights: time-axis; base: time-axis.
    Note that if the last layer was :class:`SoftmaxOverSpatialLayer`, we should use the same time-axis.
    Also we should do a check whether these time axes really match.
  Common axes (should match): batch-axis, all from base excluding base feature axis and excluding time axis.
  Keep axes: base: feature axis; weights: all remaining, e.g. extra time.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'generic_attention',
      'from': source.get_name()}


class DotAttention(GlobalAttentionContextBase):
  """
  Classic global attention: Dot-product as similarity measure between base_ctx and source.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'dot_attention',
      'from': source.get_name()}


class ConcatAttention(GlobalAttentionContextBase):
  """
  Additive attention / tanh-concat attention as similarity measure between base_ctx and source.
  This is used by Montreal, where as Stanford compared this to the dot-attention.
  The concat-attention is maybe more standard for machine translation at the moment.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'concat_attention',
      'from': source.get_name()}


class GaussWindowAttention(AttentionBase):
  """
  Interprets the incoming source as the location (float32, shape (batch,))
  and returns a gauss-window-weighting of the base around the location.
  The window size is fixed (TODO: but the variance can optionally be dynamic).
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'gauss_window_attention',
      'from': source.get_name()}


class SelfAttention(_ConcatInput):
  """
  Applies self-attention on the input. I.e., with input `x`,
  it will basically calculate

      att(Q x, K x, V x),

  where `att` is multi-head dot-attention for now, `Q`, `K`, `V` are matrices.
  The attention will be over the time-dimension.
  If there is no time-dimension, we expect to be inside a :class:`RecLayer`;
  also, this is only valid with `attention_to_past_only=True`.

  See also `dot_product_attention` here:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'self_attention',
      'from': source.get_name()}


class PositionalEncoding(_ConcatInput):
  """
  Provides positional encoding in the form of (batch, time, n_out) or (time, batch, n_out)
  where n_out is the number of channels, if it is run outside a :class:`RecLayer`,
  and (batch, n_out) or (n_out, batch)
  if run inside a :class:`RecLayer`, where it will depend on the current time frame.

  Assumes one source input with a time dimension if outside a :class:`RecLayer`.
  With `add_to_input`, it will calculate `x + input`, and the output shape is the same as the input

  The positional encoding is the same as in Tensor2Tensor.
  See :func:`TFUtil.get_positional_encoding`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'positional_encoding',
      'from': source.get_name()}


class KenLmState(_ConcatInput):
  """
  Get next word (or subword) each frame,
  accumulates string,
  keeps state of seen string so far,
  returns score (+log space, natural base e) of sequence,
  using KenLM (http://kheafield.com/code/kenlm/) (see :mod:`TFKenLM`).
  EOS (</s>) token must be used explicitly.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'kenlm',
      'from': source.get_name()}


class EditDistanceTable(_Base):
  """
  Given a source and a target, calculates the edit distance table between them.
  Source can be inside a recurrent loop.
  It uses :func:`TFNativeOp.next_edit_distance_row`.

  Usually, if you are inside a rec layer, and "output" is the :class:`ChoiceLayer`,
  you would use "from": "output"
  and "target": "layer:base:data:target" (make sure it has the time dimension).

  See also :class:`OptimalCompletionsLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'edit_distance_table',
      'from': source.get_name()}


class OptimalCompletions(_Base):
  """
  We expect to get the inputs from :class:`EditDistanceTableLayer`, esp from the prev frame, like this:
  "opt_completions": {"class": "optimal_completions", "from": "prev:edit_dist_table"}.

  You can also then define this further layer:
  "opt_completion_soft_targets": {
    "class": "eval", "eval": "tf.nn.softmax(tf.cast(source(0), tf.float32))",
    "from": "opt_completions", "out_type": {"dtype": "float32"}},
  and use that as the :class:`CrossEntropyLoss` soft targets
  for the input of the "output" :class:`ChoiceLayer`, e.g. "output_prob".
  This makes most sense when you enable beam search (even, or esp, during training).
  Note that you probably want to have this all before the last choice, where you still have more beams open.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'optimal_completions',
      'from': source.get_name()}


class MaskedComputation(_Base):
  """
  Given some input [B,T,D] and some mask [B,T] (True or False), we want to perform a computation
  only on the masked frames.
  I.e. let T' be the max seq len of the masked seq, then the masked input would be [B,T',D].
  (This masked input sequence could be calculated via ``tf.boolean_mask`` or ``tf.gather_nd``.)
  The output is [B,T',D'], i.e. we do not undo the masking.
  You are supposed to use :class:`UnmaskLayer` to undo the masking.

  The computation also works within a rec layer, i.e. the input is just [B,D] and the mask is just [B].
  In that case, if the mask is True, it will perform the computation as normal,
  and if it is False, it will just copy the prev output, and also hidden state.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'masked_computation',
      'from': source.get_name()}


class Unmask(_Base):
  """
  This is meant to be used together with :class:`MaskedComputationLayer`,
  which operates on input [B,T,D], and given a mask, returns [B,T',D'].
  This layer :class:`UnmaskLayer` is supposed to undo the masking,
  i.e. to recover the original time dimension, i.e. given [B,T',D'], we output [B,T,D'].
  This is done by repeating the output for the non-masked frames,
  via the last masked frame.

  If this layer is inside a recurrent loop, i.e. we get [B,D'] as input,
  this is a no-op, and we just return the input as is.
  In that case, the repetition logic is handled via :class:`MaskedComputationLayer`.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'unmask',
      'from': source.get_name()}


class TwoDLSTM(_Base):
  """
  2D LSTM.

  Currently only from left-to-right in the time axis.
  Can be inside a recurrent loop, or outside.
  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'twod_lstm',
      'from': source.get_name()}


class RelativePositionalEncoding(_ConcatInput):
  """
  Relative positioning term as introduced by Shaw et al., 2018

  Usually added to Self-Attention using key_shift.
  Parts of the code are adapted from Tensor2Tensor (https://github.com/tensorflow/tensor2tensor).

  Example usage::

      d[output + '_rel_pos'] = {"class": "relative_positional_encoding",
                                "from": [output + '_self_att_laynorm'],
                                "n_out": self.EncKeyTotalDim // self.AttNumHeads,
                                "forward_weights_init": self.ff_init}
      d[output + '_self_att_att'] = {"class": "self_attention",
                                     "num_heads": self.AttNumHeads,
                                     "total_key_dim": self.EncKeyTotalDim,
                                     "n_out": self.EncValueTotalDim, "from": [output + '_self_att_laynorm'],
                                     "attention_left_only": False, "attention_dropout": self.attention_dropout,
                                     "forward_weights_init": self.ff_init, "key_shift": output + '_rel_pos'}

  """

  def __init__(self):
    super().__init__()

  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:
    """
    Make layer dict
    """
    return {
      'class': 'relative_positional_encoding',
      'from': source.get_name()}
