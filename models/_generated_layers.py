"""
This file is auto-generated.
"""

from .base import ILayerMaker, LayerRef, LayerDictRaw


class _Base(ILayerMaker):
  """
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
  Hello
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
