
"""
Gammatone feature extraction.

Code by Peter Vieting, and adopted.
"""


import tensorflow as tf
import numpy
from returnn.tf.network import TFNetwork, ExternData
from returnn.tf.util.data import FeatureDim

tf1 = tf.compat.v1


class Extractor:
  """
  Extractor
  """
  def __init__(self, **kwargs):
    net_dict = get_net_dict(**kwargs)
    with tf1.Graph().as_default() as self.graph:
      self.extern_data = ExternData({"data": {"shape": (None, 1)}})
      self.input = self.extern_data.data["data"]
      self.net = TFNetwork(name="Gammatone", extern_data=self.extern_data)
      self.net.construct_from_dict(net_dict)
      self.output = self.net.get_default_output_layer().output.copy_as_batch_major()
      self.session = tf1.Session(graph=self.graph, config=tf1.ConfigProto(device_count=dict(GPU=0)))
      self.session.run(tf1.global_variables_initializer())

  def run(self, audio, seq_lens=None):
    """
    :param numpy.ndarray audio: shape [B,T,1] (T being raw samples, 16kHz)
    :param numpy.array|None seq_lens: shape [B]. if not given, assume [T]*B
    :return: features, feat_seq_lens
    :rtype: (numpy.ndarray,numpy.ndarray)
    """
    assert len(audio.shape) == 3 and audio.shape[-1] == 1
    b, t, _ = audio.shape
    if seq_lens is None:
      seq_lens = [t] * b
    return self.session.run(
      (self.output.placeholder, self.output.size_placeholder[0]),
      feed_dict={self.input.placeholder: audio, self.input.size_placeholder[0]: seq_lens})


_extractor = None


def _extract(*, audio, num_feature_filters, sample_rate, **_other):
  assert sample_rate == 16000
  global _extractor
  if not _extractor:
    _extractor = Extractor(num_channels=num_feature_filters)
  features, _ = _extractor.run(audio[numpy.newaxis, :, numpy.newaxis])
  return features[0]


def make_returnn_audio_features_func():
  """
  This can be used for ExtractAudioFeatures in RETURNN,
  e.g. in OggZipDataset or LibriSpeechCorpus or others.
  """
  return _extract


def get_net_dict(
    num_channels=50, sample_rate=16000, gt_filterbank_size=640, temporal_integration_size=400,
    temporal_integration_strides=160, normalization="batch", freq_max=7500., source="data",
):
  """
  :param int num_channels: gammatone feature output dimension
  :param int|float sample_rate: sampling rate of input waveform
  :param int gt_filterbank_size: size of gammatone filterbank (in samples)
  :param int temporal_integration_size: size of filter for temporal integration
  :param int temporal_integration_strides: strides of filter for temporal integration
  :param str|None normalization: batch, time or None
  :param float freq_max: maximum frequency of Gammatone filterbank
  :param str source: source layer name
  """
  gammatone_feature_dim = FeatureDim("gammatone_feature_dim", num_channels)
  gammatone_split_dummy_dim = FeatureDim("gammatone_split_dummy_dim", 1)
  net_dict = {
    'shift_0': {'class': 'slice', 'axis': 'T', 'slice_end': -1, 'from': source},
    'shift_1_raw': {'class': 'slice', 'axis': 'T', 'slice_start': 1, 'from': source},
    'shift_1': {'class': 'reinterpret_data', 'from': 'shift_1_raw', 'set_axes': {'T': 'time'}, 'size_base': 'shift_0'},
    'preemphasis': {'class': 'combine', 'from': ['shift_1', 'shift_0'], 'kind': 'sub'},
    'gammatone_filterbank': {
      'class': 'conv',
      'activation': 'abs',
      'filter_size': (gt_filterbank_size,),
      'forward_weights_init': {
        'class': 'GammatoneFilterbankInitializer',
        'num_channels': num_channels,
        'length': gt_filterbank_size / sample_rate,
        'sample_rate': sample_rate,
        'freq_max': freq_max},
      'from': 'preemphasis',
      'n_out': num_channels,
      'in_spatial_dims': 'T',
      'padding': 'valid'},
    'gammatone_filterbank_split': {
      'class': 'split_dims',
      'axis': 'F',
      'dims': (gammatone_feature_dim, gammatone_split_dummy_dim),
      'from': 'gammatone_filterbank'},
    'temporal_integration': {
      'class': 'conv',
      'filter_size': (temporal_integration_size, 1),
      'forward_weights_init': 'numpy.hanning({}).reshape(({}, 1, 1, 1))'.format(
        temporal_integration_size, temporal_integration_size),
      'from': 'gammatone_filterbank_split',
      'n_out': 1,
      'padding': 'valid',
      'strides': (temporal_integration_strides, 1),
      'in_spatial_dims': ["T", gammatone_feature_dim],
      'out_dim': gammatone_split_dummy_dim},
    'temporal_integration_merge': {
      'class': 'merge_dims',
      'axes': [gammatone_feature_dim, gammatone_split_dummy_dim],
      'from': 'temporal_integration'},
    'compression': {
      'class': 'eval',
      'eval': 'tf.pow(source(0) + 1e-06, 0.1)',
      'from': 'temporal_integration_merge'},
    'dct': {'class': 'dct', 'from': 'compression'},
  }
  if normalization is None:
    net_dict['output'] = {'class': 'copy', 'from': 'dct'}
  elif normalization == 'batch':
    net_dict['output'] = {'class': 'batch_norm', 'from': 'dct'}
  elif normalization == 'time':
    net_dict['output'] = {'class': 'norm', 'axes': 'T', 'from': 'dct'}
  else:
    raise NotImplementedError
  return net_dict
