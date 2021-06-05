"""
Generic decoder interface.

This is supposed to cover the decoder of an attention-based encoder-decoder and of a transducer.
"""


class Decoder:
  """
  Generic decoder, for attention-based encoder-decoder or transducer.
  Can use label-sync label topology, or time-sync (RNA/CTC), or with vertical transitions (RNN-T).
  """
  pass

