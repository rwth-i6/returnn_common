"""
Utilities for search or beam search in RETURNN.
See the :class:`ChoiceLayer` or :func:`choice` as the core of the search.
"""

from .. import nn


class SearchFuncInterface:
  """
  Interface for search.
  """

  def get_beam(self) -> nn.SearchBeam:
    """
    Return a new beam instance. dependency and name is still unset and will be set outside.
    This overwrites whatever is returned by :func:`choice`,
    and copy_as_prev_frame() is used to set the initial (prev) state beam.
    """
    raise NotImplementedError

  def choice(self, *, output: nn.Tensor, output_type: str) -> nn.Tensor:
    """
    Given an output tensor (logits or log prop), returns a beam of chosen indices.
    This is the core of the search.

    :param output: The output tensor (logits or log prob or prob).
    :param output_type: "logits" or "log_prob" or "prob".
    """
    raise NotImplementedError

  def apply_loop(self, loop: nn.Loop):
    """
    Called when the choice is being done in a loop.
    E.g. you could set loop.max_seq_len here.
    """
    pass  # nothing by default


class SearchFunc(SearchFuncInterface):
  """Via nn.choice"""
  def __init__(self, beam_size: int, max_seq_len: nn.Tensor):
    self.beam_size = beam_size
    self.max_seq_len = max_seq_len

  def get_beam(self):
    """beam"""
    return nn.SearchBeam(beam_size=self.beam_size)

  def choice(self, *, output: nn.Tensor, output_type: str) -> nn.Tensor:
    """nn.choice"""
    return nn.choice(
      output, input_type=output_type, beam_size=self.beam_size,
      target=None, search=True)

  def apply_loop(self, loop: nn.Loop):
    """set max_seq_len"""
    loop.max_seq_len = self.max_seq_len
