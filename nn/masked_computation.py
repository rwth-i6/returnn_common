"""
Masked computation. Wrap :class:`MaskedComputationLayer` in RETURNN.

https://github.com/rwth-i6/returnn_common/issues/23

"""


class MaskedComputation:
  """
  This is expected to be inside a :class:`Loop`.

  Example without nn.MaskedComputation::

      loop = nn.Loop(...)
      loop.state.y = ...  # some initial output
      loop.state.h = ...  # some initial state
      with loop:

        mask = ...  # dtype bool, shape [batch] or whatever, for current frame
        y, loop.state.h = slow_rnn(x, loop.state.h)
        y = nn.where(cond=mask, x=y, y=loop.state.y)
        loop.state.y = y

  Is equivalent to::

      loop = nn.Loop(...)
      loop.state.y = ...  # some initial output
      loop.state.h = ...  # some initial state
      with loop:

        mask = ...  # dtype bool, shape [batch] or whatever, for current (fast) frame
        with nn.MaskedComputation(mask=mask) as masked_comp:
          y, loop.state.h = slow_rnn(x, loop.state.h)
        y, loop.state.y = masked_comp.unmask(y, loop.state.y)  # access from outside

  """
