"""
Masked computation. Wrap :class:`MaskedComputationLayer` in RETURNN.

https://github.com/rwth-i6/returnn_common/issues/23

"""


class MaskedComputation:
  """
  This is expected to be inside a :class:`Loop`.

  Usage example::

      loop = nn.Loop(...)
      loop.state.y = ...  # some initial output
      loop.state.h = ...  # some initial state
      with loop:

        mask = ...  # dtype bool, shape [batch] or whatever, for current (fast) frame
        with nn.MaskedComputation(mask=mask) as masked_comp:
          loop.state.y, loop.state.h = slow_rnn(x, loop.state.h)
        y = loop.state.y  # access from outside

  This is equivalent to::

      loop = nn.Loop(...)
      loop.state.y = ...  # some initial output
      loop.state.h = ...  # some initial state
      with loop:

        mask = ...  # dtype bool, shape [batch] or whatever, for current frame
        y_, h_ = slow_rnn(x, loop.state.h)
        loop.state.y = nest.map(lambda a, b: nn.where(cond=mask, x=a, y=b), y_, loop.state.y)
        loop.state.h = nest.map(lambda a, b: nn.where(cond=mask, x=a, y=b), h_, loop.state.h)
        y = loop.state.y

  In pseudocode, non-batched (mask is just a scalar bool), it would look like::

      y = ...  # some initial output
      h = ...  # some initial state
      while True:

        mask = ...  # bool
        if mask:
          y, h = slow_rnn(x, h)

  """
