# Conventions

Code and pattern conventions.

Also see [the RETURNN migration guide](https://github.com/rwth-i6/returnn_common/blob/main/nn/Migration.md)
and [Translating TF or PyTorch or other code to RETURNN-common](https://github.com/rwth-i6/returnn_common/wiki/Translating-TF-or-PyTorch-or-other-code-to-returnn_common).


## `nn.Module` vs function

Once it has trainable parameters (`nn.Parameter`),
it should be a class, deriving from `nn.Module`.
Otherwise, it should be a function.

Examples:

* `nn.Linear`
* `nn.relu`
* `nn.SelfAttention`
* `nn.dot_attention`
* `nn.dropout`


## Recurrent state

All state is explicit
([discussion](https://github.com/rwth-i6/returnn_common/issues/31)).
Functions or modules would get a `state: nn.LayerState` argument
for the previous state
and return a `nn.LayerState` for the new state.

Modules would provide the `default_initial_state` method,
which should return a `nn.LayerState`
with a reasonable default initial state
for the recurrent `__call__` method.

TODO what if it is just a function?
A separate `..._default_initial_state` function?

Examples:

* `nn.LSTM`
* `nn.rec_cum_sum`
* `nn.rec_cum_concat`
* `nn.CausalSelfAttention`


## Stepwise vs sequence operation

E.g. [`torch.nn.LSTMCell`](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell)
vs [`torch.nn.LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

In RETURNN-common, that is a unified interface,
where a function in any case gets an `axis` (or `spatial_dim` or so) argument,
and there is the special `nn.single_step_dim` to indicate that it should operate on a single step.
In both case, it would get and return a `nn.LayerState` object.
([Discussion](https://github.com/rwth-i6/returnn_common/issues/81).)

Examples:

* `nn.LSTM`
* `nn.rec_cum_sum`
* `nn.rec_cum_concat`
* `nn.CausalSelfAttention`


## Operations on an axis or dimension

Such functions get an `axis: nn.Dim` or `spatial_dim: nn.Dim` argument.
If they generate a new axis, they should return it as well.
In that case, the input is often called `in_spatial_dim`,
and there might be an optional `out_spatial_dim`,
which can be predefined, or otherwise is automatically created
(and in any case returned then).

Examples:

* `nn.Conv1d` etc
* `nn.pool1d` etc
* `nn.reduce`
* `nn.dropout`
* `nn.rec_cum_concat`
