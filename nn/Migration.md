This is a small migration guide, from converting a raw RETURNN network dictionary to the `nn` framework.

If you do not find sth, or find sth confusing,
the reason might be that this is just incomplete,
so please give feedback, and maybe extend the framework!

For further documentation, see:

* [RETURNN common homepage](https://github.com/rwth-i6/returnn_common) has an introduction and small usage examples
* [RETURNN common principles](https://github.com/rwth-i6/returnn_common/wiki/RETURNN-common-principles)
* [RETURNN common conventions](https://github.com/rwth-i6/returnn_common/blob/main/nn/Conventions.md).
* Docstrings in the code. It is anyway very recommended to use an IDE to be able to use auto-completion, and the IDE would also automatically show you the documentation.
* [`nn.base` docstring](https://github.com/rwth-i6/returnn_common/blob/main/nn/base.py). `nn.base` defines many important base classes such as `nn.Tensor`, `nn.Module`, and has some high-level explanation of how it works internally
* [`nn.naming` docstring](https://github.com/rwth-i6/returnn_common/blob/main/nn/naming.py). `nn.naming` defines the layer names and parameter names, i.e. how a model (via `nn.Module`) and all intermediate computations map to RETURNN layers.
* [Missing pieces for first release](https://github.com/rwth-i6/returnn_common/issues/32) and [Intermediate usage before first release](https://github.com/rwth-i6/returnn_common/issues/98). This is an overview, also linking to now completed issues. The issues often come with some discussion where you find the rationality behind certain design decisions.
* [Translating TF or PyTorch or other code to returnn_common](https://github.com/rwth-i6/returnn_common/wiki/Translating-TF-or-PyTorch-or-other-code-to-returnn_common)

For many aspects and design decision, RETURNN common follows the PyTorch API.


## Setup

How to define or create the config,
how to write Sisyphus setups, etc.:
There is no definite, recommended way yet.
We are still figuring out what the nicest way is.
It's also up to you.
It's very flexible and basically allows
to do it in any way you want.

You could have the `nn` code to define the network
directly in the config,
instead of the net dict.

You can also dump a generated net dict
and put that into the config.
However, the generated net dict tends to be quite big,
closer to the TF computation graph.
So, to better understand the model definition
and be able to easily extend or change it
for one-off experiments,
it is recommended to always keep the `nn` code around,
and maybe not dump the generated net dict at all.

Next to the network (`network`),
you also directly should define
the extern data (`extern_data`).

Also see [How to handle Sisyphus hashes](https://github.com/rwth-i6/returnn_common/issues/51)
and [Extensions for Sisyphus serialization of net dict and extern data](https://github.com/rwth-i6/returnn_common/issues/104).

(Speak to Nick, Benedikt, Albert, or others about examples, but note that they are all work-in-progress. You probably find some example code in [i6_experiments](https://github.com/rwth-i6/i6_experiments/).)


## Inputs / outputs

The outputs of layers
as inputs to other layers
are referred to via their names as strings
in the net dict,
which are defined by the keys of the net dict.

In `nn`, the output of a layer/module/function
is of type `nn.Tensor`,
and this is also the input type for everything.


## Layers with parameters

`LinearLayer` etc.
-> `nn.Linear` etc., classes, derive from `nn.Module`


## Layers without parameters

Functional, e.g. `ReduceLayer`
-> `nn.reduce` etc., pure functions


## Simple arithmetic and comparisons

`CombineLayer`
-> `a + b` etc. works directly

`CompareLayer` -> `a >= b` etc. works directly


## Common math functions

Via `ActivationLayer`
-> `nn.relu` etc. works directly


## Layers with hidden state

`RecLayer`, `SelfAttentionLayer` etc.

There is no hidden state in `nn`, it is all explicit.
The `nn.LayerState` object is used to pass around state.
See `nn.LSTM` for an example.
`nn.LSTM` can operate both on a sequence or on a single frame when you pass `axis=nn.single_step_dim`.


## Loops

`RecLayer` with subnetwork
-> `nn.Loop`


## Conditions

`CondLayer`
-> `nn.Cond`


## Masked computation

`MaskedComputationLayer`
-> `nn.MaskedComputation`


## Subnetworks

`SubnetworkLayer`
-> define your own module (class, derived from `nn.Module`)


## ChoiceLayer / search

`ChoiceLayer` -> `nn.choice`.
However, also see `nn.SearchFunc` and `nn.Transformer` as an example.


## Attention

`SelfAttentionLayer` -> `nn.SelfAttention` or `nn.CausalSelfAttention`.

In general, there is also `nn.dot_attention`.

For more complete networks, there is also `nn.Transformer`, `nn.TransformerEncoder`, `nn.TransformerDecoder`, or `nn.Conformer`.


## EvalLayer / custom TF code

It should be straight-forward
to translate custom TF code
directly to `nn` code,
mostly just by replacing `tf.` with `nn.`.
See [our SpecAugment code](https://github.com/rwth-i6/returnn_common/blob/main/asr/specaugment.py) as an example (`specaugment_v1_eval_func` vs `specaugment_v2`).


## Wrapping custom layer dicts

You can use `nn.make_layer` to wrap a custom layer dict.
This can be used to partially migrate over some network definition.
However, it is recommended to avoid this and rewrite the model definition using the `nn` framework directly.
`nn.make_layer` is how `nn` works internally.


## Dimensions

After [some discussion](https://github.com/rwth-i6/returnn_common/issues/17),
it was decided to make consistent use of dimension tags (`DimensionTag`, or `nn.Dim`),
and not allow anything else to specify dimensions or axes.

The concept of axes and the concept of dimensions and dimension values (e.g. output feature dimension of `LinearLayer`) is the same when dim tags are used consistently.

The feature dimension is still treated special in some cases,
meaning it is automatically used when not specified,
via the attrib `feature_dim` of a tensor.
However, all spatial dims (or reduce dims, etc.) always need to be specified explicitly.
All non-specified dimensions are handled as batch dimensions.

Before:
```python
"y": {"class": "linear", "from": "x", "n_out": 512, "activation": None}
```
After:
```python
linear_module = nn.Linear(out_dim=nn.FeatureDim("linear", 512))
y = linear_module(x)
```

On getting the length or dim value as a tensor:
`nn.length(x, axis=axis)`
or `nn.dim_value(x, axis=axis)`.


## Parameters

All parameters (variables) are explicit in `nn`,
meaning that no RETURNN layer will create a variable
but all variables are explicitly created by the `nn` code
via creating `nn.Parameter`.

Parameters must have a unique name from the root module via an attrib chain.

Parameter initial values can be assigned via the `initial` attribute, and the `nn.init` module provides common helpers such as `nn.init.Glorot`.
Modules (`nn.Linear`, `nn.Conv` etc) should already set a sensible default init,
but this can then be easily overwritten.

You can iterate through all parameters of a module or network
via `parameters()` or `named_parameters()`.

Also see [How to define the API for parameter initialization, regularization (L2, weight dropout, etc), maybe updater opts per-param](https://github.com/rwth-i6/returnn_common/issues/59).


## Losses

RETURNN differentiates between layer classes
(derived from `LayerBase`)
and loss classes (derived from `Loss`).

`nn` does not.
`nn.cross_entropy` is just a normal function,
getting in tensors, returning a tensor.

To mark it as a loss, such that it is used for training,
you call `mark_as_loss` on the tensor.

### CTC

The normalization behaves different between RETURNN `CtcLoss` and `nn.ctc_loss`,
and actually probably not as intended for the RETURNN `CtcLoss`.
See [here](https://github.com/rwth-i6/returnn/issues/1077#issuecomment-1184929542) for details.

To get the same behavior as before:
```python
ctc_loss = nn.ctc_loss(...)
ctc_loss.mark_as_loss("ctc", custom_inv_norm_factor=nn.length(targets_time_dim))
```


## L2 / weight decay

`L2` parameter in a layer
-> `weight_decay` attrib of `nn.Parameter`

By iterating over `paramaters()`,
you can easily assign the same weight decay
to all parameters, or a subset of your model.
