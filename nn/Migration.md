This is a small migration guide, from converting a raw RETURNN network dictionary to the `nn` framework.

If you do not find sth, or find sth confusing,
the reason might be that this is just incomplete,
so please give feedback, and maybe extend the framework!


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


## Simple arithmetic

`CombineLayer` etc.
-> `a + b` etc. works directly


## Common math functions

Via `ActivationLayer`
-> `nn.relu` etc. works directly


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
ctc_loss.mark_as_loss(custom_inv_norm_factor=nn.length(targets, axis=targets_time_dim))
```


## Constraints (L2 etc.)
