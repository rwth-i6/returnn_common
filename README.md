# `returnn_common`

This repo provides common building blocks for [RETURNN](https://github.com/rwth-i6/returnn/),
such as models or networks, network creation code,
datasets, etc.


# `nn`: Network definitions, models

RETURNN originally used dicts to define the network (model, computation graph).
The network consists of layers, where each layer represents a block of operations and potentially also parameters.
In here, we adopt many conventions by PyTorch or functional Keras and other frameworks,
such that you use pure Python code to define the network (model, computation graph).
Further, a module instance does not represent the actual computation
but only once you call it with actual inputs,
then it will perform the actual computation
(create a RETURNN layer, or the corresponding RETURNN layer dict).


## Usage examples

```python
from returnn_common import nn


class MyModelBlock(nn.Module):
  def __init__(self, dim: nn.Dim, hidden: nn.Dim, dropout: float = 0.1):
    super().__init__()
    self.layer_norm = nn.LayerNorm()
    self.linear_out = nn.Linear(dim)
    self.linear_hidden = nn.Linear(hidden)
    self.dropout = dropout

  def __call__(self, x: nn.Tensor) -> nn.Tensor:
    y = self.layer_norm(x)
    y = self.linear_hidden(y)
    y = nn.sigmoid(y)
    y = self.linear_out(y)
    y = nn.dropout(y, dropout=self.dropout, axis=[nn.batch_dim])
    return x + y
```

In case you want to have this three times separately now:
```python
class MyModel(nn.Module):
  def __init__(self, dim: nn.Dim):
    super().__init__()
    self.block1 = MyModelBlock(dim * 2, dim)
    self.block2 = MyModelBlock(dim * 2, dim)
    self.block3 = MyModelBlock(dim * 2, dim)
    
  def __call__(self, x: nn.Tensor) -> nn.Tensor:
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    return x
```

Or if you want to share the parameters but run this three times:
```python
class MyModel(nn.Module):
  def __init__(self, dim: nn.Dim):
    super().__init__()
    self.block = MyModelBlock(dim * 2, dim)
    
  def __call__(self, x: nn.Tensor) -> nn.Tensor:
    x = self.block(x)
    x = self.block(x)
    x = self.block(x)
    return x
```


# Installation and usage

When this is integrated as part of a Sisyphus recipe,
the common way people use it is similar as for i6_experiments,
i.e. you would `git clone` this repo into your `recipe` directory.

## Usage as Sisyphus recipe submodule

See [i6_experiments](https://github.com/rwth-i6/i6_experiments).

## Usage via `returnn.import_`

Earlier, this was intended to be used for the [RETURNN](https://github.com/rwth-i6/returnn) `returnn.import_` mechanism.
See [returnn #436 for initial `import_` discussions](https://github.com/rwth-i6/returnn/discussions/436).
See [#2 for discussions on `import_` usage here](https://github.com/rwth-i6/returnn_common/issues/2).
Note that this might not be the preferred usage pattern anymore but this is up to you.

Usage example for config:
```python
from returnn.import_ import import_
test = import_("github.com/rwth-i6/returnn_common", "test.py", "20210602-1bc6822")
print(test.hello())
```
You can also make use of auto-completion features in your editor (e.g. PyCharm).
Add `~/returnn/_pkg_import` to your Python paths,
and use this alternative code:
```python
from returnn.import_ import import_
import_("github.com/rwth-i6/returnn_common", ".", "20210602-1bc6822")
from returnn_import.github_com.rwth_i6.returnn_common.v20210302133012_01094bef2761 import test
print(test.hello())
```

During development of a new feature in `returnn-experiments`,
you would use a special `None` placeholder for the version,
such that you can directly work in the checked out repo.
The config code looks like this:
```python
from returnn.import_ import import_
import_("github.com/rwth-i6/returnn_common", ".", None)
from returnn_import.github_com.rwth_i6.returnn_common.dev import test
print(test.hello())
```

You would also edit the code in `~/returnn/pkg/...`,
and once finished, you would commit and push to `returnn_common`,
and then change the config to that specific version (date & commit).


# Code principles

These are the ideas behind the recipes.
If you want to contribute, please try to follow them.
(If something is unclear, or even in general,
better speak with someone before you do changes, or add something.)

## Simplicity

This is supposed to be **simple**.
Functions or classes can have some options
with reasonable defaults.
This should not become too complicated.
E.g. a function to return a Librispeech corpus
should not be totally generic to cover every possible case.
When it doesn't fit your use case,
instead of making the function more complicated,
just provide your alternative `LibrispeechCustomX` class.
There should be reasonable defaults.
E.g. just `Librispeech()` will give you some reasonable dataset.

E.g. maybe ~5 arguments per function is ok
(and each argument should have some good default),
but it should not be much more.
**Better just make separate functions instead
even when there is some amount of duplicate code**
(`make_transformer` which creates a standard Transformer,
vs `make_linformer` which creates a Linformer, etc.).

## Building blocks

It should be simple to use functions
as basic building blocks to build sth more complex.
E.g. when you implement the Transformer model
(put that to `models/segmental/transformer.py`)
make functions `make_trafo_enc_block`
and `make_trafo_encoder(num_layers=..., ...)`
in `models/encoder/transformer.py`,
and then `make_transformer_decoder` and `make_transformer`
in `models/segmental/transformer.py`.
That makes **parts of it easily reusable**.
**Break it down** as much as it is reasonable.

## Code dependencies

The building blocks will naturally depend on each other.
In most cases, you should use **relative imports**
to make use of other building blocks,
and **not `import_`**.

## Data dependencies

Small files (e.g. vocabularies up to a certain size <100kb or so)
could be directly put to the repository next to the Python files.
This should be kept minimal and only be used for the most common files.
(E.g. our Librispeech BPE vocab is stored.)
The repository should stay small,
so try to avoid this if this is not really needed.

For any larger files or other files,
the idea is that this can easily be used across different systems.
So there would be a common directory structure
in some directory which could be some symlinks elsewhere.
(We could also provide some scripts to simplify handling this.)
To refer to such a file path, use the functions in `data.py`.


# Requirements

Python 3.7+.
See [#43](https://github.com/rwth-i6/returnn_common/issues/43).

Recent RETURNN (>=2022), needs behavior version >=12.
