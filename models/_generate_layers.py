"""
This will create ``_generated_layers.py``.

Originally the idea was to do it automatically,
and not keep the generated file under Git.
However, for now we make it explicit,
and we can manually explicitly call this.
"""

import os
import inspect
from returnn.tf.layers.base import LayerBase, InternalLayer

_my_dir = os.path.dirname(os.path.abspath(__file__))
_out_filename = f"{_my_dir}/_generated_layers.py"


def collect_layers():
  """
  Collect list of layers.
  """
  from returnn.tf.layers import base, basic, rec
  blacklist = set()
  ls = []
  for mod in [base, basic, rec]:
    for key, value in vars(mod).items():
      if isinstance(value, type) and issubclass(value, LayerBase):
        if value in blacklist:
          continue
        if issubclass(value, InternalLayer):
          continue
        ls.append(value)
        blacklist.add(value)
  return ls


def setup():
  """
  Setup
  """
  f = open(_out_filename, "w")
  f.write('"""This file is auto-generated."""\n')
  layer_classes = collect_layers()
  for layer_class in layer_classes:
    init_sig = inspect.signature(layer_class.__init__)
    print(layer_class, init_sig)


if __name__ == "__main__":
  setup()
