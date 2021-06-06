"""
This will create ``_generated_layers.py``.

Originally the idea was to do it automatically,
and not keep the generated file under Git.
However, for now we make it explicit,
and we can manually explicitly call this.
"""

import os
import inspect
from typing import Type
from returnn.tf.layers.base import LayerBase, InternalLayer

_my_dir = os.path.dirname(os.path.abspath(__file__))
_out_filename = f"{_my_dir}/_generated_layers.py"


def setup():
  """
  Setup
  """
  f = open(_out_filename, "w")
  print(format_multi_line_str("This file is auto-generated."), file=f)
  print("\nfrom .base import ILayerMaker, LayerRef, LayerDictRaw", file=f)
  layer_classes = collect_layers()
  for layer_class in layer_classes:
    init_sig = inspect.signature(layer_class.__init__)
    cls_str = get_module_class_name_for_layer_class(layer_class)
    cls_base = layer_class.__base__
    if issubclass(cls_base, LayerBase):
      cls_base_str = get_module_class_name_for_layer_class(cls_base)
    else:
      cls_base_str = "ILayerMaker"
    print(f"\n\nclass {cls_str}({cls_base_str}):", file=f)
    print(format_multi_line_str("Hello", indent="  "), file=f)
    print("", file=f)
    print("  def __init__(self):", file=f)
    print("    super().__init__()", file=f)
    print("", file=f)
    print("  def make_layer_dict(self, source: LayerRef, **kwargs) -> LayerDictRaw:", file=f)
    print(format_multi_line_str("Make layer dict", indent="    "), file=f)
    print("    return {", file=f)
    print(f"      'class': {layer_class.layer_class!r},", file=f)
    print("      'from': source.get_name()}", file=f)
    print(layer_class, cls_base, get_module_class_name_for_layer_class(layer_class), init_sig)


def format_multi_line_str(s: str, *, indent: str = "") -> str:
  """
  E.g. docstring etc
  """
  from io import StringIO
  ss = StringIO()
  print(indent + '"""', file=ss)
  for line in s.splitlines():
    print(indent + line, file=ss)
  print(indent + '"""', file=ss, end="")
  return ss.getvalue()


def get_module_class_name_for_layer_class(layer_class: Type[LayerBase]) -> str:
  """
  For some layer class, return the Module class name
  """
  if layer_class is LayerBase:
    return "_Base"
  assert layer_class.__name__.endswith("Layer")
  return layer_class.__name__[:-len("Layer")]


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


if __name__ == "__main__":
  setup()
