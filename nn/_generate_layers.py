"""
This will create ``_generated_layers.py``.

Originally the idea was to do it automatically,
and not keep the generated file under Git.
However, for now we make it explicit,
and we can manually explicitly call this.
"""

from __future__ import annotations
import os
import inspect
import re
from typing import Type, Optional, Dict, List
import returnn
from returnn.util import better_exchook
from returnn.util.basic import camel_case_to_snake_case
from returnn.tf.layers.base import LayerBase, InternalLayer
# noinspection PyProtectedMember
from returnn.tf.layers.basic import _ConcatInputLayer, SourceLayer
from returnn.tf.layers.basic import CombineLayer, CompareLayer, StackLayer, DropoutLayer
from returnn.tf.layers.basic import LinearLayer, ConvLayer, TransposedConvLayer
from returnn.tf.layers.basic import ConstantLayer, VariableLayer, CondLayer, SwitchLayer, SubnetworkLayer
from returnn.tf.layers.rec import RecLayer, RnnCellLayer
from returnn.tf.layers.rec import PositionalEncodingLayer, RelativePositionalEncodingLayer
from returnn.tf.layers.rec import BaseChoiceLayer, ChoiceLayer, GenericAttentionLayer

_my_dir = os.path.dirname(os.path.abspath(__file__))
_out_filename = f"{_my_dir}/_generated_layers.py"


# We use blacklists instead of whitelists such that we can more easily run this script in the future.

# These layers are deprecated or not needed for various reasons, and thus exclude them.
# Some of them are also very easily reproduced by other layers and thus not needed.
# If you think some of these are needed, or you are unsure how to get the corresponding functionality,
# please open an issue.
BlacklistLayerClassNames = {
  "_ConcatInputLayer",  # we don't do automatic concat, https://github.com/rwth-i6/returnn_common/issues/41
  "DropoutLayer",  # we do that manually

  "RecStepInfoLayer",
  "_TemplateLayer",
  "cond", "masked_computation", "subnetwork",

  "source",  # we have get_extern_data instead
  "swap_axes",
  "gather_nd",  # -> gather
  "softmax",  # misleading (because not just activation), also we will have a separate softmax activation
  "gating",
  "expand_dims",  # not sure if this is ever needed
  "weighted_sum",
  "elemwise_prod",
  "combine_dims",  # -> merge_dims
  "loss",
  "transpose",
  "accumulate_mean",
  "framewise_statistics",
  "image_summary",
  "get_rec_accumulated",  # covered by our Loop logic
  "decide_keep_beam",  # internal
  "rnn_cell",  # -> rec
  "AttentionBaseLayer",
  "GlobalAttentionContextBaseLayer",
  "generic_attention",  # -> dot
  "dot_attention",  # -> dot
  "concat_attention",
  "gauss_window_attention",
  "self_attention",
}

LayersHidden = {
  "combine",  # only needed as base
  "split",
  "get_last_hidden_state",  # we handle all state explicitly, there is no hidden state. this is only needed internally
}

BlacklistLayerArgs = {
  "range_in_axis": {"unbroadcast", "keepdims"},
}

FunctionNameMap = {
  "source": "external_data",  # but not used actually
  "norm": "normalize",
  "softmax_over_spatial": "softmax",  # generic also for normal softmax on feature
}


def setup():
  """
  Setup
  """
  print("RETURNN:", returnn.__long_version__, returnn.__file__)
  f = open(_out_filename, "w", newline='')
  print('"""', file=f)
  print(f"This file is auto-generated by {os.path.basename(__file__)}.", file=f)
  print(f"RETURNN:", returnn.__long_version__, file=f)
  print("", file=f)
  print("These are the RETURNN layers directly wrapped.", file=f)
  print("Note that we intentionally exclude some layers or options for more consistency.", file=f)
  print("Please file an issue if you miss something.", file=f)
  print('"""', file=f)
  print("", file=f)
  print("from __future__ import annotations", file=f)
  print("from typing import Union, Optional, Tuple, List, Dict, Any", file=f)
  print("from returnn.util.basic import NotSpecified", file=f)
  print("from returnn.tf.util.basic import DimensionTag", file=f)
  print("from .base import NameCtx, ILayerMaker, _ReturnnWrappedLayerBase, Layer, LayerRef, LayerDictRaw", file=f)
  layer_classes = collect_layers()
  signatures = {}  # type: Dict[Type[LayerBase], LayerSignature]
  for layer_class in layer_classes:
    sig = LayerSignature(layer_class, signatures)
    signatures[layer_class] = sig
    cls_str = get_module_class_name_for_layer_class(sig)
    if layer_class != LayerBase:
      cls_base_str = get_module_class_name_for_layer_class(sig.derived_layer())
    else:
      cls_base_str = "_ReturnnWrappedLayerBase"

    print(f"\n\nclass {cls_str}({cls_base_str}):", file=f)
    if layer_class.__doc__:
      print('  """', end="", file=f)
      for line in layer_class.__doc__.splitlines(keepends=True):
        print(line if line.strip() else line.strip(" "), end="", file=f)
      print('  """', file=f)
    else:
      print(format_multi_line_str("(undocumented...)", indent="  "), file=f)

    print(f"  returnn_layer_class = {sig.layer_class.layer_class!r}", file=f)
    print(f"  has_recurrent_state = {sig.has_recurrent_state()}", file=f)
    print(f"  has_variables = {sig.has_variables()}", file=f)

    if sig.need_module_init():
      print("", file=f)
      print("  # noinspection PyShadowingBuiltins,PyShadowingNames", file=f)
      print("  def __init__(self,", file=f)
      printed_keyword_only_symbol = False
      for _, param in sig.params.items():
        if param.is_module_init_arg():
          if not printed_keyword_only_symbol and param.inspect_param.kind == param.inspect_param.KEYWORD_ONLY:
            print("               *,", file=f)
            printed_keyword_only_symbol = True
          print(f"               {param.get_module_param_code_str()},", file=f)
      print(f"               {'**kwargs' if layer_class != LayerBase else ''}):", file=f)
      print('    """', file=f)
      if sig.docstring:
        for line in sig.docstring.splitlines():
          print(("    " + line) if line else "", file=f)
        print("", file=f)
      for _, param in sig.params.items():
        if param.is_module_init_arg():
          print(param.get_module_param_docstring(indent="    "), file=f)
      print('    """', file=f)
      print(f"    {sig.get_init_super_call_code_str()}", file=f)
      for _, param in sig.params.items():
        if param.is_module_init_arg():
          print(f"    self.{param.get_module_param_name()} = {param.get_module_param_name()}", file=f)

    if sig.has_module_init_args() or sig.has_defined_base_params():
      print("", file=f)
      print("  def get_opts(self):", file=f)
      print(format_multi_line_str("Return all options", indent="    "), file=f)
      print("    opts = {", file=f)
      for _, param in sig.params.items():
        if param.is_module_init_arg():
          print(f"      '{param.returnn_name}': self.{param.get_module_param_name()},", file=f)
      print("    }", file=f)
      print("    opts = {key: value for (key, value) in opts.items() if value is not NotSpecified}", file=f)
      if layer_class != LayerBase:
        if sig.has_defined_base_params():
          print("    opts.update(super().get_opts())", file=f)
          for key in sig.get_defined_base_params():
            print(f"    opts.pop({key!r})", file=f)
          print("    return opts", file=f)
        else:
          print("    return {**opts, **super().get_opts()}", file=f)
      else:
        print("    return opts", file=f)

    if layer_class.layer_class:
      print("", file=f)
      if sig.has_source_param() or sig.has_recurrent_state() or sig.has_module_call_args():
        print("  # noinspection PyShadowingBuiltins,PyShadowingNames", file=f)
        print("  def make_layer_dict(self,", file=f)
        if sig.has_source_param():
          print(f"                      {sig.get_module_call_source_param_code_str()},", file=f)
        if sig.has_module_call_args() or sig.has_recurrent_state():
          print("                      *,", file=f)
          if sig.has_recurrent_state():
            print(f"                      {sig.get_module_call_state_param_code_str('state')},", file=f)
            print(f"                      {sig.get_module_call_state_param_code_str('initial_state')},", file=f)
          for param in sig.get_module_call_args():
            print(f"                      {param.get_module_param_code_str()},", file=f)
        print("                      ) -> LayerDictRaw:",  file=f)
      else:
        print("  def make_layer_dict(self) -> LayerDictRaw:", file=f)
      print(format_multi_line_str("Make layer dict", indent="    "), file=f)
      if sig.has_source_param():
        if sig.need_multiple_sources():
          print(
            "    assert isinstance(source, (tuple, list)) and all(isinstance(s, LayerRef) for s in source)",
            file=f)
        elif sig.support_multiple_sources():
          print(
            "    assert (\n"
            "      isinstance(source, LayerRef) or\n"
            "      (isinstance(source, (tuple, list)) and all(isinstance(s, LayerRef) for s in source)))",
            file=f)
        else:
          print("    assert isinstance(source, LayerRef)", file=f)
      if sig.has_module_call_args() or sig.has_recurrent_state():
        print("    args = {", file=f)
        if sig.has_recurrent_state():
          print(f"      'state': state,", file=f)
          print(f"      'initial_state': initial_state,", file=f)
        for param in sig.get_module_call_args():
          print(f"      '{param.returnn_name}': {param.get_module_param_name()},", file=f)
        print("    }", file=f)
        print("    args = {key: value for (key, value) in args.items() if value is not NotSpecified}", file=f)
      print("    return {", file=f)
      print(f"      'class': {layer_class.layer_class!r},", file=f)
      if sig.has_source_param():
        print("      'from': source,", file=f)
      if sig.has_module_call_args() or sig.has_recurrent_state():
        print("      **args,", file=f)
      print("      **self.get_opts()}", file=f)
    else:
      print("", file=f)
      print("  make_layer_dict = ILayerMaker.make_layer_dict  # abstract", file=f)

    # Make function if this is functional
    name = get_module_class_name_for_layer_class(sig)
    if sig.is_functional() and not layer_class.__name__.startswith("_") and layer_class.layer_class:
      module_name = name
      name = camel_case_to_snake_case(name.lstrip("_"))
      if name in FunctionNameMap:
        name = FunctionNameMap[name]
      if sig.layer_class.layer_class in LayersHidden:
        name = "_" + name
      print("\n", file=f)
      print("# noinspection PyShadowingBuiltins,PyShadowingNames", file=f)
      prefix = f"def {name}("
      print(f"{prefix}", file=f)
      prefix = " " * len(prefix)
      args = []
      if sig.has_source_param():
        print(f"{prefix}{sig.get_module_call_source_param_code_str()},", file=f)
        args.append("source")
      print(f"{prefix}*,", file=f)
      if sig.has_recurrent_state():
        print(f"{prefix}{sig.get_module_call_state_param_code_str('state')},", file=f)
        print(f"{prefix}{sig.get_module_call_state_param_code_str('initial_state')},", file=f)
        args.extend(("state", "initial_state"))
      mod_args = sig.get_all_derived_args()
      if mod_args:
        for param in mod_args:
          print(f"{prefix}{param.get_module_param_code_str()},", file=f)
          args.append(param.get_module_param_name())
      print(f"{prefix}name: Optional[Union[str, NameCtx]] = None) -> Layer:", file=f)
      print('  """', file=f)
      if layer_class.__doc__:
        for i, line in enumerate(layer_class.__doc__.splitlines(keepends=True)):
          if i == 0 and not line.strip():
            continue
          print(line if line.strip() else line.strip(" "), end="", file=f)
        print("", file=f)
      if sig.docstring:
        for line in sig.docstring.splitlines():
          print(("  " + line) if line else "", file=f)
        print("", file=f)
      if sig.has_source_param():
        print(f"  {sig.get_module_call_source_docstring()}", file=f)
      if sig.has_recurrent_state():
        print(f"  {sig.get_module_call_state_docstring('state')}", file=f)
        print(f"  {sig.get_module_call_state_docstring('initial_state')}", file=f)
      for param in mod_args:
        print(param.get_module_param_docstring(indent="  "), file=f)
      print("  :param str|None name:", file=f)
      print('  """', file=f)
      if any(p.is_module_init_arg() for p in mod_args):
        print(f"  mod = {module_name}(", file=f)
        for param in mod_args:
          if param.is_module_init_arg():
            print(f"    {param.get_module_param_name()}={param.get_module_param_name()},", file=f)
        print("    )", file=f)
      else:
        print(f"  mod = {module_name}()", file=f)
      module_call_args = []
      for param in mod_args:
        if not param.is_module_init_arg():
          module_call_args.append(param)
      if sig.has_source_param() and not module_call_args:
        if sig.has_recurrent_state():
          print(f"  return mod(source, state=state, initial_state=initial_state, name=name)", file=f)
        else:
          print(f"  return mod(source, name=name)", file=f)
      else:
        print(f"  return mod(", file=f)
        if sig.has_source_param():
          print("    source,", file=f)
        if sig.has_recurrent_state():
          print("    state=state,", file=f)
          print("    initial_state=initial_state,", file=f)
        for param in module_call_args:
          print(f"    {param.get_module_param_name()}={param.get_module_param_name()},", file=f)
        print("    name=name)", file=f)

    print(name, sig)


class LayerSignature:
  """
  Like inspect.Signature but extended and specific for RETURNN layers.

  We try to handle it in a generic way, although some layers need special handling.
  """
  def __init__(self, layer_class: Type[LayerBase], others: Dict[Type[LayerBase], LayerSignature]):
    self.layer_class = layer_class
    self.others = others
    self.inspect_init_sig = inspect.signature(layer_class.__init__)
    self.params = {}  # type: Dict[str, LayerSignature.Param]
    self.docstring = None  # type: Optional[str]
    self._defined_base_params = []  # type: List[str]
    self._init_args()
    self._parse_init_docstring()
    self._find_super_call_assignments()

  def has_source_param(self) -> bool:
    """
    Whether this layer has a "from" arg.
    """
    if issubclass(
          self.layer_class,
          (SourceLayer, ConstantLayer, VariableLayer, CondLayer, SwitchLayer, GenericAttentionLayer)):
      return False
    return True

  def support_multiple_sources(self) -> bool:
    """
    Whether "from" supports multiple sources (list of layers).
    When :func:`need_multiple_sources` returns true, this ofc also implies that it supports it,
    and we do not necessarily list all those cases here.
    """
    if issubclass(self.layer_class, (CombineLayer, CompareLayer, StackLayer)):
      return True
    return False

  def need_multiple_sources(self) -> bool:
    """
    Whether "from" needs multiple sources (list of layers).
    """
    if self.layer_class.layer_class == "eval":
      return False
    if issubclass(self.layer_class, (CombineLayer, CompareLayer, StackLayer)):
      return True
    if self.layer_class.layer_class in {"dot"}:
      return True
    return False

  # noinspection PyMethodMayBeStatic
  def default_source(self) -> Optional[str]:
    """
    If there is a reasonable default "from", return repr.
    """
    if issubclass(self.layer_class, (RecLayer, SubnetworkLayer)):
      return "()"
    return None

  def get_module_call_source_param_code_str(self):
    """
    Code for `source` param
    """
    assert self.has_source_param()
    s = "source: "
    if self.need_multiple_sources():
      s += "Union[List[LayerRef], Tuple[LayerRef]]"
    elif self.support_multiple_sources():
      s += "Union[LayerRef, List[LayerRef], Tuple[LayerRef]]"
    else:
      s += "LayerRef"
    default = self.default_source()
    if default:
      s += " = " + default
    return s

  def get_module_call_source_docstring(self):
    """
    Code for docstring of `source` param
    """
    s = ":param "
    if self.need_multiple_sources():
      s += "list[LayerRef]|tuple[LayerRef]"
    elif self.support_multiple_sources():
      s += "LayerRef|list[LayerRef]|tuple[LayerRef]"
    else:
      s += "LayerRef"
    s += " source:"
    return s

  def get_module_call_state_param_code_str(self, param_name: str):
    """
    Code for `state` param
    """
    assert self.has_recurrent_state()
    return f"{param_name}: Optional[Union[LayerRef, Dict[str, LayerRef], NotSpecified]] = NotSpecified"

  def get_module_call_state_docstring(self, param_name: str):
    """
    Code for docstring of `source` param
    """
    assert self.has_recurrent_state()
    return f":param LayerRef|list[LayerRef]|tuple[LayerRef]|NotSpecified|None {param_name}:"

  def has_module_init_args(self) -> bool:
    """
    Whether there are other call args (despite source)
    """
    for _, param in self.params.items():
      if param.is_module_init_arg():
        return True
    return False

  def has_module_call_args(self) -> bool:
    """
    Whether there are other call args (despite source)
    """
    return bool(self.get_module_call_args())

  def get_all_derived_args(self, stop_bases=None) -> List[Param]:
    """
    Get all module args, including bases.
    """
    if stop_bases is None:
      # Derive some reasonable default.
      if self.is_functional():
        if self.layer_class is DropoutLayer:
          stop_bases = (LayerBase,)  # special case
        else:
          stop_bases = (LayerBase, _ConcatInputLayer)
      else:  # not functional
        stop_bases = ()  # just all
    blacklist = set()
    ls = []
    sig = self
    while sig:
      for _, param in sig.params.items():
        if param.returnn_name in blacklist:
          continue
        ls.append(param)
        blacklist.add(param.returnn_name)
      if not issubclass(sig.layer_class.__base__, LayerBase):
        break
      if sig.layer_class.__base__ in stop_bases:
        break
      blacklist.update(sig._defined_base_params)
      sig = sig.derived_layer()
    return ls

  def get_module_call_args(self) -> List[Param]:
    """
    Get all module call args, including bases.
    """
    ls = []
    for param in self.get_all_derived_args():
      if param.is_module_call_arg():
        ls.append(param)
    return ls

  def is_functional(self) -> bool:
    """
    :return: Whether this is purely functional, i.e. it has no params/variables.
      Also see: https://github.com/rwth-i6/returnn_common/issues/30
    """
    if self.layer_class is VariableLayer:
      # Even though this obviously has a variable, I think the functional API is nicer for this.
      return True
    return not self.has_variables()

  def has_recurrent_state(self) -> bool:
    """
    :return: whether the layer has recurrent state. that implies an extended API like:

      Inside a loop::

        mod = Module(...)
        out, state = mod(in, state=prev_state)

      Outside a loop::

        mod = Module(...)
        out, last_state = mod(in, [initial_state=initial_state])
    """
    if (
          getattr(self.layer_class.get_rec_initial_extra_outputs, "__func__")
          is getattr(LayerBase.get_rec_initial_extra_outputs, "__func__")):
      # Not derived, so no rec state.
      return False
    # Some special cases where the rec state is just an internal implementation detail
    # and not supposed to be set by the user.
    if issubclass(self.layer_class, BaseChoiceLayer):
      return False
    return True

  _IgnoreParamNames = {
    "self", "name", "network", "output",
    "n_out", "out_type", "sources", "target", "loss", "loss_", "size_target",
    "name_scope", "reuse_params",
    "rec_previous_layer", "control_dependencies_on_output",
    "state", "initial_state", "initial_output",
    "extra_deps", "collocate_with",
    "batch_norm",
    "is_output_layer", "register_as_extern_data",
    "copy_output_loss_from_source_idx",
  }

  _LayerClassesWithExplicitDim = {
    LinearLayer, ConvLayer, TransposedConvLayer, RecLayer, RnnCellLayer,
    PositionalEncodingLayer, RelativePositionalEncodingLayer,
    "get_last_hidden_state"}

  _LayerClassesWithExplicitTarget = {
    ChoiceLayer}

  def _init_args(self):
    # n_out is handled specially
    if self._LayerClassesWithExplicitDim.intersection((self.layer_class, self.layer_class.layer_class)):
      self.params["n_out"] = LayerSignature.Param(
        self,
        inspect.Parameter(
          name="n_out",
          kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
        param_type_s="int",
        docstring="output dimension")
    if self.layer_class in self._LayerClassesWithExplicitTarget:
      self.params["target"] = LayerSignature.Param(
        self,
        inspect.Parameter(
          name="target",
          kind=inspect.Parameter.KEYWORD_ONLY),
        param_type_s="LayerBase",
        docstring="target")
    if self.layer_class.layer_class in BlacklistLayerArgs:
      blacklist = BlacklistLayerArgs[self.layer_class.layer_class]
    else:
      blacklist = set()
    for name, param in self.inspect_init_sig.parameters.items():
      # Ignore a number of params which are handled explicitly.
      if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
        continue
      if name.startswith("_"):
        continue
      if name in self._IgnoreParamNames:
        continue
      if name in blacklist:
        continue
      param = inspect.Parameter(name=param.name, kind=param.KEYWORD_ONLY, default=param.default)
      self.params[name] = LayerSignature.Param(self, param)

  def _parse_init_docstring(self):
    if not self.layer_class.__init__.__doc__:
      return
    base_prefix = "    "
    lines = []
    last_was_param = False
    param = None  # type: Optional[LayerSignature.Param]
    for line in self.layer_class.__init__.__doc__.splitlines():
      if not line or not line.strip():
        if lines and lines[-1]:
          lines.append("")
        last_was_param = False
        continue
      assert line.startswith(base_prefix)
      line = line[len(base_prefix):]
      if last_was_param and line.startswith("  "):
        if param:
          param.docstring += "\n" + line[2:]
        continue
      param = None
      last_was_param = False
      if line.strip().startswith(":param "):
        last_was_param = True
        assert line.startswith(":")
        _, param_s, doc_s = line.split(":", 2)
        assert isinstance(param_s, str) and isinstance(doc_s, str)
        assert param_s.startswith("param ")
        param_s = param_s[len("param "):]
        if " " not in param_s:
          param_name = param_s
          param_type_s = None
        else:
          param_type_s, param_name = param_s.rsplit(" ", 1)
          assert isinstance(param_type_s, str) and isinstance(param_name, str)
        if param_name.startswith("_"):
          continue
        if param_name in self._IgnoreParamNames and param_name not in self.params:
          continue
        if param_name not in self.params:  # some typo or bugs we might have in some RETURNN version
          continue
        assert param_name in self.params, f"{self!r}, line: {line!r}"
        param = self.params[param_name]
        if doc_s:
          assert doc_s.startswith(" ")
          doc_s = doc_s[1:]
        param.docstring = doc_s
        param.param_type_s = param_type_s
      else:
        lines.append(line)
    if lines and lines[-1]:
      lines.append("")
    self.docstring = "\n".join(lines)

  def __repr__(self):
    args = []
    if self.has_source_param():
      args.append("source")
    if self.has_recurrent_state():
      args.append("state")
    args += [arg.get_module_param_name() for arg in self.get_all_derived_args()]
    return f"<{self.__class__.__name__} {self.layer_class.__name__} ({', '.join(args)})>"

  def _find_super_call_assignments(self):
    """
    Inspects the super call of the layer class in RETURNN, extracts the super call parameters from there.
    """
    # get code as string list
    code = inspect.getsource(self.layer_class.__init__).splitlines()

    # get list of lines starting with super and take the first one
    lines = [line.strip() for line in code if line.strip().startswith("super")]
    if not lines:
      self._super_call_assignments = None
      return

    base_sig = self.derived_layer()
    assert base_sig

    # reformat the super call to extract what we need
    # get the first line which contains super to get the super call
    super_call = lines[0]  # super_call=super(ChoiceLayer, self).__init__(beam_size=beam_size, search=search, **kwargs)
    # only keep the part after init of super_call
    call = super_call.split(".")[1]  # call = "__init__(beam_size=beam_size, search=search, **kwargs)"
    assert call.startswith("__init__")
    assert call.endswith(")")
    # remove "__init__(" and ") from the call string
    call_pruned = call[len("__init__("):-len(")")]  # call pruned = "beam_size=beam_size, search=search, **kwargs"

    # get list of tuples for parameter with (param_name, value)
    self._super_call_assignments = []
    for arg in call_pruned.split(","):
      arg = arg.strip()
      if arg.strip() == "**kwargs":
        continue
      key, value = arg.split("=")
      key, value = key.strip(), value.strip()
      if key in self._IgnoreParamNames:
        continue
      assert key in base_sig.params
      param = base_sig.params[key]
      self._defined_base_params.append(key)
      if key == value and value not in self.params:
        if param.param_type_s and "NotSpecified" not in param.param_type_s:
          assert "Optional" in param.param_type_s or "None" in param.param_type_s
          value = "None"
        else:
          value = "NotSpecified"
      self._super_call_assignments.append((key, value))

  def has_init_super_call_assignments(self) -> bool:
    """
    :return: whether we have a specific super call for __init__
    """
    return bool(self._super_call_assignments)

  def has_defined_base_params(self) -> bool:
    """
    :return: whether it defines base params through super init
    """
    return bool(self.get_defined_base_params())

  def get_defined_base_params(self) -> List[str]:
    """
    :return: base params which are explicitly defined by this layer
    """
    return [p for p in self._defined_base_params if p not in self.params]

  def need_module_init(self) -> bool:
    """
    :return: whether we need to implement __init__ for this module
    """
    if self.has_module_init_args():
      return True
    if self.derived_layer() and self.has_init_super_call_assignments():
      return True
    return False

  def get_init_super_call_code_str(self) -> str:
    """
    Inspects the super call of the layer class in RETURNN, extracts the super call parameters from there,
    removes unwanted parameters and builds a super call which can be written as super call into _generated_layers.py
    for that class.

    :return: Code string for the super call e.g `"super().__init__(...)"`
    """
    tup_ls = self._super_call_assignments
    if tup_ls is None:
      return 'super().__init__()'
    if not tup_ls:
      return 'super().__init__(**kwargs)'
    tup_ls = [
      (key, value) for (key, value) in tup_ls if value not in self.params or self.params[value].is_module_init_arg()]
    tup_ls = [f"{key}={value}" for (key, value) in tup_ls]
    tup_ls += ["**kwargs"]
    return "super().__init__(%s)" % ", ".join(tup_ls)

  def derived_layer(self) -> Optional[LayerSignature]:
    """
    :return: the layer signature of the base layer class if this is derived from another layer
    """
    cls_base = self.layer_class.__base__
    if issubclass(cls_base, LayerBase):
      if cls_base.__name__ in BlacklistLayerClassNames or cls_base.layer_class in BlacklistLayerClassNames:
        return self.others[LayerBase]
      return self.others[cls_base]
    return None

  def has_variables(self):
    """
    :return: whether this layers has variables. this is somewhat heuristically
    :rtype: bool
    """
    # somewhat heuristically
    for param in self.params.values():
      if "weights_init" in param.returnn_name:
        return True
    if self.layer_class.layer_class in {"variable", "batch_norm"}:
      return True
    derived = self.derived_layer()
    if derived:
      return derived.has_variables()
    return False

  class Param:
    """
    One param
    """
    def __init__(self, parent: LayerSignature, param: inspect.Parameter,
                 *,
                 docstring: Optional[str] = None,
                 param_type_s: Optional[str] = None):
      self.parent = parent
      self.inspect_param = param
      self.returnn_name = param.name
      self.docstring = docstring
      self.param_type_s = param_type_s

    def __repr__(self):
      return f"<Param {self.parent.layer_class.__name__} {self.returnn_name}>"

    def is_module_init_arg(self):
      """
      Whether this param should become a Module __init__ arg.
      """
      return not self.is_module_call_arg()

    def is_module_call_arg(self):
      """
      Whether this param should become a Module call arg.
      """
      if self.returnn_name in {"reuse_params", "chunking_layer", "axes", "axis"}:
        return True
      if not self.param_type_s:
        return False
      return "LayerBase" in self.param_type_s

    def get_module_param_name(self):
      """
      Param name used for module.
      """
      return self.returnn_name.lower()

    def get_module_param_code_str(self):
      """
      Get code for param
      """
      s = self.get_module_param_name()
      s += f": {self.get_module_param_type_code_str()}"
      if self.inspect_param.default is not self.inspect_param.empty:
        s += " = NotSpecified"
      return s

    def get_module_param_type_code_str(self):
      """
      Param type
      """
      if self.returnn_name == "reuse_params":
        return "Optional[Union[LayerRef, Dict[str, Any]]]"
      if self.returnn_name == "chunking_layer":
        return "LayerRef"
      if self.returnn_name == "unit":
        return "str"
      if self.returnn_name == "max_seq_len":
        return "Optional[Union[str, int]]"
      if not self.param_type_s:
        return "Any"
      t = self.param_type_s
      optional = False
      if t.endswith("|None"):
        t = t[:-len("|None")]
        optional = True
      if t.startswith("None|"):
        t = t[len("None|"):]
        optional = True
      if "-" in t:
        return "Any"
      t = re.sub(r"\blist\b", "List", t)
      t = re.sub(r"\btuple\b", "Tuple", t)
      t = re.sub(r"\bdict\b", "Dict", t)
      t = re.sub(r"\bLayerBase\b", "LayerRef", t)
      t = re.sub(r",(?=\S)", ", ", t)
      if "|" in t:
        if "[" in t or "(" in t:
          return "Any"
        t = f'Union[{", ".join(t.split("|"))}]'
      if optional:
        return f"Optional[{t}]"
      return t

    def get_module_param_docstring(self, indent="  "):
      """
      Docstring
      """
      s = indent + ":param "
      if self.param_type_s:
        s += self.param_type_s + " "
      s += self.get_module_param_name()
      s += ":"
      if self.docstring:
        lines = self.docstring.splitlines()
        if lines[0]:
          s += " " + lines[0]
        for line in lines[1:]:
          s += "\n" + indent + "  " + line
      return s


def format_multi_line_str(s: str, *, indent: str = "") -> str:
  """
  E.g. docstring etc
  """
  from io import StringIO
  ss = StringIO()
  print(indent + '"""', file=ss)
  for line in s.splitlines():
    print((indent + line) if line.strip() else "", file=ss)
  print(indent + '"""', file=ss, end="")
  return ss.getvalue()


def get_module_class_name_for_layer_class(sig: LayerSignature) -> str:
  """
  For some layer class, return the Module class name
  """
  layer_class = sig.layer_class
  if layer_class is LayerBase:
    return "_Base"
  name = layer_class.__name__
  assert name.endswith("Layer")
  name = name[:-len("Layer")]
  if name.startswith("_"):
    return name
  if layer_class.layer_class in LayersHidden or sig.is_functional() or sig.has_recurrent_state():
    return "_" + name  # we make a public function for it, but the module is hidden
  return name


def collect_layers():
  """
  Collect list of layers.
  """
  from returnn.tf.layers import base, basic, rec
  ls = []
  added = set()
  for mod in [base, basic, rec]:
    for key, value in vars(mod).items():
      if isinstance(value, type) and issubclass(value, LayerBase):
        if value in added:
          continue
        if value.layer_class and value.layer_class in BlacklistLayerClassNames:
          continue
        if value.__name__ in BlacklistLayerClassNames:
          continue
        if issubclass(value, InternalLayer):
          continue
        ls.append(value)
        added.add(value)
  return ls


if __name__ == "__main__":
  better_exchook.install()
  setup()
