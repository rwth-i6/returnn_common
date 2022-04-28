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
import typing
import collections.abc
from typing import Type, Optional, Union, Dict, List, Tuple, Sequence, Set
import returnn
from returnn.util import better_exchook
from returnn.util.basic import camel_case_to_snake_case, NotSpecified
from returnn.tf.layers.base import LayerBase, InternalLayer
# noinspection PyProtectedMember
from returnn.tf.layers.basic import _ConcatInputLayer, CopyLayer
from returnn.tf.layers.basic import DropoutLayer
from returnn.tf.layers.basic import VariableLayer, SubnetworkLayer
from returnn.tf.layers.rec import RecLayer
from returnn.tf.layers.rec import BaseChoiceLayer

_my_dir = os.path.dirname(os.path.abspath(__file__))
_out_filename = f"{_my_dir}/_generated_layers.py"


# We use blacklists instead of whitelists such that we can more easily run this script in the future.

# These layers are deprecated or not needed for various reasons, and thus exclude them.
# Some of them are also very easily reproduced by other layers and thus not needed.
# If you think some of these are needed, or you are unsure how to get the corresponding functionality,
# please open an issue.
BlacklistLayerClassNames = {
  "_ConcatInputLayer",  # we don't do automatic concat, https://github.com/rwth-i6/returnn_common/issues/41

  # all these manual
  "variable",
  "concat",
  "activation",
  "dropout",
  "batch_norm",
  "linear",
  "conv",
  "transposed_conv",
  "pool",
  "rec",
  "self_attention",
  "concat_attention",
  "gauss_window_attention",
  "relative_positional_encoding",
  "gating",
  "layer_norm",
  "norm",
  "switch",
  "rand_int",  # random instead

  "RecStepInfoLayer",
  "_TemplateLayer",
  "cond",  # explicitly
  "masked_computation", "unmask",  # explicitly
  "subnetwork",  # explicitly

  "source",  # we have get_extern_data instead
  "reinterpret_data",
  "rec_last_output",  # explicitly
  "swap_axes",
  "gather_nd",  # -> gather
  "softmax",  # misleading (because not just activation), also we will have a separate softmax activation
  "expand_dims",  # not sure if this is ever needed
  "weighted_sum",
  "elemwise_prod",
  "combine_dims",  # -> merge_dims
  "split_batch_time",  # should have different API. or just unflatten_batch?
  "loss",
  "transpose",
  "accumulate_mean",
  "framewise_statistics",
  "image_summary",
  "get_rec_accumulated",  # covered by our Loop logic
  "decide_keep_beam",  # internal
  "rnn_cell",  # -> rec
  "generic_attention",  # -> dot
  "dot_attention",  # -> dot
  "AttentionBaseLayer",
  "GlobalAttentionContextBaseLayer",
  "twod_lstm",
}

LayersHidden = {
  "combine",  # only needed as base; we should have all common combining functions wrapped here
  "eval",  # we should have all common functions wrapped here
  "compare",  # we should have all common comparison functions wrapped here
  "split",
  "get_last_hidden_state",  # we handle all state explicitly, there is no hidden state. this is only needed internally
  "pool",  # https://github.com/rwth-i6/returnn_common/issues/61
}

LayersWithoutSourceArg = {
  "source",
  "constant", "variable",
  "train_flag", "global_train_step",
  "cond", "switch",
  "generic_attention",
  "range", "rand_int", "random", "random_state_init",
  "ctc_loss",
  "sparse_softmax_cross_entropy_with_logits",
  "edit_distance",
}

LayersSupportingMultipleSources = {
  "eval", "combine", "compare", "stack"
}

LayersNeedingMultipleSources = LayersSupportingMultipleSources.copy() - {"eval", "compare"}

LayersExplicitFixedMultipleSources = {
  "dot": 2,
}

LayerClassesWithExplicitDim = {
  "linear", "conv", "transposed_conv", "rec", "rnn_cell",
  "positional_encoding", "relative_positional_encoding",
  "get_last_hidden_state",
}

LayerClassesWithExplicitTarget = {
  "choice",
}

IgnoreLayerArgs = {
  "self", "name", "network", "output",
  "n_out", "out_type", "sources", "target", "loss", "loss_", "size_target",
  "param_device", "trainable", "custom_param_importer",
  "only_on_eval", "only_on_search",
  "L2", "darc1", "spatial_smoothing", "param_variational_noise",
  "activation",  # more explicitly decoupled. https://github.com/rwth-i6/returnn_common/issues/62
  "name_scope", "reuse_params",
  "rec_previous_layer", "control_dependencies_on_output",
  "state", "initial_state", "initial_output",
  "extra_deps", "collocate_with",
  "batch_norm",
  "is_output_layer", "need_last", "register_as_extern_data",
  "copy_output_loss_from_source_idx",
  "num_splits", "size_splits",  # out_dims instead
  # keep dims should never be needed
  "keepdims", "keep_dims",
  "add_var2_if_empty", "add_time_axis", "add_batch_axis", "with_batch_dim",
  "unbroadcast",
  # keep order should be correct by default (with new behavior version) and not needed otherwise
  "keep_order",
  # order of axes should never matter
  "enforce_batch_dim_axis", "enforce_batch_major", "enforce_time_major",
  "red1", "red2", "var1", "var2",  # single reduce, and also var automatically, because we always want unique dims
  "auto_use_channel_first", "use_channel_first",
  # no need because of tags
  "output_dim_via_time_from",
}

PerLayerIgnoreArgs = {
  "copy": {"in_dim", "out_dim"},
  "stack": {"axis"},
}

# Mandatory == non-optional
# We derive this already from the signature.
# However, here we add some more, just for returnn-common.
PerLayerMandatoryArgs = {
  "layer_norm": {"in_dim"},
  "slice": {"out_dim"},
  "slice_nd": {"out_spatial_dim"},
  "scatter_nd": {"out_spatial_dim"},
  "range": {"out_spatial_dim"},
  "conv": {"out_dim", "in_spatial_dims"},
  "pool": {"in_spatial_dims"},
  "transposed_conv": {"out_dim", "in_spatial_dims"},
  "stack": {"out_spatial_dim"},
  "remove": {"out_dim"},
  "split_batch_beam": {"beam_dim"},
}

PerLayerOptionalArgs = {
  "choice": {"target": "None"},
}

FunctionNameMap = {
  "source": "external_data",  # but not used actually because blacklisted
  "norm": "normalize",
  "softmax_over_spatial": "softmax",  # generic also for normal softmax on feature
  "window": "rec_window",  # for non-recurrent case, we would provide some own custom wrapper
  "cumsum": "rec_cum_sum",  # for non-recurrent case, we would provide some own custom wrapper. also consistency
  "cum_concat": "rec_cum_concat",  # consistency
}

PerLayerOutDimArgs = {
  "slice": ["out_dim"],
  "slice_nd": ["out_spatial_dim"],
  "range": ["out_spatial_dim"],
  "range_from_length": ["out_spatial_dim"],
  "window": ["window_dim", "out_spatial_dim"],
  # pad?
  # unflatten_nd?
  "repeat": ["out_dim"],
  # tile?
  "conv": ["out_spatial_dims"],
  "transposed_conv": ["out_spatial_dims"],
  "pool": ["out_spatial_dims"],
  # reduce_out?
  "merge_dims": ["out_dim"],
  "stack": ["out_spatial_dim"],
  "prefix_in_time": ["out_dim"],
  "postfix_in_time": ["out_dim"],
  "time_chunking": ["out_dim"],
  # shift_axis?
  "resize": ["out_dim"],
  "remove": ["out_dim"],
  "split_batch_beam": ["beam_dim"],
  "edit_distance_table": ["out_dim"],
  "cum_concat": ["out_spatial_dim"],
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
  print("from typing import Union, Optional, Tuple, Sequence, Dict, Any", file=f)
  print("import numpy", file=f)
  print("import tensorflow as tf", file=f)
  print("from returnn.util.basic import NotSpecified", file=f)
  print("from .. import nn", file=f)
  layer_classes = collect_layers()
  signatures = {}  # type: Dict[Type[LayerBase], LayerSignature]
  for layer_class in layer_classes:
    sig = LayerSignature(layer_class, signatures)
    signatures[layer_class] = sig
    if layer_class == LayerBase:
      # We now don't have any Modules at all anymore, all is functional, so we don't need the base class.
      # We might want to clean up this later maybe but for now I leave the remaining logic in.
      continue
    cls_str = get_module_class_name_for_layer_class(sig)
    if layer_class != LayerBase:
      cls_base_str = get_module_class_name_for_layer_class(sig.derived_layer())
    else:
      cls_base_str = "nn.ReturnnWrappedLayerBase"

    if not sig.is_functional() or layer_class == LayerBase:
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
        # Note: For the __call__, we do not need the nn.scoped decorator because it does not make sense to wrap it
        # into an own subnetwork.
        res_type_str = "Tuple[nn.Tensor, nn.LayerState]" if sig.has_recurrent_state() else "nn.Tensor"
        if any([
              sig.has_source_param(),
              sig.explicit_source_list(),
              sig.has_recurrent_state(),
              sig.has_module_call_args()]):
          print("  # noinspection PyShadowingBuiltins,PyShadowingNames", file=f)
          print("  def __call__(self,", file=f)
          if sig.has_source_param():
            print(f"               {sig.get_module_call_source_param_code_str()},", file=f)
          elif sig.explicit_source_list():
            for i in range(sig.explicit_source_list()):
              print(f"               {sig.get_module_call_source_param_code_str(explicit_idx=i)},", file=f)
          if sig.has_module_call_args() or sig.has_recurrent_state():
            print("               *,", file=f)
            if sig.has_recurrent_state():
              print(f"               {sig.get_module_call_state_param_code_str('state')},", file=f)
            for param in sig.get_module_call_args():
              print(f"               {param.get_module_param_code_str()},", file=f)
          print(f"               ) -> {res_type_str}:",  file=f)
        else:
          print(f"  def __call__(self) -> {res_type_str}:", file=f)
        print(format_multi_line_str("Make layer dict", indent="    "), file=f)
        if sig.has_source_param():
          if sig.need_multiple_sources():
            print(
              "    assert isinstance(source, (tuple, list)) and all(isinstance(s, nn.Tensor) for s in source)",
              file=f)
          elif sig.support_multiple_sources():
            print(
              "    assert (\n"
              "      isinstance(source, nn.Tensor) or\n"
              "      (isinstance(source, (tuple, list)) and all(isinstance(s, nn.Tensor) for s in source)))",
              file=f)
          else:
            print("    assert isinstance(source, nn.Tensor)", file=f)
        elif sig.explicit_source_list():
          for i in range(sig.explicit_source_list()):
            print(f"    assert isinstance(source{i + 1}, nn.Tensor)", file=f)
        if sig.has_module_call_args() or sig.has_recurrent_state():
          if sig.has_module_call_args():
            print("    args = {", file=f)
            for param in sig.get_module_call_args():
              print(f"      '{param.returnn_name}': {param.get_module_param_name()},", file=f)
            print("    }", file=f)
            print("    args = {key: value for (key, value) in args.items() if value is not NotSpecified}", file=f)
          else:
            print("    args = {}", file=f)
          if sig.has_recurrent_state():
            # There must be an axis argument.
            assert "axis" in sig.params
            print("    self.handle_recurrent_state(args, axis=axis, state=state)", file=f)
        if sig.has_recurrent_state():
          print("    layer = nn.make_layer({", file=f)
        else:
          print("    return nn.make_layer({", file=f)
        print(f"      'class': {layer_class.layer_class!r},", file=f)
        if sig.has_source_param():
          print("      'from': source,", file=f)
        elif sig.explicit_source_list():
          print(
            f"      'from': [{', '.join('source' + str(i + 1) for i in range(sig.explicit_source_list()))}],", file=f)
        if sig.has_module_call_args() or sig.has_recurrent_state():
          print("      **args,", file=f)
        print("      **self.get_opts()}, module=self)", file=f)
        if sig.has_recurrent_state():
          print("    out_state = self.returnn_layer_get_recurrent_state(layer)", file=f)
          print("    return layer, out_state", file=f)
      else:
        print("", file=f)
        print("  __call__ = nn.ReturnnWrappedLayerBase.__call__  # abstract", file=f)

    # Make function if this is functional
    name = get_module_class_name_for_layer_class(sig)
    if sig.is_functional() and not layer_class.__name__.startswith("_") and layer_class.layer_class:
      name = camel_case_to_snake_case(name.lstrip("_"))
      if name in FunctionNameMap:
        name = FunctionNameMap[name]
      # Also see get_module_class_name_for_layer_class for the hidden logic.
      if sig.layer_class.layer_class in LayersHidden:
        name = "_" + name
      print("\n", file=f)

      res_types = ["nn.Tensor"]
      res_args = ["layer"]
      if layer_class.layer_class in PerLayerOutDimArgs:
        res_types_, res_args_ = [], []
        for out_dim_arg in PerLayerOutDimArgs[layer_class.layer_class]:
          param = sig.params[out_dim_arg]
          res_types_.append("Sequence[nn.Dim]" if "Sequence" in param.param_type_s else "nn.Dim")
          res_args_.append(out_dim_arg)
          if "None" not in param.param_type_s:
            param.param_type_s += "|None"
          if param.inspect_param.default == param.inspect_param.empty:
            param.inspect_param = param.inspect_param.replace(default="None")
        if len(res_types_) > 1:
          res_types.append("Tuple[" + ", ".join(res_types_) + "]")
          res_args.append("(" + ", ".join(res_args_) + ")")
        else:
          res_types.extend(res_types_)
          res_args.extend(res_args_)
      if sig.has_recurrent_state():
        res_types.append("nn.LayerState")
        res_args.append("out_state")

      print("# noinspection PyShadowingBuiltins,PyShadowingNames", file=f)
      prefix = f"def {name}("
      print(f"{prefix}", file=f)
      prefix = " " * len(prefix)
      if sig.has_source_param():
        print(f"{prefix}{sig.get_module_call_source_param_code_str()},", file=f)
      elif sig.explicit_source_list():
        for i in range(sig.explicit_source_list()):
          print(f"{prefix}{sig.get_module_call_source_param_code_str(explicit_idx=i)},", file=f)
      print(f"{prefix}*,", file=f)
      if sig.has_recurrent_state():
        print(f"{prefix}{sig.get_module_call_state_param_code_str('state')},", file=f)
      mod_args = sig.get_all_derived_args()
      for param in mod_args:
        print(f"{prefix}{param.get_module_param_code_str()},", file=f)
      res_type = "Tuple[%s]" % ", ".join(res_types) if len(res_types) > 1 else res_types[0]
      print(
        f"{prefix}name: Optional[Union[str, nn.NameCtx]] = None)"
        f" -> {res_type}:",
        file=f)
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
      elif sig.explicit_source_list():
        for i in range(sig.explicit_source_list()):
          print(f"  {sig.get_module_call_source_docstring(explicit_idx=i)}", file=f)
      if sig.has_recurrent_state():
        print(f"  {sig.get_module_call_state_docstring('state')}", file=f)
      for param in mod_args:
        print(param.get_module_param_docstring(indent="  "), file=f)
      print("  :param str|nn.NameCtx|None name:", file=f)
      print(f"  :return: {', '.join(res_args)}", file=f)
      print('  """', file=f)

      if layer_class.layer_class in PerLayerOutDimArgs:
        for out_dim_arg in PerLayerOutDimArgs[layer_class.layer_class]:
          param = sig.params[out_dim_arg]
          print(f"  if {out_dim_arg} is None or {out_dim_arg} is NotSpecified:", file=f)
          in_dim_arg = None
          if out_dim_arg in ("out_dim", "out_spatial_dim") and "axis" in sig.params and "in_dim" not in sig.params:
            in_dim_arg = "axis"
          if out_dim_arg == "out_spatial_dim" and "in_spatial_dim" in sig.params:
            in_dim_arg = "in_spatial_dim"
          if out_dim_arg == "out_spatial_dims" and "in_spatial_dims" in sig.params:
            in_dim_arg = "in_spatial_dims"
          if not in_dim_arg and "axes" in sig.params:
            in_dim_arg = "axes"
          description = (
            f"{{_name_str(name, {name.lstrip('_')!r})}}:{out_dim_arg}")

          if "Sequence" in param.param_type_s:
            assert in_dim_arg is not None
            description += "{i}"
            print(
              f"    {out_dim_arg} = [\n"
              f"      nn.Dim(kind=d.kind,"
              f" description=f{description!r})\n"
              f"      for i, d in enumerate({in_dim_arg})]", file=f)
          elif in_dim_arg == "axes":
            print(
              f"    if any(d.is_batch_dim() for d in axes):\n"
              f"      kind = nn.Dim.Types.Batch\n"
              f"    elif any(d.is_feature_dim() for d in axes):\n"
              f"      kind = nn.Dim.Types.Feature\n"
              f"    else:\n"
              f"      kind = nn.Dim.Types.Spatial\n"
              f"    {out_dim_arg} = nn.Dim(kind=kind, description"
              f"=f{description!r})", file=f)
            pass
          elif in_dim_arg:
            print(
              f"    {out_dim_arg} = nn.Dim(\n"
              f"      kind={in_dim_arg}.kind, description=f{description!r})", file=f)
          else:
            print(
              f"    {out_dim_arg} = nn.SpatialDim("
              f"f{description!r})", file=f)

      if sig.has_recurrent_state() or mod_args:
        print("  args = {", file=f)
        for param in mod_args:
          print(f"    '{param.returnn_name}': {param.get_module_param_name()},", file=f)
        print("    }", file=f)
        print("  args = {key: value for (key, value) in args.items() if value is not NotSpecified}", file=f)
      if sig.has_recurrent_state():
        print(
          "  nn.ReturnnWrappedLayerBase.handle_recurrent_state("
          "args, axis=axis, state=state)", file=f)
      if res_args == ["layer"]:
        print("  return nn.make_layer({", file=f)
      else:
        print("  layer = nn.make_layer({", file=f)
      print(f"    'class': {layer_class.layer_class!r},", file=f)
      if sig.has_source_param():
        print("    'from': source,", file=f)
      elif sig.explicit_source_list():
        print(f"    'from': [{', '.join('source' + str(i + 1) for i in range(sig.explicit_source_list()))}],", file=f)
      if sig.has_recurrent_state() or mod_args:
        print(f"    **args}}, name=name or {name.lstrip('_')!r})", file=f)
      else:
        print(f"    }}, name=name or {name.lstrip('_')!r})", file=f)
      if sig.has_recurrent_state():
        print("  out_state = nn.ReturnnWrappedLayerBase.returnn_layer_get_recurrent_state(layer)", file=f)
      if res_args != ["layer"]:
        print(f"  return {', '.join(res_args)}", file=f)

    print(name, sig)

  print(
    "\n\n"
    "def _name_str(name: Optional[Union[str, nn.NameCtx]], default: str) -> str:\n"
    "  if name is None or isinstance(name, str):\n"
    "    return f'{nn.NameCtx.current_ctx().get_abs_name()}:{name or default}'\n"
    "  if isinstance(name, nn.NameCtx):\n"
    "    return name.get_abs_name()\n"
    "  raise TypeError(f'name type {type(name)} not supported')\n", file=f, end="")


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
    self._post_proc()
    self._find_super_call_assignments()

  def has_source_param(self) -> bool:
    """
    Whether this layer has a "from" arg.
    """
    if self.layer_class.layer_class in LayersWithoutSourceArg:
      return False
    if self.explicit_source_list():
      return False
    return True

  def support_multiple_sources(self) -> bool:
    """
    Whether "from" supports multiple sources (list of layers).
    When :func:`need_multiple_sources` returns true, this ofc also implies that it supports it,
    and we do not necessarily list all those cases here.
    """
    if self.layer_class.layer_class in LayersSupportingMultipleSources:
      return True
    return False

  def need_multiple_sources(self) -> bool:
    """
    Whether "from" needs multiple sources (list of layers).
    """
    if self.layer_class.layer_class in LayersNeedingMultipleSources:
      return True
    return False

  def explicit_source_list(self) -> Optional[int]:
    """
    If returned value is given, it means that instead of source: list[Layers],
    we have source1, source2 etc, the number returned here.
    """
    return LayersExplicitFixedMultipleSources.get(self.layer_class.layer_class, None)

  # noinspection PyMethodMayBeStatic
  def default_source(self) -> Optional[str]:
    """
    If there is a reasonable default "from", return repr.
    """
    if issubclass(self.layer_class, (RecLayer, SubnetworkLayer)):
      return "()"
    return None

  def get_module_call_source_param_code_str(self, explicit_idx: Optional[int] = None):
    """
    Code for `source` param
    """
    if explicit_idx is not None:
      return f"source{explicit_idx + 1}: nn.Tensor"
    assert self.has_source_param()
    s = "source: "
    if self.need_multiple_sources():
      s += "Sequence[nn.Tensor]"
    elif self.support_multiple_sources():
      s += "Union[nn.Tensor, Sequence[nn.Tensor]]"
    else:
      s += "nn.Tensor"
    default = self.default_source()
    if default:
      s += " = " + default
    return s

  def get_module_call_source_docstring(self, explicit_idx: Optional[int] = None):
    """
    Code for docstring of `source` param
    """
    if explicit_idx is not None:
      return f":param nn.Tensor source{explicit_idx + 1}:"
    s = ":param "
    if self.need_multiple_sources():
      s += "Sequence[nn.Tensor]"
    elif self.support_multiple_sources():
      s += "nn.Tensor|Sequence[nn.Tensor]"
    else:
      s += "nn.Tensor"
    s += " source:"
    return s

  def get_module_call_state_param_code_str(self, param_name: str):
    """
    Code for `state` param
    """
    assert self.has_recurrent_state()
    return f"{param_name}: Optional[Union[nn.Tensor, Dict[str, nn.Tensor], NotSpecified]] = NotSpecified"

  def get_module_call_state_docstring(self, param_name: str):
    """
    Code for docstring of `source` param
    """
    assert self.has_recurrent_state()
    return f":param nn.Tensor|Sequence[nn.Tensor]|NotSpecified|None {param_name}:"

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
          stop_bases = (LayerBase, _ConcatInputLayer, CopyLayer)
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
        out, last_state = mod(in, [state=initial_state])
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

  def _init_args(self):
    # n_out is handled specially
    if self.layer_class.layer_class in LayerClassesWithExplicitDim:
      self.params["out_dim"] = LayerSignature.Param(
        self,
        inspect.Parameter(
          name="out_dim",
          kind=inspect.Parameter.POSITIONAL_OR_KEYWORD),
        param_type_s="Dim",
        docstring="output feature dimension")
    if self.layer_class.layer_class in LayerClassesWithExplicitTarget:
      self.params["target"] = LayerSignature.Param(
        self,
        inspect.Parameter(
          name="target",
          kind=inspect.Parameter.KEYWORD_ONLY),
        param_type_s="nn.Tensor",
        docstring="target")
    for name, param in self.inspect_init_sig.parameters.items():
      # Ignore a number of params which are handled explicitly.
      if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
        continue
      if name.startswith("_"):
        continue
      if name in IgnoreLayerArgs:
        continue
      if name in PerLayerIgnoreArgs.get(self.layer_class.layer_class, ()):
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
        if param_name in IgnoreLayerArgs and param_name not in self.params:
          continue
        if param_name not in self.params:  # some typo or bugs we might have in some RETURNN version
          continue
        assert param_name in self.params, f"{self!r}, line: {line!r}"
        param = self.params[param_name]
        if doc_s:
          assert doc_s.startswith(" ")
          doc_s = doc_s[1:]
        param.docstring = doc_s
        if param_type_s:
          param_type_s = re.sub(r"\breturnn\.tf\.util\.data\.", "", param_type_s)
          param_type_s = re.sub(r"\btyping\.Sequence\b", "Sequence", param_type_s)
          param_type_s = re.sub(r"\bDimensionTag\b", "Dim", param_type_s)
          param_type_s = re.sub(r"\bDim\|str\b", "nn.Dim", param_type_s)
          param_type_s = re.sub(r"\bstr\|Dim\b", "nn.Dim", param_type_s)
          param_type_s = re.sub(r"\b(?<!nn\.)Dim\b", "nn.Dim", param_type_s)
          param_type_s = re.sub(r"\bTensor\b", "nn.Tensor", param_type_s)
          param_type_s = re.sub(r"\bLayerBase\b", "nn.Tensor", param_type_s)
        if param.inspect_param.default != param.inspect_param.empty and param_name in {"axis", "axes"}:
          if "None" not in param_type_s:
            param.inspect_param = param.inspect_param.replace(default=inspect.Parameter.empty)
        param.param_type_s = param_type_s
      else:
        lines.append(line)
    if lines and lines[-1]:
      lines.append("")
    self.docstring = "\n".join(lines)

  @classmethod
  def _handle_axis_like_arg(cls, param: Param):
    if param.param_type_s:
      replace_types = {"int": "Dim", "str": "Dim"}  # they get merged by typing.Union, so duplicates are no problem
      res_t_s = LayerSignature.Param.translate_param_type_code_to_typing_code(
        param.param_type_s, replace_types=replace_types,
        allow_optional=param.inspect_param.default != inspect.Parameter.empty)

      class _NnDummyScope:
        Dim = "nn.Dim"
        Tensor = "nn.Tensor"

      res_t = eval(res_t_s, {
        "typing": typing,
        "Optional": Optional, "Union": Union,
        "List": List, "Tuple": Tuple, "Sequence": Sequence,
        "Set": Set, "Dict": Dict,
        "Dim": "nn.Dim", "Tensor": "nn.Tensor", "NotSpecified": NotSpecified,
        "nn": _NnDummyScope})

      def _convert(t) -> str:
        if t is None:
          return "None"
        if isinstance(t, type):
          if issubclass(t, type(None)):
            return "None"
          return t.__name__
        if isinstance(t, str):
          return t
        if isinstance(t, typing.ForwardRef):
          if t.__forward_arg__ in {"Tensor", "nn.Tensor"}:
            return "nn.Tensor"
          return t.__forward_arg__
        if typing.get_origin(t) == Union:
          return "|".join(_convert(t_) for t_ in typing.get_args(t))
        if typing.get_origin(t) == tuple:
          if Ellipsis == typing.get_args(t)[-1]:
            return "tuple[%s]" % ", ".join(_convert(t_) for t_ in typing.get_args(t)[:-1])
          return "(%s)" % ", ".join(_convert(t_) for t_ in typing.get_args(t))
        if typing.get_origin(t) == list:
          return "list[%s]" % ", ".join(_convert(t_) for t_ in typing.get_args(t))
        if typing.get_origin(t) == collections.abc.Sequence:
          return "Sequence[%s]" % ", ".join(_convert(t_) for t_ in typing.get_args(t))
        if typing.get_origin(t) == dict:
          return "dict[%s]" % ", ".join(_convert(t_) for t_ in typing.get_args(t))
        raise TypeError(f"res {res_t}, t {t}, origin {typing.get_origin(t)} for param {param}")
      param.param_type_s = _convert(res_t)
      return

    # Simpler fallback
    types = []
    if param.returnn_name not in {"shape", "dims", "axes", "out_dims"}:
      types.append("Dim")
    if param.param_type_s:
      if "list" in param.param_type_s or "tuple" in param.param_type_s:
        types.append("list[Dim]")
    if param.inspect_param.default != inspect.Parameter.empty:
      types.append("None")
    param.param_type_s = "|".join(types)

  def _post_proc(self):
    if self.has_recurrent_state() and "axis" not in self.params:
      # Might need to fix the layer. But anyway just add this now.
      self.params["axis"] = LayerSignature.Param(
        parent=self,
        param=inspect.Parameter(
          name="axis", kind=inspect.Parameter.KEYWORD_ONLY,
          default=None, annotation=None),
        param_type_s="Dim",
        docstring="axis to operate over, or nn.single_step_dim")
    if "axes" in self.params and "axis" in self.params:
      param_ = self.params.pop("axes")
      param = self.params["axis"]
      param.param_type_s = "nn.Dim|Sequence[nn.Dim]"
      if param_.docstring:
        param.docstring += "\n"
        param.docstring += param_.docstring
      param.inspect_param = param.inspect_param.replace(default=inspect.Parameter.empty)  # make sure not optional
    if "axis" in self.params:
      param = self.params["axis"]
      param.inspect_param = param.inspect_param.replace(default=inspect.Parameter.empty)  # make sure not optional
    for name, param in self.params.items():
      if name == "out_shape":
        if "_MarkedDim" in param.param_type_s:
          param.param_type_s = "nn.OutShapeType"
        continue
      if name in PerLayerMandatoryArgs.get(self.layer_class.layer_class, ()):
        param.inspect_param = param.inspect_param.replace(default=inspect.Parameter.empty)  # make not optional
      if name in PerLayerOptionalArgs.get(self.layer_class.layer_class, ()):
        opt_value_s = PerLayerOptionalArgs[self.layer_class.layer_class][name]
        param.inspect_param = param.inspect_param.replace(default=opt_value_s)  # make optional
        if opt_value_s == "None":
          param.param_type_s += "|None"
      if name in {"axis", "axes"} or (param.param_type_s and "Dim" in param.param_type_s):
        self._handle_axis_like_arg(param)
    if "window_size" in self.params and "window_dim" in self.params:
      param_ = self.params.pop("window_size")
      param = self.params["window_dim"]
      if param_.docstring:
        param.docstring += "\n"
        param.docstring += param_.docstring

  def __repr__(self):
    args = []
    if self.has_source_param():
      args.append("source")
    elif self.explicit_source_list():
      for i in range(self.explicit_source_list()):
        args.append(f"source{i + 1}")
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
      if key in IgnoreLayerArgs:
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
      if cls_base == CopyLayer:
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
      if self.returnn_name in {"reuse_params", "chunking_layer", "axes", "axis", "in_spatial_dim", "in_spatial_dims"}:
        return True
      if not self.param_type_s:
        return False
      return "LayerBase" in self.param_type_s or "Tensor" in self.param_type_s

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

    @classmethod
    def translate_param_type_code_to_typing_code(cls, t: str, *,
                                                 replace_types: Optional[Dict[str, Union[str, None]]] = None,
                                                 allow_optional: bool = True) -> str:
      """
      Convert old-style param type code to new-style typing code.
      """
      def _translate(s: str) -> (str, int):
        end = len(s)
        post_replacements = []
        start_ = 0
        m_ = None
        while True:
          m = re.search(r"[()\[\]]", s[start_:])
          if not m:
            break
          m_ = m.group()
          p = m.start() + start_
          if m_ in ")]":
            end = p
            s = s[:end]
            break
          else:
            res, end_ = _translate(s[p + 1:])
            end_ += p + 1
            post_replacements.append(s[p] + res + s[end_])
            end_ += 1
            while end_ < end and not s[end_].strip():
              end_ += 1
            if end_ < end and s[p] == "(" and s[end_:end_ + 2] == "->":
              pass
            s = s[:p] + "{" + "_" * (end_ - p - 2) + "}" + s[end_:]
            start_ = end_

        if "," in s:
          parts = s.split(",")
          parts = [_translate(p)[0] for p in parts]
          s = ", ".join(parts)
          if m_ == ")":
            s = f'Tuple[{s}]'

        else:
          if "->" in s:
            return "callable", end  # cannot really handle yet

          # Sequence instead of list/tuple.
          s = re.sub(r"\blist\b", "Sequence", s)
          s = re.sub(r"\btuple\b", "Sequence", s)
          s = re.sub(r"\bset\b", "Set", s)
          s = re.sub(r"\bdict\b", "Dict", s)
          s = re.sub(r"\bLayerBase\b", "nn.Tensor", s)
          s = re.sub(r",(?=\S)", ", ", s)
          if "|" in s:
            assert "," not in s  # should be inside brackets and thus replaced above
            parts = s.split("|")
            parts = [p.strip() for p in parts]
            optional = False
            if replace_types:
              parts = [replace_types.get(p, p) for p in parts]
              parts = [p for p in parts if p]
            if "None" in parts:
              parts.remove("None")
              if allow_optional:
                optional = True
            if len(parts) >= 2:
              s = f'Union[{", ".join(parts)}]'
            elif len(parts) == 1:
              s = parts[0]
            else:
              optional = False
              s = "None" if allow_optional else "Any"
            if optional:
              s = f"Optional[{s}]"
          elif replace_types:
            s = replace_types.get(s, s)
            if not s:
              s = "None" if allow_optional else "Any"

        for rep in post_replacements:
          if rep[0] == "(" and rep[-1] == ")":
            rep = rep[1:-1]  # should never be needed
          m = re.search(r"{_*}", s)
          assert m
          p = m.start()
          if p >= len("Tuple") and s[p - len("Tuple"):p] == "Tuple" and rep[-1] == "]":
            s = s[:p] + rep[:-1] + ", ...]" + s[m.end():]
          else:
            s = s[:p] + rep + s[m.end():]
        return s, end

      t, _ = _translate(t)
      return t

    def get_module_param_type_code_str(self):
      """
      Param type
      """
      if self.returnn_name == "reuse_params":
        return "Optional[Union[nn.Tensor, Dict[str, Any]]]"
      if self.returnn_name == "chunking_layer":
        return "nn.Tensor"
      if self.returnn_name == "unit":
        return "str"
      if self.returnn_name == "max_seq_len":
        return "Optional[Union[str, int]]"
      if self.returnn_name == "axis" and not self.param_type_s:
        return "nn.Dim"
      if not self.param_type_s:
        return "Any"
      return self.translate_param_type_code_to_typing_code(self.param_type_s)

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
  # LayersHidden is our explicit list.
  # When some layer is purely functional (is_functional), then we just make the function public
  # but keep the wrapped module hidden.
  # When it has recurrent state, we anyway better use explicit public wrappers.
  # https://github.com/rwth-i6/returnn_common/issues/31
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
