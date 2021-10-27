"""
container functions
"""
from returnn_common.nn import *


class ModuleList(Module):
  """
  Module list, getting passed an Interable of Modules and creates a list of Modules in that order
  """
  def __init__(self, modules: Optional[Iterable[Module]] = None):
    super().__init__()
    self._modules = []
    if modules is not None:
      for idx, module in enumerate(modules):
        setattr(self, str(idx), module)

  def _get_makers(self):
    return {key: value for (key, value) in vars(self).items() if isinstance(value, ILayerMaker)}

  def append(self, module: Module) -> "ModuleList":
    """
    appends one module to the list
    """
    setattr(self, str(len(self)), module)
    return self

  def extend(self, modules: Iterable[Module]) -> "ModuleList":
    """
    appends multiple modules to the list
    """
    for module in modules:
      self.append(module)
    return self

  def __len__(self) -> int:
    return len(self._get_makers())

  def __iter__(self) -> Iterator[Module]:
    return iter(self._get_makers().values())

  forward = ILayerMaker.forward  # stays abstract


class Sequential(ModuleList):
  """
  Sequential Module, takes callable of Modules which are then executed in sequence
  """

  def __init__(self, *modules):
    super().__init__()
    if len(modules) == 1 and isinstance(modules[0], Dict):
      for key, module in modules[0].items():
        setattr(self, key, module)
    else:
      for idx, module in enumerate(modules):
        setattr(self, str(idx), module)

  def forward(self, inp) -> LayerRef:
    """
    Forward
    """
    for module in self:
      inp = module(inp)
    return inp