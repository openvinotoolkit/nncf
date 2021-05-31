from torch import nn

from nncf.common.graph.graph import NNCFNodeName
from nncf.common.pruning.structs import PrunedLayerInfoBase


class PrunedModuleInfo(PrunedLayerInfoBase):
    def __init__(self, node_name: NNCFNodeName, module: nn.Module, operand, node_id: int):
        super().__init__(node_name, node_id)
        self.module = module
        self.operand = operand
        self.key = self.node_name
