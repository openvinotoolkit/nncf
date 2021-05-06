from torch import nn

from nncf.common.graph.graph import NNCFNodeName


class PrunedModuleInfo:
    def __init__(self, module_node_name: NNCFNodeName, module: nn.Module, operand, node_id: int):
        self.module_node_name = module_node_name
        self.module = module
        self.operand = operand
        self.nncf_node_id = node_id
        self.key = self.module_node_name
