from nncf.common.graph.operator_metatypes import OperatorMetatypeRegistry
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.torch.dynamic_graph.structs import NamespaceTarget
from nncf.common.hardware.opset import HWConfigOpName

FX_OPERATOR_METATYPES = OperatorMetatypeRegistry("operator_metatypes")

@FX_OPERATOR_METATYPES.register()
class FXEmbeddingMetatype(OperatorMetatype):
    name = "EmbeddingOp"
    module_to_function_names = {NamespaceTarget.TORCH_NN_FUNCTIONAL: ["embedding"]}
    hw_config_names = [HWConfigOpName.EMBEDDING]
    weight_port_ids = [0]