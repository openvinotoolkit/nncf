from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import HWFusedPatterns

# from nncf.experimental.onnx.graph.pattern_operations import CONV_OPERATIONS
# from nncf.experimental.onnx.graph.pattern_operations import RELU_OPERATIONS


def _get_onnx_hw_fused_patterns() -> HWFusedPatterns:
    CONV_OPERATIONS = {'type': ['Conv'],
                       'label': 'CONV'}

    RELU_OPERATIONS = {'type': ['Relu'],
                       'label': 'RELU'}

    hw_fused_patterns = HWFusedPatterns()

    conv_ops = GraphPattern()
    conv_ops.add_node(**CONV_OPERATIONS)
    hw_fused_patterns.register(conv_ops, CONV_OPERATIONS['label'], match=False)

    relu_ops = GraphPattern()
    relu_ops.add_node(**RELU_OPERATIONS)
    hw_fused_patterns.register(relu_ops, RELU_OPERATIONS['label'], match=False)

    hw_fused_patterns.register(conv_ops + relu_ops, 'CONV + RELU', match=True)
    return hw_fused_patterns


ONNX_HW_FUSED_PATTERNS = _get_onnx_hw_fused_patterns()
