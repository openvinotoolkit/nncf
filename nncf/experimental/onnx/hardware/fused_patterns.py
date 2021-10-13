from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import HWFusedPatterns


def _get_onnx_hw_fused_patterns() -> HWFusedPatterns:
    CONV_OPERATIONS = {'type': ['Conv'],
                       'label': 'CONV'}

    RELU_OPERATIONS = {'type': ['Relu', 'Clip'],
                       'label': 'RELU'}

    BN_OPERATIONS = {'type': ['BatchNormalization'],
                     'label': 'BN'}

    hw_fused_patterns = HWFusedPatterns()

    conv_ops = GraphPattern()
    conv_ops.add_node(**CONV_OPERATIONS)
    hw_fused_patterns.register(conv_ops, CONV_OPERATIONS['label'], match=False)

    relu_ops = GraphPattern()
    relu_ops.add_node(**RELU_OPERATIONS)
    hw_fused_patterns.register(relu_ops, RELU_OPERATIONS['label'], match=False)

    bn_ops = GraphPattern()
    bn_ops.add_node(**BN_OPERATIONS)
    hw_fused_patterns.register(bn_ops, BN_OPERATIONS['label'], match=False)

    hw_fused_patterns.register(conv_ops + relu_ops, 'CONV + RELU', match=True)
    hw_fused_patterns.register(conv_ops + bn_ops + relu_ops, 'CONV + BN + RELU', match=True)
    hw_fused_patterns.register(conv_ops + bn_ops, 'CONV + BN', match=True)
    return hw_fused_patterns


ONNX_HW_FUSED_PATTERNS = _get_onnx_hw_fused_patterns()
