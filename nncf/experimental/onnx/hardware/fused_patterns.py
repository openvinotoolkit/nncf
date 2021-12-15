from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import HWFusedPatterns


def create_h_sigmoid_act() -> GraphPattern:
    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    sigmoid_node = pattern.add_node(label='SIGMOID', type='Sigmoid')
    mul_node = pattern.add_node(label='MUL', type='Mul')

    pattern.add_edge(input_pattern_node, sigmoid_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(sigmoid_node, mul_node)
    return pattern


def _get_onnx_hw_fused_patterns() -> HWFusedPatterns:
    CONV_OPERATIONS = {'type': ['Conv'],
                       'label': 'CONV'}

    RELU_OPERATIONS = {'type': ['Relu', 'Clip'],
                       'label': 'RELU'}

    BN_OPERATIONS = {'type': ['BatchNormalization'],
                     'label': 'BN'}

    hw_fused_patterns = HWFusedPatterns()

    relu_ops = GraphPattern()
    relu_ops.add_node(**RELU_OPERATIONS)

    h_sigmoid = create_h_sigmoid_act()

    ACTIVATIONS = relu_ops | h_sigmoid
    hw_fused_patterns.register(ACTIVATIONS, 'ACTIVATIONS', match=False)

    conv_ops = GraphPattern()
    conv_ops.add_node(**CONV_OPERATIONS)
    hw_fused_patterns.register(conv_ops, CONV_OPERATIONS['label'], match=False)

    bn_ops = GraphPattern()
    bn_ops.add_node(**BN_OPERATIONS)
    hw_fused_patterns.register(bn_ops, BN_OPERATIONS['label'], match=False)

    hw_fused_patterns.register(conv_ops + ACTIVATIONS, 'CONV + ACTIVATIONS', match=True)
    hw_fused_patterns.register(conv_ops + bn_ops + ACTIVATIONS, 'CONV + BN + ACTIVATIONS', match=True)
    hw_fused_patterns.register(conv_ops + bn_ops, 'CONV + BN', match=True)
    return hw_fused_patterns


ONNX_HW_FUSED_PATTERNS = _get_onnx_hw_fused_patterns()
