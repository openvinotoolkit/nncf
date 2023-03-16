"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import PatternNames
from nncf.common.utils.registry import Registry
from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.hardware.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.onnx.hardware.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.onnx.hardware.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.onnx.hardware.pattern_operations import LINEAR_OPERATIONS


ONNX_HW_FUSED_PATTERNS = Registry('onnx')

# BLOCK PATTERNS


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.SCALE_SHIFT)
def create_scale_shift():
    pattern = GraphPattern()
    pattern_input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                             GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                   GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD, SUBTRACT',
                                   GraphPattern.METATYPE_ATTR: [om.ONNXAddLayerMetatype, om.ONNXSubMetatype]})
    pattern.add_edge(pattern_input_node, mul_node)
    pattern.add_edge(mul_node, add_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.SWISH_WITH_SIGMOID)
def create_swish_with_sigmoid():
    pattern = GraphPattern()
    pattern_input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                             GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    sigmoid_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SIGMOID',
                                       GraphPattern.METATYPE_ATTR: om.ONNXSigmoidMetatype})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                   GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype})

    pattern.add_edge(pattern_input_node, sigmoid_node)
    pattern.add_edge(sigmoid_node, mul_node)
    pattern.add_edge(pattern_input_node, mul_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.SWISH_WITH_HARD_SIGMOID)
def create_swish_with_hard_sigmoid():
    pattern = GraphPattern()
    pattern_input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                             GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    hard_sigmoid_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'HARD_SIGMOID',
                                            GraphPattern.METATYPE_ATTR: om.ONNXHardSigmoidMetatype})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                   GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype})

    pattern.add_edge(pattern_input_node, hard_sigmoid_node)
    pattern.add_edge(hard_sigmoid_node, mul_node)
    pattern.add_edge(pattern_input_node, mul_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.MATMUL_SOFTMAX_MATMUL)
def create_matmul_softmax_matmul():
    pattern = GraphPattern()
    softmax_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SOFTMAX',
                                    GraphPattern.METATYPE_ATTR: om.ONNXSoftmaxMetatype})
    mat_mul_1_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL_1',
                                      GraphPattern.METATYPE_ATTR: om.ONNXLinearMetatype})
    mat_mul_2_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL_2',
                                      GraphPattern.METATYPE_ATTR: om.ONNXLinearMetatype})

    any_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ANY',
                                GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})

    pattern.add_edge(mat_mul_1_1, softmax_1)
    pattern.add_edge(softmax_1, mat_mul_2_1)
    pattern.add_edge(any_1, mat_mul_2_1)

    softmax_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SOFTMAX',
                                    GraphPattern.METATYPE_ATTR: om.ONNXSoftmaxMetatype})
    add_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                GraphPattern.METATYPE_ATTR: om.ONNXAddLayerMetatype})
    mat_mul_1_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL_1',
                                      GraphPattern.METATYPE_ATTR: om.ONNXLinearMetatype})
    mat_mul_2_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL_2',
                                      GraphPattern.METATYPE_ATTR: om.ONNXLinearMetatype})

    any_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ANY',
                                GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})

    pattern.add_edge(mat_mul_1_2, add_2)
    pattern.add_edge(add_2, softmax_2)
    pattern.add_edge(softmax_2, mat_mul_2_2)
    pattern.add_edge(any_2, mat_mul_2_2)

    return pattern


# INPUT PROCESSING


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.INPUT_SHIFT_SCALE)
def create_input_shift_scale():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                     GraphPattern.METATYPE_ATTR: InputNoopMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD, SUBTRACT',
                                   GraphPattern.METATYPE_ATTR: [om.ONNXAddLayerMetatype, om.ONNXSubMetatype]})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype})

    pattern.add_edge(input_node, add_node)
    pattern.add_edge(add_node, multiply_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.INPUT_PROCESSING)
def create_input_add():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                     GraphPattern.METATYPE_ATTR: InputNoopMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD, MULTIPLY',
                                   GraphPattern.METATYPE_ATTR: [om.ONNXAddLayerMetatype, om.ONNXMulLayerMetatype]})

    pattern.add_edge(input_node, add_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.INPUT_SCALE_SHIFT)
def create_input_scale_shift():
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                        GraphPattern.METATYPE_ATTR: InputNoopMetatype})
    scale_shift = create_scale_shift()

    pattern.join_patterns(scale_shift)
    return pattern


# COMBINATIONS

@ONNX_HW_FUSED_PATTERNS.register(PatternNames.ACTIVATIONS_BATCH_NORM)
def create_activations_batch_norm():
    activations = atomic_activations_operations()
    batch_norm = batch_normalization_operations()
    activations.join_patterns(batch_norm)
    return activations


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.ACTIVATIONS_SCALE_SHIFT)
def create_activations_scale_shift():
    activations = atomic_activations_operations()
    scale_shift = create_scale_shift()
    activations.join_patterns(scale_shift)
    return activations


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_BATCH_NORM)
def create_arithmetic_batch_norm():
    arithmetic = arithmetic_operations()
    batch_norm = batch_normalization_operations()
    arithmetic.join_patterns(batch_norm)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_ACTIVATIONS)
def create_arithmetic_activations():
    arithmetic = arithmetic_operations()
    activations = atomic_activations_operations()
    arithmetic.join_patterns(activations)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_SCALE_SHIFT)
def create_arithmetic_scale_shift():
    arithmetic = arithmetic_operations()
    scale_shift = create_scale_shift()
    arithmetic.join_patterns(scale_shift)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_BATCH_NORM_ACTIVATIONS)
def create_arithmetic_batch_norm_activations():
    arithmetic_batch_norm = create_arithmetic_batch_norm()
    activations = atomic_activations_operations()
    arithmetic_batch_norm.join_patterns(activations)
    return arithmetic_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_SCALE_SHIFT_ACTIVATIONS)
def create_arithmetic_scale_shift_activations():
    arithmetic_scale_shift = create_arithmetic_scale_shift()
    activations = atomic_activations_operations()
    arithmetic_scale_shift.join_patterns(activations)
    return arithmetic_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_ACTIVATIONS_BATCH_NORM)
def create_arithmetic_activations_batch_norm():
    arithmetic_activations = create_arithmetic_activations()
    batch_norm = batch_normalization_operations()
    arithmetic_activations.join_patterns(batch_norm)
    return arithmetic_activations


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_ACTIVATIONS_SCALE_SHIFT)
def create_arithmetic_activations_scale_shift():
    arithmetic_activations = create_arithmetic_activations()
    scale_shift = create_scale_shift()
    arithmetic_activations.join_patterns(scale_shift)
    return arithmetic_activations


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.BATCH_NORM_ACTIVATIONS)
def create_batch_norm_activations():
    batch_norm = batch_normalization_operations()
    activations = atomic_activations_operations()
    batch_norm.join_patterns(activations)
    return batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.SCALE_SHIFT_ACTIVATIONS)
def create_scale_shift_activations():
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()
    scale_shift.join_patterns(activations)
    return scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ARITHMETIC)
def create_linear_arithmetic():
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    linear.join_patterns(arithmetic)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_BATCH_NORM)
def create_linear_batch_norm():
    linear = linear_operations()
    batch_norm = batch_normalization_operations()
    linear.join_patterns(batch_norm)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ACTIVATIONS)
def create_linear_activations():
    linear = linear_operations()
    activations = atomic_activations_operations()
    linear.join_patterns(activations)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_BATCH_NORM_ACTIVATIONS)
def create_linear_batch_norm_activations():
    linear_batch_norm = create_linear_batch_norm()
    activations = atomic_activations_operations()
    linear_batch_norm.join_patterns(activations)
    return linear_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_SCALE_SHIFT_ACTIVATIONS)
def create_linear_scale_shift_activations():
    linear_scale_shift = create_linear_scale_shift()
    activations = atomic_activations_operations()
    linear_scale_shift.join_patterns(activations)
    return linear_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ACTIVATIONS_BATCH_NORM)
def create_linear_activations_batch_norm():
    linear_activations = create_linear_activations()
    batch_norm = batch_normalization_operations()
    linear_activations.join_patterns(batch_norm)
    return linear_activations


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ACTIVATIONS_SCALE_SHIFT)
def create_linear_activations_scale_shift():
    linear_activations = create_linear_activations()
    scale_shift = create_scale_shift()
    linear_activations.join_patterns(scale_shift)
    return linear_activations


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_BATCH_NORM_SCALE_SHIFT_ACTIVATIONS)
def create_linear_bn_scale_shift_activation() -> GraphPattern:
    linear_batch_norm = create_linear_batch_norm()
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()

    linear_batch_norm.join_patterns(scale_shift)
    linear_batch_norm.join_patterns(activations)
    return linear_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.BATCH_NORM_SCALE_SHIFT_ACTIVATIONS)
def create_bn_scale_shift_activation() -> GraphPattern:
    batch_norm = batch_normalization_operations()
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()

    batch_norm.join_patterns(scale_shift)
    batch_norm.join_patterns(activations)
    return batch_norm

# DEVICE PATTERNS


@ONNX_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_SCALE_SHIFT)
def create_linear_scale_shift():
    linear = linear_operations()
    batch_norm = create_scale_shift()
    linear.join_patterns(batch_norm)
    return linear


def linear_operations():
    pattern = GraphPattern()
    pattern.add_node(**LINEAR_OPERATIONS)
    return pattern


def batch_normalization_operations():
    pattern = GraphPattern()
    pattern.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    return pattern


def atomic_activations_operations():
    pattern = GraphPattern()
    pattern.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)

    swish_sigmoid = create_swish_with_sigmoid()
    pattern.add_pattern_alternative(swish_sigmoid)

    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    pattern.add_pattern_alternative(swish_hard_sigmoid)
    return pattern


def arithmetic_operations():
    pattern = GraphPattern()
    pattern.add_node(**ARITHMETIC_OPERATIONS)
    return pattern
