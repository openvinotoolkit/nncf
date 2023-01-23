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

from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import PatternsManager
from nncf.common.utils.registry import Registry
from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.hardware.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.onnx.hardware.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.onnx.hardware.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.onnx.hardware.pattern_operations import LINEAR_OPERATIONS


ONNX_HW_FUSED_PATTERNS = Registry('onnx')


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_OPERATIONS)
def linear_operations():
    pattern = GraphPattern()
    pattern.add_node(**LINEAR_OPERATIONS)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.BATCH_NORMALIZATION_OPERATIONS)
def batch_normalization_operations():
    pattern = GraphPattern()
    pattern.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ATOMIC_ACTIVATIONS_OPERATIONS)
def atomic_activations_operations():
    pattern = GraphPattern()
    pattern.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_OPERATIONS)
def arithmetic_operations():
    pattern = GraphPattern()
    pattern.add_node(**ARITHMETIC_OPERATIONS)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ACTIVATIONS_BATCH_NORM)
def create_activations_batch_norm():
    activations = atomic_activations_operations()
    batch_norm = batch_normalization_operations()
    activations.join_patterns(batch_norm)
    return activations


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ACTIVATIONS_SCALE_SHIFT)
def create_activations_scale_shift():
    activations = atomic_activations_operations()
    scale_shift = create_scale_shift_add()
    activations.join_patterns(scale_shift)
    return activations


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SWISH_SIGMOID_BATCH_NORM)
def create_swish_sigmoid_batch_norm():
    swish_sigmoid = create_swish_with_sigmoid()
    batch_norm = batch_normalization_operations()
    swish_sigmoid.join_patterns(batch_norm)
    return swish_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SWISH_SIGMOID_SCALE_SHIFT)
def create_swish_sigmoid_multiply_add():
    swish_sigmoid = create_swish_with_sigmoid()
    scale_shift = create_scale_shift_add()
    swish_sigmoid.join_patterns(scale_shift)
    return swish_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SWISH_HARD_SIGMOID_BATCH_NORM)
def create_swish_hard_sigmoid_batch_norm():
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    batch_norm = batch_normalization_operations()
    swish_hard_sigmoid.join_patterns(batch_norm)
    return swish_hard_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SWISH_HARD_SIGMOID_SCALE_SHIFT)
def create_swish_hard_sigmoid_scale_shift():
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    scale_shift = create_scale_shift_add()
    swish_hard_sigmoid.join_patterns(scale_shift)
    return swish_hard_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SWISH_WITH_SIGMOID)
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


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SWISH_WITH_HARD_SIGMOID)
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


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_BATCH_NORM)
def create_arithmetic_batch_norm():
    arithmetic = arithmetic_operations()
    batch_norm = batch_normalization_operations()
    arithmetic.join_patterns(batch_norm)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_ACTIVATIONS)
def create_arithmetic_activations():
    arithmetic = arithmetic_operations()
    activations = atomic_activations_operations()
    arithmetic.join_patterns(activations)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_SWISH_SIGMOID)
def create_arithmetic_swish_sigmoid():
    arithmetic = arithmetic_operations()
    swish_sigmoid = create_swish_with_sigmoid()
    arithmetic.join_patterns(swish_sigmoid)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_SWISH_HARD_SIGMOID)
def create_arithmetic_swish_hard_sigmoid():
    arithmetic = arithmetic_operations()
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    arithmetic.join_patterns(swish_hard_sigmoid)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_SCALE_SHIFT)
def create_arithmetic_scale_shift():
    arithmetic = arithmetic_operations()
    scale_shift = create_scale_shift_add()
    arithmetic.join_patterns(scale_shift)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_BATCH_NORM_ACTIVATIONS)
def create_arithmetic_batch_norm_activations():
    arithmetic_batch_norm = create_arithmetic_batch_norm()
    activations = atomic_activations_operations()
    arithmetic_batch_norm.join_patterns(activations)
    return arithmetic_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_BATCH_NORM_SWISH_SIGMOID)
def create_arithmetic_batch_norm_swish_sigmoid():
    arithmetic_batch_norm = create_arithmetic_batch_norm()
    swish_sigmoid = create_swish_with_sigmoid()
    arithmetic_batch_norm.join_patterns(swish_sigmoid)
    return arithmetic_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_BATCH_NORM_SWISH_HARD_SIGMOID)
def create_arithmetic_batch_norm_swish_hard_sigmoid():
    arithmetic_batch_norm = create_arithmetic_batch_norm()
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    arithmetic_batch_norm.join_patterns(swish_hard_sigmoid)
    return arithmetic_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_SCALE_SHIFT_ACTIVATIONS)
def create_arithmetic_scale_shift_activations():
    arithmetic_scale_shift = create_arithmetic_scale_shift()
    activations = atomic_activations_operations()
    arithmetic_scale_shift.join_patterns(activations)
    return arithmetic_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_SCALE_SHIFT_SWISH_SIGMOID)
def create_arithmetic_scale_shift_swish_sigmoid():
    arithmetic_scale_shift = create_arithmetic_scale_shift()
    swish_sigmoid = create_swish_with_sigmoid()
    arithmetic_scale_shift.join_patterns(swish_sigmoid)
    return arithmetic_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_SCALE_SHIFT_SWISH_HARD_SIGMOID)
def create_arithmetic_scale_shift_swish_hard_sigmoid():
    arithmetic_scale_shift = create_arithmetic_scale_shift()
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    arithmetic_scale_shift.join_patterns(swish_hard_sigmoid)
    return arithmetic_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_ACTIVATIONS_BATCH_NORM)
def create_arithmetic_activations_batch_norm():
    arithmetic_activations = create_arithmetic_activations()
    batch_norm = batch_normalization_operations()
    arithmetic_activations.join_patterns(batch_norm)
    return arithmetic_activations


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_ACTIVATIONS_SCALE_SHIFT)
def create_arithmetic_activations_scale_shift():
    arithmetic_activations = create_arithmetic_activations()
    scale_shift = create_scale_shift_add()
    arithmetic_activations.join_patterns(scale_shift)
    return arithmetic_activations


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_SWISH_SIGMOID_BATCH_NORM)
def create_arithmetic_swish_sigmoid_batch_norm():
    arithmetic_swish_sigmoid = create_arithmetic_swish_sigmoid()
    barch_norm = batch_normalization_operations()
    arithmetic_swish_sigmoid.join_patterns(barch_norm)
    return arithmetic_swish_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_HARD_SIGMOID_SCALE_SHIFT)
def create_arithmetic_hard_sigmoid_scale_shift():
    arithmetic_swish_sigmoid = create_arithmetic_swish_sigmoid()
    scale_shift = create_scale_shift_add()
    arithmetic_swish_sigmoid.join_patterns(scale_shift)
    return arithmetic_swish_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_SWISH_HARD_SIGMOID_BATCH_NORM)
def create_arithmetic_swish_hard_sigmoid_batch_norm():
    arithmetic_swish_hard_sigmoid = create_arithmetic_swish_hard_sigmoid()
    barch_norm = batch_normalization_operations()
    arithmetic_swish_hard_sigmoid.join_patterns(barch_norm)
    return arithmetic_swish_hard_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_HARD_HARD_SIGMOID_SCALE_SHIFT)
def create_arithmetic_hard_hard_sigmoid_scale_shift():
    arithmetic_swish_hard_sigmoid = create_arithmetic_swish_hard_sigmoid()
    scale_shift = create_scale_shift_add()
    arithmetic_swish_hard_sigmoid.join_patterns(scale_shift)
    return arithmetic_swish_hard_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.BATCH_NORM_ACTIVATIONS)
def create_batch_norm_activations():
    batch_norm = batch_normalization_operations()
    activations = atomic_activations_operations()
    batch_norm.join_patterns(activations)
    return batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.BATCH_NORM_SWISH_SIGMOID)
def create_batch_norm_swish_sigmoid():
    batch_norm = batch_normalization_operations()
    swish_sigmoid = create_swish_with_sigmoid()
    batch_norm.join_patterns(swish_sigmoid)
    return batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.BATCH_NORM_SWISH_HARD_SIGMOID)
def create_batch_norm_swish_hard_sigmoid():
    batch_norm = batch_normalization_operations()
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    batch_norm.join_patterns(swish_hard_sigmoid)
    return batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SCALE_SHIFT_ACTIVATIONS)
def create_scale_shift_activations():
    scale_shift = create_scale_shift_add()
    activations = atomic_activations_operations()
    scale_shift.join_patterns(activations)
    return scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SCALE_SHIFT_SWISH_SIGMOID)
def create_scale_shift_swish_sigmoid():
    scale_shift = create_scale_shift_add()
    swish_sigmoid = create_swish_with_sigmoid()
    scale_shift.join_patterns(swish_sigmoid)
    return scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SCALE_SHIFT_SWISH_HARD_SIGMOID)
def create_scale_shift_swish_hard_sigmoid():
    scale_shift = create_scale_shift_add()
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    scale_shift.join_patterns(swish_hard_sigmoid)
    return scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_ADD_MULTIPLY)
def create_input_add_multiply():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.ONNXAddLayerMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype})

    pattern.add_edge(input_node, add_node)
    pattern.add_edge(add_node, multiply_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_SCALE_SHIFT)
def create_input_scale_shift():
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                        GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    scale_shift = create_scale_shift_add()

    pattern.join_patterns(scale_shift)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_PROCESSING)
def create_input_add():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD, MULTIPLY',
                                   GraphPattern.METATYPE_ATTR: [om.ONNXAddLayerMetatype, om.ONNXMulLayerMetatype]})

    pattern.add_edge(input_node, add_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_ARITHMETIC)
def create_linear_arithmetic():
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    linear.join_patterns(arithmetic)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_BATCH_NORM)
def create_linear_batch_norm():
    linear = linear_operations()
    batch_norm = batch_normalization_operations()
    linear.join_patterns(batch_norm)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SCALE_SHIFT)
def create_linear_scale_shift():
    linear = linear_operations()
    batch_norm = create_scale_shift_add()
    linear.join_patterns(batch_norm)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_ACTIVATIONS)
def create_linear_activations():
    linear = linear_operations()
    activations = atomic_activations_operations()
    linear.join_patterns(activations)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SWISH_SIGMOID)
def create_linear_swish_sigmoid():
    linear = linear_operations()
    swish_sigmoid = create_swish_with_sigmoid()
    linear.join_patterns(swish_sigmoid)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SWISH_HARD_SIGMOID)
def create_linear_swish_hard_sigmoid():
    linear = linear_operations()
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    linear.join_patterns(swish_hard_sigmoid)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_BATCH_NORM_ACTIVATIONS)
def create_linear_batch_norm_activations():
    linear_batch_norm = create_linear_batch_norm()
    activations = atomic_activations_operations()
    linear_batch_norm.join_patterns(activations)
    return linear_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_BATCH_NORM_SWISH_SIGMOID)
def create_linear_batch_norm_swish_sigmoid():
    linear_batch_norm = create_linear_batch_norm()
    swish_sigmoid = create_swish_with_sigmoid()
    linear_batch_norm.join_patterns(swish_sigmoid)
    return linear_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_BATCH_NORM_SWISH_HARD_SIGMOID)
def create_linear_batch_norm_swish_hard_sigmoid():
    linear_batch_norm = create_linear_batch_norm()
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    linear_batch_norm.join_patterns(swish_hard_sigmoid)
    return linear_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SCALE_SHIFT_ACTIVATIONS)
def create_linear_scale_shift_activations():
    linear_scale_shift = create_linear_scale_shift()
    activations = atomic_activations_operations()
    linear_scale_shift.join_patterns(activations)
    return linear_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SCALE_SHIFT_SWISH_SIGMOID)
def create_linear_scale_shift_swish_sigmoid():
    linear_scale_shift = create_linear_scale_shift()
    swish_sigmoid = create_swish_with_sigmoid()
    linear_scale_shift.join_patterns(swish_sigmoid)
    return linear_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SCALE_SHIFT_SWISH_HARD_SIGMOID)
def create_linear_scale_shift_swish_hard_sigmoid():
    linear_scale_shift = create_linear_scale_shift()
    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    linear_scale_shift.join_patterns(swish_hard_sigmoid)
    return linear_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_ACTIVATIONS_BATCH_NORM)
def create_linear_activations_batch_norm():
    linear_activations = create_linear_activations()
    batch_norm = batch_normalization_operations()
    linear_activations.join_patterns(batch_norm)
    return linear_activations


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SWISH_SIGMOID_BATCH_NORM)
def create_linear_swish_sigmoid_batch_norm():
    linear_swish_sigmoid = create_linear_swish_sigmoid()
    batch_norm = batch_normalization_operations()
    linear_swish_sigmoid.join_patterns(batch_norm)
    return linear_swish_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SWISH_HARD_SIGMOID_BATCH_NORM)
def create_linear_swish_hard_sigmoid_batch_norm():
    linear_swish_hard_sigmoid = create_linear_swish_hard_sigmoid()
    batch_norm = batch_normalization_operations()
    linear_swish_hard_sigmoid.join_patterns(batch_norm)
    return linear_swish_hard_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_ACTIVATIONS_SCALE_SHIFT)
def create_linear_activations_scale_shift():
    linear_activations = create_linear_activations()
    scale_shift = create_scale_shift_add()
    linear_activations.join_patterns(scale_shift)
    return linear_activations


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SWISH_SIGMOID_SCALE_SHIFT)
def create_linear_swish_sigmoid_scale_shift():
    linear_swish_sigmoid = create_linear_swish_sigmoid()
    scale_shift = create_scale_shift_add()
    linear_swish_sigmoid.join_patterns(scale_shift)
    return linear_swish_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SWISH_HARD_SIGMOID_SCALE_SHIFT)
def create_linear_swish_hard_sigmoid_scale_shift():
    linear_swish_hard_sigmoid = create_linear_swish_hard_sigmoid()
    scale_shift = create_scale_shift_add()
    linear_swish_hard_sigmoid.join_patterns(scale_shift)
    return linear_swish_hard_sigmoid


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.SCALE_SHIFT_ADD)
def create_scale_shift_add():
    pattern = GraphPattern()
    pattern_input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: '*INPUT_NODE*',
                                             GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                   GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.ONNXAddLayerMetatype})
    pattern.add_edge(pattern_input_node, mul_node)
    pattern.add_edge(mul_node, add_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_MULTIPLY_SUBTRACT)
def create_input_multiply_subtract():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: NNCFGraphNodeType.INPUT_NODE})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype})
    subtract_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SUBTRACT',
                                        GraphPattern.METATYPE_ATTR: om.ONNXSubMetatype})

    pattern.add_edge(input_node, multiply_node)
    pattern.add_edge(multiply_node, subtract_node)
    return pattern
