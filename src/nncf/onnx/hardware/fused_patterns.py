# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import HWFusedPatternNames
from nncf.common.utils.registry import Registry
from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.graph.metatypes.groups import ARITHMETIC_OPERATIONS
from nncf.onnx.graph.metatypes.groups import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.onnx.graph.metatypes.groups import BATCH_NORMALIZATION_OPERATIONS
from nncf.onnx.graph.metatypes.groups import LINEAR_OPERATIONS

ONNX_HW_FUSED_PATTERNS = Registry("onnx")

# BLOCK PATTERNS


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.MVN)
def create_mvn() -> GraphPattern:
    pattern = GraphPattern()
    pattern_input_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "*INPUT_NODE*", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    reduce_mean_node_1 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "REDUCE_MEAN_1", GraphPattern.METATYPE_ATTR: om.ONNXReduceMeanMetatype}
    )
    sub_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "SUBTRACT",
            GraphPattern.METATYPE_ATTR: [om.ONNXSubMetatype],
        }
    )
    pow_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "POW",
            GraphPattern.METATYPE_ATTR: [om.ONNXPowMetatype],
        }
    )
    reduce_mean_node_2 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "REDUCE_MEAN_2", GraphPattern.METATYPE_ATTR: om.ONNXReduceMeanMetatype}
    )
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.ONNXAddLayerMetatype})
    sqrt_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SQRT", GraphPattern.METATYPE_ATTR: om.ONNXSqrtMetatype})
    div_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "DIV", GraphPattern.METATYPE_ATTR: om.ONNXDivLayerMetatype})

    pattern.add_edge(pattern_input_node, reduce_mean_node_1)
    pattern.add_edge(reduce_mean_node_1, sub_node)
    pattern.add_edge(pattern_input_node, sub_node)
    pattern.add_edge(sub_node, pow_node)
    pattern.add_edge(pow_node, reduce_mean_node_2)
    pattern.add_edge(reduce_mean_node_2, add_node)
    pattern.add_edge(add_node, sqrt_node)
    pattern.add_edge(sqrt_node, div_node)
    pattern.add_edge(sub_node, div_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.MVN_SCALE_SHIFT)
def create_mvn_scale_shift() -> GraphPattern:
    mvn = create_mvn()
    scale_shift = create_scale_shift()

    mvn.join_patterns(scale_shift)
    return mvn


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.GELU)
def create_gelu() -> GraphPattern:
    pattern = GraphPattern()
    pattern_input_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "*INPUT_NODE*", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    div_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "DIV",
            GraphPattern.METATYPE_ATTR: [om.ONNXDivLayerMetatype, om.ONNXMulLayerMetatype],
        }
    )
    erf_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "ERF",
            GraphPattern.METATYPE_ATTR: om.ONNXErfMetatype,
        }
    )
    add_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "ADD",
            GraphPattern.METATYPE_ATTR: [om.ONNXAddLayerMetatype, om.ONNXSubMetatype],
        }
    )
    mul_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "MUL",
            GraphPattern.METATYPE_ATTR: [om.ONNXMulLayerMetatype, om.ONNXDivLayerMetatype],
        }
    )
    pattern.add_edge(pattern_input_node, div_node)
    pattern.add_edge(div_node, erf_node)
    pattern.add_edge(erf_node, add_node)
    pattern.add_edge(add_node, mul_node)
    pattern.add_edge(pattern_input_node, mul_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SCALE_SHIFT)
def create_scale_shift() -> GraphPattern:
    pattern = GraphPattern()
    pattern_input_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "*INPUT_NODE*", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    mul_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype}
    )
    add_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "ADD, SUBTRACT",
            GraphPattern.METATYPE_ATTR: [om.ONNXAddLayerMetatype, om.ONNXSubMetatype],
        }
    )
    pattern.add_edge(pattern_input_node, mul_node)
    pattern.add_edge(mul_node, add_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SHIFT_SCALE)
def create_shift_scale() -> GraphPattern:
    pattern = GraphPattern()
    add_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "ADD, SUBTRACT",
            GraphPattern.METATYPE_ATTR: [om.ONNXAddLayerMetatype, om.ONNXSubMetatype],
        }
    )
    mul_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "MULTIPLY, DIV",
            GraphPattern.METATYPE_ATTR: [om.ONNXMulLayerMetatype, om.ONNXDivLayerMetatype],
        }
    )

    pattern.add_edge(add_node, mul_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SWISH_WITH_SIGMOID)
def create_swish_with_sigmoid() -> GraphPattern:
    pattern = GraphPattern()
    pattern_input_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "*INPUT_NODE*", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    sigmoid_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "SIGMOID", GraphPattern.METATYPE_ATTR: om.ONNXSigmoidMetatype}
    )
    mul_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype}
    )

    pattern.add_edge(pattern_input_node, sigmoid_node)
    pattern.add_edge(sigmoid_node, mul_node)
    pattern.add_edge(pattern_input_node, mul_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SWISH_WITH_HARD_SIGMOID)
def create_swish_with_hard_sigmoid() -> GraphPattern:
    pattern = GraphPattern()
    pattern_input_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "*INPUT_NODE*", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    hard_sigmoid_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "HARD_SIGMOID", GraphPattern.METATYPE_ATTR: om.ONNXHardSigmoidMetatype}
    )
    mul_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype}
    )

    pattern.add_edge(pattern_input_node, hard_sigmoid_node)
    pattern.add_edge(hard_sigmoid_node, mul_node)
    pattern.add_edge(pattern_input_node, mul_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.HSWISH_ACTIVATION_WITHOUT_DENOMINATOR)
def create_hswish_without_denominator() -> GraphPattern:
    pattern = GraphPattern()
    any_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "ANY", GraphPattern.METATYPE_ATTR: GraphPattern.ANY_PATTERN_NODE_TYPE}
    )
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.ONNXAddLayerMetatype})
    relu_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "RELU", GraphPattern.METATYPE_ATTR: om.ONNXReluMetatype})
    multiply_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype}
    )

    pattern.add_edge(any_node, add_node)
    pattern.add_edge(add_node, relu_node)
    pattern.add_edge(relu_node, multiply_node)
    pattern.add_edge(any_node, multiply_node)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.HSWISH_ACTIVATION)
def create_hswish() -> GraphPattern:
    div_pattern = GraphPattern()
    hswish = create_hswish_without_denominator()
    div_pattern.add_node(**{GraphPattern.LABEL_ATTR: "DIV", GraphPattern.METATYPE_ATTR: om.ONNXDivLayerMetatype})
    hswish.join_patterns(div_pattern)
    return hswish


# INPUT PROCESSING


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_SCALE_SHIFT)
def create_input_scale_shift() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: InputNoopMetatype})
    scale_shift = create_scale_shift()

    pattern.join_patterns(scale_shift)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_SHIFT_SCALE)
def create_input_shift_scale() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: InputNoopMetatype})
    shift_scale = create_shift_scale()

    pattern.join_patterns(shift_scale)
    return pattern


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_PROCESSING)
def create_input_add() -> GraphPattern:
    pattern = GraphPattern()
    input_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: InputNoopMetatype}
    )
    add_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "ADD, MULTIPLY",
            GraphPattern.METATYPE_ATTR: [om.ONNXAddLayerMetatype, om.ONNXMulLayerMetatype],
        }
    )

    pattern.add_edge(input_node, add_node)
    return pattern


# COMBINATIONS


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ACTIVATIONS_BATCH_NORM)
def create_activations_batch_norm() -> GraphPattern:
    activations = atomic_activations_operations()
    batch_norm = batch_normalization_operations()
    activations.join_patterns(batch_norm)
    return activations


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ACTIVATIONS_SCALE_SHIFT)
def create_activations_scale_shift() -> GraphPattern:
    activations = atomic_activations_operations()
    scale_shift = create_scale_shift()
    activations.join_patterns(scale_shift)
    return activations


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_BATCH_NORM)
def create_arithmetic_batch_norm() -> GraphPattern:
    arithmetic = arithmetic_operations()
    batch_norm = batch_normalization_operations()
    arithmetic.join_patterns(batch_norm)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_ACTIVATIONS)
def create_arithmetic_activations() -> GraphPattern:
    arithmetic = arithmetic_operations()
    activations = atomic_activations_operations()
    arithmetic.join_patterns(activations)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_SCALE_SHIFT)
def create_arithmetic_scale_shift() -> GraphPattern:
    arithmetic = arithmetic_operations()
    scale_shift = create_scale_shift()
    arithmetic.join_patterns(scale_shift)
    return arithmetic


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_BATCH_NORM_ACTIVATIONS)
def create_arithmetic_batch_norm_activations() -> GraphPattern:
    arithmetic_batch_norm = create_arithmetic_batch_norm()
    activations = atomic_activations_operations()
    arithmetic_batch_norm.join_patterns(activations)
    return arithmetic_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_SCALE_SHIFT_ACTIVATIONS)
def create_arithmetic_scale_shift_activations() -> GraphPattern:
    arithmetic_scale_shift = create_arithmetic_scale_shift()
    activations = atomic_activations_operations()
    arithmetic_scale_shift.join_patterns(activations)
    return arithmetic_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_ACTIVATIONS_BATCH_NORM)
def create_arithmetic_activations_batch_norm() -> GraphPattern:
    arithmetic_activations = create_arithmetic_activations()
    batch_norm = batch_normalization_operations()
    arithmetic_activations.join_patterns(batch_norm)
    return arithmetic_activations


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_ACTIVATIONS_SCALE_SHIFT)
def create_arithmetic_activations_scale_shift() -> GraphPattern:
    arithmetic_activations = create_arithmetic_activations()
    scale_shift = create_scale_shift()
    arithmetic_activations.join_patterns(scale_shift)
    return arithmetic_activations


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.BATCH_NORM_ACTIVATIONS)
def create_batch_norm_activations() -> GraphPattern:
    batch_norm = batch_normalization_operations()
    activations = atomic_activations_operations()
    batch_norm.join_patterns(activations)
    return batch_norm


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SCALE_SHIFT_ACTIVATIONS)
def create_scale_shift_activations() -> GraphPattern:
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()
    scale_shift.join_patterns(activations)
    return scale_shift


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ARITHMETIC)
def create_linear_arithmetic() -> GraphPattern:
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    linear.join_patterns(arithmetic)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BATCH_NORM)
def create_linear_batch_norm() -> GraphPattern:
    linear = linear_operations()
    batch_norm = batch_normalization_operations()
    linear.join_patterns(batch_norm)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ACTIVATIONS)
def create_linear_activations() -> GraphPattern:
    linear = linear_operations()
    activations = atomic_activations_operations()
    linear.join_patterns(activations)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BATCH_NORM_ACTIVATIONS)
def create_linear_batch_norm_activations() -> GraphPattern:
    linear_batch_norm = create_linear_batch_norm()
    activations = atomic_activations_operations()
    linear_batch_norm.join_patterns(activations)
    return linear_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_SCALE_SHIFT_ACTIVATIONS)
def create_linear_scale_shift_activations() -> GraphPattern:
    linear_scale_shift = create_linear_scale_shift()
    activations = atomic_activations_operations()
    linear_scale_shift.join_patterns(activations)
    return linear_scale_shift


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ACTIVATIONS_BATCH_NORM)
def create_linear_activations_batch_norm() -> GraphPattern:
    linear_activations = create_linear_activations()
    batch_norm = batch_normalization_operations()
    linear_activations.join_patterns(batch_norm)
    return linear_activations


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ACTIVATIONS_SCALE_SHIFT)
def create_linear_activations_scale_shift() -> GraphPattern:
    linear_activations = create_linear_activations()
    scale_shift = create_scale_shift()
    linear_activations.join_patterns(scale_shift)
    return linear_activations


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BATCH_NORM_SCALE_SHIFT_ACTIVATIONS)
def create_linear_bn_scale_shift_activation() -> GraphPattern:
    linear_batch_norm = create_linear_batch_norm()
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()

    linear_batch_norm.join_patterns(scale_shift)
    linear_batch_norm.join_patterns(activations)
    return linear_batch_norm


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_SQUEEZE_ACTIVATIONS)
def create_linear_squeeze_activation() -> GraphPattern:
    linear = linear_operations()
    squeeze = squeeze_operation()
    activations = atomic_activations_operations()

    linear.join_patterns(squeeze)
    linear.join_patterns(activations)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_SQUEEZE_ARITHMETIC_ACTIVATIONS)
def create_linear_squeeze_arithmetic_activation() -> GraphPattern:
    linear = linear_operations()
    squeeze = squeeze_operation()
    arithmetic_activations = create_arithmetic_activations()

    linear.join_patterns(squeeze)
    linear.join_patterns(arithmetic_activations)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.BATCH_NORM_SCALE_SHIFT_ACTIVATIONS)
def create_bn_scale_shift_activation() -> GraphPattern:
    batch_norm = batch_normalization_operations()
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()

    batch_norm.join_patterns(scale_shift)
    batch_norm.join_patterns(activations)
    return batch_norm


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ARITHMETIC_ACTIVATIONS)
def create_linear_arithmetic_activations() -> GraphPattern:
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    activations = atomic_activations_operations()

    linear.join_patterns(arithmetic)
    linear.join_patterns(activations)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_SHIFT_SCALE)
def create_linear_shift_scale() -> GraphPattern:
    linear = linear_operations()
    shift_scale = create_shift_scale()

    linear.join_patterns(shift_scale)
    return linear


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ARITHMETIC_ACTIVATIONS_ARITHMETIC)
def create_linear_arithmetic_activations_arithmetic() -> GraphPattern:
    linear_arithmetic_activations = create_linear_arithmetic_activations()
    arithmetic = arithmetic_operations()

    linear_arithmetic_activations.join_patterns(arithmetic)
    return linear_arithmetic_activations


# DEVICE PATTERNS


@ONNX_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_SCALE_SHIFT)
def create_linear_scale_shift() -> GraphPattern:
    linear = linear_operations()
    batch_norm = create_scale_shift()
    linear.join_patterns(batch_norm)
    return linear


def linear_operations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.METATYPE_ATTR: LINEAR_OPERATIONS, GraphPattern.LABEL_ATTR: "LINEAR"})
    return pattern


def batch_normalization_operations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(
        **{GraphPattern.METATYPE_ATTR: BATCH_NORMALIZATION_OPERATIONS, GraphPattern.LABEL_ATTR: "BATCH_NORMALIZATION"}
    )
    return pattern


def atomic_activations_operations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(
        **{GraphPattern.METATYPE_ATTR: ATOMIC_ACTIVATIONS_OPERATIONS, GraphPattern.LABEL_ATTR: "ATOMIC_ACTIVATIONS"}
    )

    swish_sigmoid = create_swish_with_sigmoid()
    pattern.add_pattern_alternative(swish_sigmoid)

    swish_hard_sigmoid = create_swish_with_hard_sigmoid()
    pattern.add_pattern_alternative(swish_hard_sigmoid)

    hswish = create_hswish()
    pattern.add_pattern_alternative(hswish)

    hswish_without_denominator = create_hswish_without_denominator()
    pattern.add_pattern_alternative(hswish_without_denominator)

    gelu = create_gelu()
    pattern.add_pattern_alternative(gelu)
    return pattern


def arithmetic_operations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.METATYPE_ATTR: ARITHMETIC_OPERATIONS, GraphPattern.LABEL_ATTR: "ARITHMETIC"})
    return pattern


def squeeze_operation() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "SQUEEZE", GraphPattern.METATYPE_ATTR: om.ONNXSqueezeMetatype})
    return pattern
