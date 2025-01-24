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

from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import HWFusedPatternNames
from nncf.common.utils.registry import Registry
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.metatypes.groups import ARITHMETIC_OPERATIONS
from nncf.openvino.graph.metatypes.groups import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.openvino.graph.metatypes.groups import BATCH_NORMALIZATION_OPERATIONS
from nncf.openvino.graph.metatypes.groups import ELEMENTWISE_OPERATIONS
from nncf.openvino.graph.metatypes.groups import LINEAR_OPERATIONS

OPENVINO_HW_FUSED_PATTERNS = Registry("openvino")

# BLOCK PATTERNS


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ADD_SCALE_SHIFT_OUTPUT)
def create_add_scale_shift_output() -> GraphPattern:
    pattern = GraphPattern()
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    mul_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    result_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_OUTPUT", GraphPattern.METATYPE_ATTR: om.OVResultMetatype}
    )

    pattern.add_edge(add_node_1, mul_node)
    pattern.add_edge(mul_node, add_node_2)
    pattern.add_edge(add_node_2, result_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.BATCH_INDEX)
def create_batch_index() -> GraphPattern:
    pattern = GraphPattern()
    subtract_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "SUBTRACT", GraphPattern.METATYPE_ATTR: om.OVSubtractMetatype}
    )
    multiply_node_1 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )
    multiply_node_2 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    unsqueeze_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "UNSQUEEZE", GraphPattern.METATYPE_ATTR: om.OVUnsqueezeMetatype}
    )
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node_1 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: om.OVConcatMetatype}
    )
    concat_node_2 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: om.OVConcatMetatype}
    )
    reshape_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "RESHAPE", GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype}
    )
    convolution_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONVOLUTION", GraphPattern.METATYPE_ATTR: om.OVConvolutionMetatype}
    )

    pattern.add_edge(subtract_node, multiply_node_1)
    pattern.add_edge(multiply_node_1, multiply_node_2)
    pattern.add_edge(multiply_node_2, add_node)
    pattern.add_edge(add_node, unsqueeze_node)
    pattern.add_edge(unsqueeze_node, concat_node_1)
    pattern.add_edge(concat_node_1, concat_node_2)
    pattern.add_edge(concat_node_2, reshape_node)
    pattern.add_edge(reshape_node, convolution_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.MVN_SCALE_SHIFT)
def create_mvn() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "MVN", GraphPattern.METATYPE_ATTR: om.OVMVNMetatype})
    scale_shift = create_scale_shift()

    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.NORMALIZE_L2_MULTIPLY)
def create_normalize() -> GraphPattern:
    pattern = GraphPattern()
    normalize_l2_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "NORMALIZEL2", GraphPattern.METATYPE_ATTR: om.OVNormalizeL2Metatype}
    )
    multiply_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )

    pattern.add_edge(normalize_l2_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_WITH_BIAS)
def create_biased_op() -> GraphPattern:
    pattern = GraphPattern()
    linear_node = pattern.add_node(**{GraphPattern.METATYPE_ATTR: LINEAR_OPERATIONS, GraphPattern.LABEL_ATTR: "LINEAR"})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD_BIAS", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(linear_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SCALE_SHIFT)
def create_scale_shift() -> GraphPattern:
    pattern = GraphPattern()
    mul_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    pattern.add_edge(mul_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SHIFT_SCALE)
def create_shift_scale() -> GraphPattern:
    pattern = GraphPattern()
    add_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "ADD, SUBTRACT",
            GraphPattern.METATYPE_ATTR: [om.OVAddMetatype, om.OVSubtractMetatype],
        }
    )
    mul_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "MULTIPLY, DIV",
            GraphPattern.METATYPE_ATTR: [om.OVMultiplyMetatype, om.OVDivideMetatype],
        }
    )
    pattern.add_edge(add_node, mul_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SOFTMAX_DIV)
def create_softmax_div() -> GraphPattern:
    pattern = GraphPattern()
    exp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "EXP", GraphPattern.METATYPE_ATTR: om.OVExpMetatype})
    sum_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "REDUCE_SUM", GraphPattern.METATYPE_ATTR: om.OVSumMetatype})
    divide_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "DIVIDE", GraphPattern.METATYPE_ATTR: om.OVDivideMetatype}
    )

    pattern.add_edge(exp_node, sum_node)
    pattern.add_edge(exp_node, divide_node)
    pattern.add_edge(sum_node, divide_node)
    return pattern


# ACTIVATIONS


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.HSWISH_ACTIVATION)
def create_hswish() -> GraphPattern:
    pattern = GraphPattern()
    linear_node = pattern.add_node(**{GraphPattern.METATYPE_ATTR: LINEAR_OPERATIONS, GraphPattern.LABEL_ATTR: "LINEAR"})
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD_BIAS", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "CLAMP", GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node_1 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )
    multiply_node_2 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )

    pattern.add_edge(linear_node, add_node_1)
    pattern.add_edge(add_node_1, add_node_2)
    pattern.add_edge(add_node_2, clamp_node)
    pattern.add_edge(add_node_1, multiply_node_1)
    pattern.add_edge(clamp_node, multiply_node_1)
    pattern.add_edge(multiply_node_1, multiply_node_2)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.HSWISH_ACTIVATION_V2)
def create_hswish_pattern_2() -> GraphPattern:
    pattern = GraphPattern()
    input_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "ADD, MULTIPLY, REDUCE_MEAN, SQUEEZE",
            GraphPattern.METATYPE_ATTR: [
                om.OVAddMetatype,
                om.OVMultiplyMetatype,
                om.OVReduceMeanMetatype,
                om.OVSqueezeMetatype,
            ],
        }
    )
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "CLAMP", GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node_1 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )
    multiply_node_2 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )

    pattern.add_edge(input_node, add_node)
    pattern.add_edge(add_node, clamp_node)
    pattern.add_edge(clamp_node, multiply_node_1)
    pattern.add_edge(input_node, multiply_node_2)
    pattern.add_edge(multiply_node_1, multiply_node_2)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.HSWISH_ACTIVATION_WITHOUT_DENOMINATOR)
def create_hswish_without_denominator() -> GraphPattern:
    pattern = GraphPattern()
    linear_node = pattern.add_node(**{GraphPattern.METATYPE_ATTR: LINEAR_OPERATIONS, GraphPattern.LABEL_ATTR: "LINEAR"})
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD_BIAS", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "CLAMP", GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )

    pattern.add_edge(linear_node, add_node_1)
    pattern.add_edge(add_node_1, add_node_2)
    pattern.add_edge(add_node_2, clamp_node)
    pattern.add_edge(add_node_1, multiply_node)
    pattern.add_edge(clamp_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SWISH_WITH_HARD_SIGMOID)
def create_swish_with_hardsigmoid() -> GraphPattern:
    pattern = GraphPattern()
    linear_node = pattern.add_node(**{GraphPattern.METATYPE_ATTR: LINEAR_OPERATIONS, GraphPattern.LABEL_ATTR: "LINEAR"})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD_BIAS", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    hard_sigmoid_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "HARDSIGMOID", GraphPattern.METATYPE_ATTR: om.OVHardSigmoidMetatype}
    )
    multiply_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )

    pattern.add_edge(linear_node, add_node)
    pattern.add_edge(add_node, hard_sigmoid_node)
    pattern.add_edge(add_node, multiply_node)
    pattern.add_edge(hard_sigmoid_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SOFTMAX)
def create_softmax() -> GraphPattern:
    pattern = GraphPattern()
    exp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "EXP", GraphPattern.METATYPE_ATTR: om.OVExpMetatype})
    sum_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "REDUCE_SUM", GraphPattern.METATYPE_ATTR: om.OVSumMetatype})
    power_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "POWER", GraphPattern.METATYPE_ATTR: om.OVPowerMetatype})
    multiply_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )

    pattern.add_edge(exp_node, sum_node)
    pattern.add_edge(sum_node, power_node)
    pattern.add_edge(exp_node, multiply_node)
    pattern.add_edge(power_node, multiply_node)
    return pattern


# INPUT PROCESSING


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_PROCESSING)
def create_input_convert_transpose_processing() -> GraphPattern:
    input_convert_transpose = create_input_convert_transpose()
    pattern = GraphPattern()
    pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "ADD, MULTIPLY, SUBTRACT",
            GraphPattern.METATYPE_ATTR: [om.OVAddMetatype, om.OVMultiplyMetatype, om.OVSubtractMetatype],
        }
    )

    input_convert_transpose.join_patterns(pattern)
    return input_convert_transpose


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_ADD)
def create_input_convert_transpose_reverse_add() -> GraphPattern:
    input_convert_transpose = create_input_convert_transpose()
    pattern = GraphPattern()
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SPLIT", GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: om.OVConcatMetatype}
    )
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(split_node, concat_node)
    pattern.add_edge(concat_node, add_node)
    input_convert_transpose.join_patterns(pattern)
    return input_convert_transpose


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_SCALE_SHIFT)
def create_input_convert_transpose_reverse_scale_shift() -> GraphPattern:
    pattern = GraphPattern()
    model_input = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype}
    )
    convert_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONVERT", GraphPattern.METATYPE_ATTR: om.OVConvertMetatype}
    )
    transpose_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "TRANSPOSE", GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype}
    )
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SPLIT", GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: om.OVConcatMetatype}
    )
    scale_shift = create_scale_shift()

    pattern.add_edge(model_input, convert_node)
    pattern.add_edge(convert_node, transpose_node)
    pattern.add_edge(transpose_node, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT)
def create_input_convert_transpose_scale_shift() -> GraphPattern:
    input_convert_transpose = create_input_convert_transpose()
    scale_shift = create_scale_shift()
    input_convert_transpose.join_patterns(scale_shift)
    return input_convert_transpose


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_PROCESSING)
def create_input_processing() -> GraphPattern:
    pattern = GraphPattern()
    model_input = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype}
    )
    processing_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "SUBTRACT, MULTIPLY, ADD",
            GraphPattern.METATYPE_ATTR: [om.OVSubtractMetatype, om.OVMultiplyMetatype, om.OVAddMetatype],
        }
    )

    pattern.add_edge(model_input, processing_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_REVERSE_ADD)
def create_input_reverse_add() -> GraphPattern:
    pattern = GraphPattern()
    model_input = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype}
    )
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SPLIT", GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: om.OVConcatMetatype}
    )
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(model_input, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.add_edge(concat_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_REVERSE_SCALE_SHIFT)
def create_input_reverse_scale_shift() -> GraphPattern:
    pattern = GraphPattern()
    model_input = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype}
    )
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SPLIT", GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: om.OVConcatMetatype}
    )
    scale_shift = create_scale_shift()

    pattern.add_edge(model_input, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_SCALE_SHIFT)
def create_input_scale_shift() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    scale_shift = create_scale_shift()

    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_SHIFT_SCALE)
def create_input_shift_scale() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    shift_scale = create_shift_scale()

    pattern.join_patterns(shift_scale)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_TRANSPOSE_PROCESSING)
def create_input_transpose_processing() -> GraphPattern:
    pattern = GraphPattern()
    model_input = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype}
    )
    transpose_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "TRANSPOSE", GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype}
    )
    processing_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "ADD, MULTIPLY, SUBTRACT",
            GraphPattern.METATYPE_ATTR: [om.OVAddMetatype, om.OVMultiplyMetatype, om.OVSubtractMetatype],
        }
    )

    pattern.add_edge(model_input, transpose_node)
    pattern.add_edge(transpose_node, processing_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_TRANSPOSE_REVERSE_ADD)
def create_input_transpose_reverse_add() -> GraphPattern:
    pattern = GraphPattern()
    model_input = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype}
    )
    transpose_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "TRANSPOSE", GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype}
    )
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SPLIT", GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: om.OVConcatMetatype}
    )
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(model_input, transpose_node)
    pattern.add_edge(transpose_node, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.add_edge(concat_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.INPUT_TRANSPOSE_SCALE_SHIFT)
def create_input_transpose_scale_shift() -> GraphPattern:
    pattern = GraphPattern()
    model_input = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype}
    )
    transpose_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "TRANSPOSE", GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype}
    )
    scale_shift = create_scale_shift()

    pattern.add_edge(model_input, transpose_node)
    pattern.join_patterns(scale_shift)
    return pattern


# COMBINATIONS


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ACTIVATIONS_BATCH_NORM)
def create_activations_batch_norm() -> GraphPattern:
    activations = atomic_activations_operations()
    batch_norm = batch_normalization_operations()
    activations.join_patterns(batch_norm)
    return activations


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_ACTIVATIONS)
def create_arithmetic_activations() -> GraphPattern:
    arithmetic = arithmetic_operations()
    activations = atomic_activations_operations()
    arithmetic.join_patterns(activations)
    return arithmetic


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_ACTIVATIONS_ARITHMETIC)
def create_arithmetic_activations_arithmetic() -> GraphPattern:
    arithmetic_activations = create_arithmetic_activations()
    arithmetic = arithmetic_operations()
    arithmetic_activations.join_patterns(arithmetic)
    return arithmetic_activations


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.BATCH_NORM_ACTIVATIONS)
def create_batch_norm_activations() -> GraphPattern:
    batch_norm = batch_normalization_operations()
    activations = atomic_activations_operations()
    batch_norm.join_patterns(activations)
    return batch_norm


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ACTIVATIONS)
def create_linear_activations() -> GraphPattern:
    linear = linear_operations()
    activations = atomic_activations_operations()
    linear.join_patterns(activations)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ARITHMETIC)
def create_linear_arithmetic() -> GraphPattern:
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    linear.join_patterns(arithmetic)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ARITHMETIC_ACTIVATIONS)
def create_linear_arithmetic_activations() -> GraphPattern:
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    activations = atomic_activations_operations()

    linear.join_patterns(arithmetic)
    linear.join_patterns(activations)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_SHIFT_SCALE)
def create_linear_shift_scale() -> GraphPattern:
    linear = linear_operations()
    shift_scale = create_shift_scale()
    linear.join_patterns(shift_scale)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ARITHMETIC_ACTIVATIONS_ARITHMETIC)
def create_linear_arithmetic_activations_arithmetic() -> GraphPattern:
    linear_arithmetic_activations = create_linear_arithmetic_activations()
    arithmetic = arithmetic_operations()

    linear_arithmetic_activations.join_patterns(arithmetic)
    return linear_arithmetic_activations


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_SQUEEZE_ACTIVATIONS)
def create_linear_squeeze_activation() -> GraphPattern:
    linear = linear_operations()
    squeeze = squeeze_operation()
    activations = atomic_activations_operations()

    linear.join_patterns(squeeze)
    linear.join_patterns(activations)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_SQUEEZE_ARITHMETIC_ACTIVATIONS)
def create_linear_squeeze_arithmetic_activation() -> GraphPattern:
    linear = linear_operations()
    squeeze = squeeze_operation()
    arithmetic_activations = create_arithmetic_activations()

    linear.join_patterns(squeeze)
    linear.join_patterns(arithmetic_activations)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.MVN_SCALE_SHIFT_ACTIVATIONS)
def create_mvn_scale_shift_activations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "MVN", GraphPattern.METATYPE_ATTR: om.OVMVNMetatype})
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()

    pattern.join_patterns(scale_shift)
    pattern.join_patterns(activations)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ACTIVATIONS_UNSQUEEZE_BN_SQUEEZE)
def create_linear_activations_unsqueeze_bn_squeeze():
    linear_biased = create_biased_op()
    activations = atomic_activations_operations()
    unsqueeze_op = unsqueeze_operation()
    scale_shift = create_scale_shift()
    squeeze_op = squeeze_operation()

    linear_biased.join_patterns(activations)
    linear_biased.join_patterns(unsqueeze_op)
    linear_biased.join_patterns(scale_shift)
    linear_biased.join_patterns(squeeze_op)
    return linear_biased


# DEVICE PATTERNS


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.HSWISH_ACTIVATION_CLAMP_MULTIPLY)
def create_clamp_mult_const() -> GraphPattern:
    pattern = GraphPattern()
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "CLAMP", GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )

    pattern.add_edge(clamp_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.SCALE_SHIFT_ACTIVATIONS)
def create_scale_shift_activations() -> GraphPattern:
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()
    scale_shift.join_patterns(activations)
    return scale_shift


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_SCALE_SHIFT)
def create_linear_scale_shift() -> GraphPattern:
    linear = linear_operations()
    scale_shift = create_scale_shift()
    linear.join_patterns(scale_shift)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BIASED_SCALE_SHIFT)
def create_linear_biased_scale_shift() -> GraphPattern:
    linear_biased = create_biased_op()
    scale_shift = create_scale_shift()
    linear_biased.join_patterns(scale_shift)
    return linear_biased


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ACTIVATION_SCALE_SHIFT)
def create_linear_activation_scale_shift() -> GraphPattern:
    linear_activations = create_linear_activations()
    scale_shift = create_scale_shift()

    linear_activations.join_patterns(scale_shift)
    return linear_activations


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BIASED_ACTIVATION_SCALE_SHIFT)
def create_linear_biased_activation_scale_shift() -> GraphPattern:
    linear_biased = create_biased_op()
    activations = atomic_activations_operations()
    scale_shift = create_scale_shift()

    linear_biased.join_patterns(activations)
    linear_biased.join_patterns(scale_shift)
    return linear_biased


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BATCH_TO_SPACE_SCALE_SHIFT_ACTIVATIONS)
def create_linear_batch_to_space_scale_shift_activations() -> GraphPattern:
    linear = linear_operations()
    batch_to_space = batch_to_space_operation()
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()

    linear.join_patterns(batch_to_space)
    linear.join_patterns(scale_shift)
    linear.join_patterns(activations)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BATCH_TO_SPACE_ARITHMETIC_ACTIVATIONS)
def create_linear_batch_to_space_arithmetic_activations() -> GraphPattern:
    linear = linear_operations()
    batch_to_space = batch_to_space_operation()
    arithmetic = arithmetic_operations()
    activations = atomic_activations_operations()

    linear.join_patterns(batch_to_space)
    linear.join_patterns(arithmetic)
    linear.join_patterns(activations)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ELEMENTWISE)
def create_linear_elementwise() -> GraphPattern:
    linear = linear_operations()
    elementwise = elementwise_operations()
    linear.join_patterns(elementwise)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BIASED_ELEMENTWISE)
def create_linear_biased_elementwise() -> GraphPattern:
    linear_biased = create_biased_op()
    elementwise = elementwise_operations()
    linear_biased.join_patterns(elementwise)
    return linear_biased


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ACTIVATION_ELEMENTWISE)
def create_linear_activation_elementwise() -> GraphPattern:
    linear_activations = create_linear_activations()
    elementwise = elementwise_operations()

    linear_activations.join_patterns(elementwise)
    return linear_activations


@OPENVINO_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BIASED_ACTIVATION_ELEMENTWISE)
def create_linear_biased_activation_elementwise() -> GraphPattern:
    linear_biased = create_biased_op()
    activations = atomic_activations_operations()
    elementwise = elementwise_operations()

    linear_biased.join_patterns(activations)
    linear_biased.join_patterns(elementwise)
    return linear_biased


def elementwise_operations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.METATYPE_ATTR: ELEMENTWISE_OPERATIONS, GraphPattern.LABEL_ATTR: "ELEMENTWISE"})
    return pattern


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
    return pattern


def arithmetic_operations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.METATYPE_ATTR: ARITHMETIC_OPERATIONS, GraphPattern.LABEL_ATTR: "ARITHMETIC"})
    return pattern


def squeeze_operation() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "SQUEEZE", GraphPattern.METATYPE_ATTR: om.OVSqueezeMetatype})
    return pattern


def unsqueeze_operation() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: "UNSQUEEZE", GraphPattern.METATYPE_ATTR: om.OVUnsqueezeMetatype})
    return pattern


def batch_to_space_operation() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "BATCH_TO_SPACE", GraphPattern.METATYPE_ATTR: om.OVBatchToSpaceMetatype}
    )
    return pattern


def create_input_convert_transpose() -> GraphPattern:
    pattern = GraphPattern()
    model_input = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MODEL_INPUT", GraphPattern.METATYPE_ATTR: om.OVParameterMetatype}
    )
    convert_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONVERT", GraphPattern.METATYPE_ATTR: om.OVConvertMetatype}
    )
    transpose_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "TRANSPOSE", GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype}
    )
    pattern.add_edge(model_input, convert_node)
    pattern.add_edge(convert_node, transpose_node)
    return pattern
