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

from nncf.common.graph.patterns import PatternsManager
from nncf.common.graph.patterns import GraphPattern
from nncf.common.utils.registry import Registry
from nncf.experimental.openvino_native.graph.metatypes import openvino_metatypes as om
from nncf.experimental.openvino_native.hardware.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import ELEMENTWISE_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import LINEAR_OPERATIONS

OPENVINO_HW_FUSED_PATTERNS = Registry('openvino')


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.SWISH_ACTIVATION)
def create_swish_pattern():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    swish_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SWISH',
                                     GraphPattern.METATYPE_ATTR: om.OVSwishMetatype})

    pattern.add_edge(linear_node, add_node)
    pattern.add_edge(add_node, swish_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.SE_BLOCK)
def create_se_pattern():
    pattern = GraphPattern()
    reduce_mean_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'REDUCE_MEAN',
                                           GraphPattern.METATYPE_ATTR: om.OVReduceMeanMetatype})
    linear_node_1 = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RELU, PRELU',
                                            GraphPattern.METATYPE_ATTR: [om.OVReluMetatype, om.OVPReluMetatype]})
    linear_node_2 = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SIGMOID',
                                            GraphPattern.METATYPE_ATTR: om.OVSigmoidMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(reduce_mean_node, linear_node_1)
    pattern.add_edge(linear_node_1, add_node_1)
    pattern.add_edge(add_node_1, activation_node_1)
    pattern.add_edge(activation_node_1, linear_node_2)
    pattern.add_edge(linear_node_2, add_node_2)
    pattern.add_edge(add_node_2, activation_node_2)
    pattern.add_edge(activation_node_2, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.SE_BLOCK_SWISH_ACTIVATION)
def create_se_swish_pattern():
    pattern = GraphPattern()
    reduce_mean_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'REDUCE_MEAN',
                                           GraphPattern.METATYPE_ATTR: om.OVReduceMeanMetatype})
    linear_node_1 = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SWISH',
                                            GraphPattern.METATYPE_ATTR: om.OVSwishMetatype})
    linear_node_2 = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SIGMOID',
                                            GraphPattern.METATYPE_ATTR: om.OVSigmoidMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(reduce_mean_node, linear_node_1)
    pattern.add_edge(linear_node_1, add_node_1)
    pattern.add_edge(add_node_1, activation_node_1)
    pattern.add_edge(activation_node_1, linear_node_2)
    pattern.add_edge(linear_node_2, add_node_2)
    pattern.add_edge(add_node_2, activation_node_2)
    pattern.add_edge(activation_node_2, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.OPERATION_WITH_BIAS)
def create_biased_op_pattern():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(linear_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.SCALE_SHIFT_ADD)
def create_scale_shift_add_pattern():
    pattern = GraphPattern()
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                     GraphPattern.METATYPE_ATTR: om.OVMulMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    pattern.add_edge(mul_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.ADD_SCALE_SHIFT)
def create_add_scaleshift_pattern():
    pattern = GraphPattern()
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                     GraphPattern.METATYPE_ATTR: om.OVMulMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    result_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RESULT',
                                   GraphPattern.METATYPE_ATTR: om.OVResultMetatype})

    pattern.add_edge(add_node_1, mul_node)
    pattern.add_edge(mul_node, add_node_2)
    pattern.add_edge(add_node_2, result_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.MVN_SCALE_SHIFT)
def create_mvn_pattern():
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MVN',
                        GraphPattern.METATYPE_ATTR: om.OVMVNMetatype})
    scale_shift = create_scale_shift_add_pattern()

    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.NORMALIZE_L2)
def create_normalize_pattern():
    pattern = GraphPattern()
    normalize_l2_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'NORMALIZEL2',
                                            GraphPattern.METATYPE_ATTR: om.OVNormalizeL2Metatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(normalize_l2_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_SCALE_SHIFT)
def create_input_scaleshift_pattern():
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                        GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    scale_shift = create_scale_shift_add_pattern()

    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_TRANSPOSE_SCALE_SHIFT)
def create_input_transpose_scaleshift_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    scale_shift = create_scale_shift_add_pattern()

    pattern.add_edge(input_node, transpose_node)
    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT)
def create_input_convert_transpose_scaleshift_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    convert_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONVERT',
                                       GraphPattern.METATYPE_ATTR: om.OVConvertMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    scale_shift = create_scale_shift_add_pattern()

    pattern.add_edge(input_node, convert_node)
    pattern.add_edge(convert_node, transpose_node)
    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_ADD)
def create_input_add_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(input_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_SUBTRACT)
def create_input_subtract_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    subtract_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SUBTRACT',
                                        GraphPattern.METATYPE_ATTR: om.OVSubMetatype})

    pattern.add_edge(input_node, subtract_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_TRANSPOSE_ADD)
def create_input_transpose_add_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(input_node, transpose_node)
    pattern.add_edge(transpose_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_CONVERT_TRANSPOSE_ADD)
def create_input_convert_transpose_add_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    convert_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONVERT',
                                       GraphPattern.METATYPE_ATTR: om.OVConvertMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(input_node, convert_node)
    pattern.add_edge(convert_node, transpose_node)
    pattern.add_edge(transpose_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_MULTIPLY)
def create_input_mul_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(input_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_TRANSPOSE_MULTIPLY)
def create_input_transpose_mul_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(input_node, transpose_node)
    pattern.add_edge(transpose_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_CONVERT_TRANSPOSE_MULTIPLY)
def create_input_convert_transpose_mul_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    convert_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONVERT',
                                       GraphPattern.METATYPE_ATTR: om.OVConvertMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(input_node, convert_node)
    pattern.add_edge(convert_node, transpose_node)
    pattern.add_edge(transpose_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_REVERSE_INPUT_CHANNELS_SCALE_SHIFT)
def create_input_reverse_input_channel_scaleshift_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SPLIT',
                                     GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                      GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    scale_shift = create_scale_shift_add_pattern()

    pattern.add_edge(input_node, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_CONVERT_TRANSPOSE_REVERSE_INPUT_CHANNELS_SCALE_SHIFT)
def create_input_convert_transpose_reverse_input_channel_scaleshift_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    convert_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONVERT',
                                       GraphPattern.METATYPE_ATTR: om.OVConvertMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SPLIT',
                                     GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                      GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    scale_shift = create_scale_shift_add_pattern()

    pattern.add_edge(input_node, convert_node)
    pattern.add_edge(convert_node, transpose_node)
    pattern.add_edge(transpose_node, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_REVERSE_INPUT_CHANNELS_ADD)
def create_input_reverse_input_channel_add_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SPLIT',
                                     GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                      GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(input_node, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.add_edge(concat_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_TRANSPOSE_REVERSE_INPUT_CHANNELS_ADD)
def create_input_transpose_reverse_input_channel_add_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SPLIT',
                                     GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                      GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(input_node, transpose_node)
    pattern.add_edge(transpose_node, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.add_edge(concat_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.INPUT_CONVERT_TRANSPOSE_REVERSE_INPUT_CHANNELS_ADD)
def create_input_convert_transpose_reverse_input_channel_add_pattern():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'INPUT',
                                     GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    convert_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONVERT',
                                       GraphPattern.METATYPE_ATTR: om.OVConvertMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SPLIT',
                                     GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                      GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(input_node, convert_node)
    pattern.add_edge(convert_node, transpose_node)
    pattern.add_edge(transpose_node, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.add_edge(concat_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.SOFTMAX)
def create_softmax_pattern():
    pattern = GraphPattern()
    exp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'EXP',
                                   GraphPattern.METATYPE_ATTR: om.OVExpMetatype})
    sum_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'REDUCE_SUM',
                                   GraphPattern.METATYPE_ATTR: om.OVSumMetatype})
    power_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'POWER',
                                     GraphPattern.METATYPE_ATTR: om.OVPowerMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(exp_node, sum_node)
    pattern.add_edge(sum_node, power_node)
    pattern.add_edge(exp_node, multiply_node)
    pattern.add_edge(power_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.SOFTMAX_DIV)
def create_softmax_div_pattern():
    pattern = GraphPattern()
    exp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'EXP',
                                   GraphPattern.METATYPE_ATTR: om.OVExpMetatype})
    sum_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'REDUCE_SUM',
                                   GraphPattern.METATYPE_ATTR: om.OVSumMetatype})
    divide_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'DIVIDE',
                                      GraphPattern.METATYPE_ATTR: om.OVDivMetatype})

    pattern.add_edge(exp_node, sum_node)
    pattern.add_edge(exp_node, divide_node)
    pattern.add_edge(sum_node, divide_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.SOFTMAX_RESHAPE_MATMUL)
def create_softmax_reshape_matmul_pattern():
    pattern = GraphPattern()
    softmax_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SOFTMAX',
                                       GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    reshape_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RESHAPE',
                                         GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    reshape_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RESHAPE',
                                         GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    matmul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL',
                                      GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})

    pattern.add_edge(softmax_node, reshape_node_1)
    pattern.add_edge(add_node, reshape_node_2)
    pattern.add_edge(reshape_node_2, transpose_node)
    pattern.add_edge(transpose_node, matmul_node)
    pattern.add_edge(reshape_node_1, matmul_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.SOFTMAX_RESHAPE_TRANSPOSE_MATMUL)
def create_softmax_reshape_transpose_matmul_pattern():
    pattern = GraphPattern()
    softmax_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SOFTMAX',
                                       GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    reshape_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RESHAPE',
                                       GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    matmul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL',
                                      GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})

    pattern.add_edge(add_node, reshape_node)
    pattern.add_edge(reshape_node, transpose_node)
    pattern.add_edge(transpose_node, matmul_node)
    pattern.add_edge(softmax_node, matmul_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.STABLE_DIFFUSION)
def create_stable_diffusion_pattern():
    pattern = GraphPattern()
    softmax_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SOFTMAX',
                                       GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    reshape_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RESHAPE',
                                         GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    reshape_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RESHAPE',
                                         GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype})
    matmul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL',
                                      GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})

    pattern.add_edge(reshape_node_1, transpose_node)
    pattern.add_edge(transpose_node, reshape_node_2)
    pattern.add_edge(reshape_node_2, matmul_node)
    pattern.add_edge(softmax_node, matmul_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.SOFTMAX_RESHAPE_TRANSPOSE_GATHER_MATMUL)
def create_softmax_reshape_transpose_gather_matmul_pattern():
    pattern = GraphPattern()
    softmax_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SOFTMAX',
                                       GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    reshape_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RESHAPE',
                                       GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    gather_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'GATHER',
                                      GraphPattern.METATYPE_ATTR: om.OVGatherMetatype})
    matmul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL',
                                      GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})

    pattern.add_edge(add_node, reshape_node)
    pattern.add_edge(reshape_node, transpose_node)
    pattern.add_edge(transpose_node, gather_node)
    pattern.add_edge(softmax_node, matmul_node)
    pattern.add_edge(gather_node, matmul_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.HSWISH_ACTIVATION_WITHOUT_DENOMINATOR)
def create_hswish_without_denominator_pattern():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CLAMP',
                                     GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(linear_node, add_node_1)
    pattern.add_edge(add_node_1, add_node_2)
    pattern.add_edge(add_node_2, clamp_node)
    pattern.add_edge(add_node_1, multiply_node)
    pattern.add_edge(clamp_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.HSWISH_ACTIVATION)
def create_hswish_pattern():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CLAMP',
                                     GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMulMetatype})
    multiply_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(linear_node, add_node_1)
    pattern.add_edge(add_node_1, add_node_2)
    pattern.add_edge(add_node_2, clamp_node)
    pattern.add_edge(add_node_1, multiply_node_1)
    pattern.add_edge(clamp_node, multiply_node_1)
    pattern.add_edge(multiply_node_1, multiply_node_2)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.HSWISH_ACTIVATION_V2)
def create_hswish_pattern_2():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD, MULTIPLY, REDUCE_MEAN, SQUEEZE',
                                     GraphPattern.METATYPE_ATTR: [om.OVAddMetatype,
                                                                  om.OVMulMetatype,
                                                                  om.OVReduceMeanMetatype,
                                                                  om.OVSqueezeMetatype]})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CLAMP',
                                     GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMulMetatype})
    multiply_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(input_node, add_node)
    pattern.add_edge(add_node, clamp_node)
    pattern.add_edge(clamp_node, multiply_node_1)
    pattern.add_edge(input_node, multiply_node_2)
    pattern.add_edge(multiply_node_1, multiply_node_2)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.FC_BN_HSWISH_ACTIVATION)
def create_fc_bn_hswish_pattern():
    pattern = GraphPattern()
    unsqueeze_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'UNSQUEEZE',
                                         GraphPattern.METATYPE_ATTR: om.OVUnsqueezeMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    squeeze_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SQUEEZE',
                                       GraphPattern.METATYPE_ATTR: om.OVSqueezeMetatype})

    pattern.add_edge(unsqueeze_node, multiply_node)
    pattern.add_edge(multiply_node, add_node)
    pattern.add_edge(add_node, squeeze_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.BATCH_INDEX)
def create_batch_index_pattern():
    pattern = GraphPattern()
    subtract_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SUBTRACT',
                                        GraphPattern.METATYPE_ATTR: om.OVSubMetatype})
    multiply_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMulMetatype})
    multiply_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMulMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    unsqueeze_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'UNSQUEEZE',
                                         GraphPattern.METATYPE_ATTR: om.OVUnsqueezeMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                        GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    concat_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                        GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    reshape_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RESHAPE',
                                       GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype})
    convolution_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONVOLUTION',
                                           GraphPattern.METATYPE_ATTR: om.OVConvolutionMetatype})

    pattern.add_edge(subtract_node, multiply_node_1)
    pattern.add_edge(multiply_node_1, multiply_node_2)
    pattern.add_edge(multiply_node_2, add_node)
    pattern.add_edge(add_node, unsqueeze_node)
    pattern.add_edge(unsqueeze_node, concat_node_1)
    pattern.add_edge(concat_node_1, concat_node_2)
    pattern.add_edge(concat_node_2, reshape_node)
    pattern.add_edge(reshape_node, convolution_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.EXPERIMENTAL_DETECTRON_DETECTION_OUTPUT_ADD)
def create_experimentaldetectrondetectionoutput_add_pattern():
    pass


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.EXPERIMENTAL_DETECTRON_ROI_FEATURE_EXTRACTOR_ADD)
def create_experimentaldetectronroifeatureextractor_add_pattern():
    pass


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.EQUAL_LOGICALNOT)
def create_equal_logicalnot_pattern():
    pattern = GraphPattern()
    equal_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'EQUAL',
                                     GraphPattern.METATYPE_ATTR: om.OVEqualMetatype})
    logical_not_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'LOGICAL_NOT',
                                        GraphPattern.METATYPE_ATTR: om.OVNotMetatype})

    pattern.add_edge(equal_node, logical_not_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_OPERATIONS)
def linear_operations():
    pattern = GraphPattern()
    pattern.add_node(**LINEAR_OPERATIONS)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.BATCH_NORMALIZATION_OPERATIONS)
def batch_normalization_operations():
    pattern = GraphPattern()
    pattern.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.ATOMIC_ACTIVATIONS_OPERATIONS)
def atomic_activations_operations():
    pattern = GraphPattern()
    pattern.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.ARITHMETIC_OPERATIONS)
def arithmetic_operations():
    pattern = GraphPattern()
    pattern.add_node(**ARITHMETIC_OPERATIONS)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.HSWISH_ACTIVATION_CLAMP_MULTIPLY)
def create_clamp_mult_const_pattern():
    pattern = GraphPattern()
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CLAMP',
                                     GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMulMetatype})

    pattern.add_edge(clamp_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_SCALE_SHIFT)
def create_linear_scale_shift():
    linear = linear_operations()
    scale_shift = create_add_scaleshift_pattern()
    return linear.join_patterns(scale_shift)


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_BIASED_SCALE_SHIFT)
def create_linear__biased_scale_shift():
    linear_biased = create_biased_op_pattern()
    scale_shift = create_add_scaleshift_pattern()
    return linear_biased.join_patterns(scale_shift)


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_ACTIVATION_SCALE_SHIFT)
def create_linear_biased_scale_shift():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    activation_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RELU, PRELU, SIGMOID',
                                          GraphPattern.METATYPE_ATTR: [om.OVReluMetatype,
                                                                       om.OVPReluMetatype,
                                                                       om.OVSigmoidMetatype]})
    scale_shift = create_add_scaleshift_pattern()

    pattern.add_edge(linear_node, activation_node)
    return pattern.join_patterns(scale_shift)


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_BIASED_ACTIVATION_SCALE_SHIFT)
def create_linear_biased_activation_scale_shift():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RELU, PRELU, SIGMOID',
                                          GraphPattern.METATYPE_ATTR: [om.OVReluMetatype,
                                                                       om.OVPReluMetatype,
                                                                       om.OVSigmoidMetatype]})

    scale_shift = create_add_scaleshift_pattern()

    pattern.add_edge(linear_node, add_node)
    pattern.add_edge(add_node, activation_node)

    return pattern.join_patterns(scale_shift)


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_ELEMENTWISE)
def create_linear_elementwise():
    linear = linear_operations()
    elementwise = elementwise_operations()
    return linear.join_patterns(elementwise)


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_BIASED_ELEMENTWISE)
def create_linear_biased_elementwise():
    linear_biased = create_biased_op_pattern()
    elementwise = elementwise_operations()
    return linear_biased.join_patterns(elementwise)


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_ACTIVATION_ELEMENTWISE)
def create_linear_biased_elementwise():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    activation_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RELU, PRELU, SIGMOID',
                                          GraphPattern.METATYPE_ATTR: [om.OVReluMetatype,
                                                                       om.OVPReluMetatype,
                                                                       om.OVSigmoidMetatype]})
    elementwise = elementwise_operations()

    pattern.add_edge(linear_node, activation_node)
    return pattern.join_patterns(elementwise)


@OPENVINO_HW_FUSED_PATTERNS.register(PatternsManager.LINEAR_BIASED_ACTIVATION_ELEMENTWISE)
def create_linear_biased_activation_elementwise():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RELU, PRELU, SIGMOID',
                                          GraphPattern.METATYPE_ATTR: [om.OVReluMetatype,
                                                                       om.OVPReluMetatype,
                                                                       om.OVSigmoidMetatype]})

    elementwise = elementwise_operations()

    pattern.add_edge(linear_node, add_node)
    pattern.add_edge(add_node, activation_node)

    return pattern.join_patterns(elementwise)


def elementwise_operations():
    pattern = GraphPattern()
    pattern.add_node(**ELEMENTWISE_OPERATIONS)
    return pattern
