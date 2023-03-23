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

from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import PatternNames
from nncf.common.utils.registry import Registry
from nncf.experimental.openvino_native.graph.metatypes import openvino_metatypes as om
from nncf.experimental.openvino_native.hardware.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import ELEMENTWISE_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import LINEAR_OPERATIONS

OPENVINO_HW_FUSED_PATTERNS = Registry('openvino')

# BLOCK PATTERNS


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.ADD_SCALE_SHIFT_OUTPUT)
def create_add_scale_shift_output():
    pattern = GraphPattern()
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                   GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    result_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_OUTPUT',
                                   GraphPattern.METATYPE_ATTR: om.OVResultMetatype})

    pattern.add_edge(add_node_1, mul_node)
    pattern.add_edge(mul_node, add_node_2)
    pattern.add_edge(add_node_2, result_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.BATCH_INDEX)
def create_batch_index():
    pattern = GraphPattern()
    subtract_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SUBTRACT',
                                        GraphPattern.METATYPE_ATTR: om.OVSubtractMetatype})
    multiply_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})
    multiply_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})
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


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.MVN_SCALE_SHIFT)
def create_mvn():
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MVN',
                        GraphPattern.METATYPE_ATTR: om.OVMVNMetatype})
    scale_shift = create_scale_shift()

    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.NORMALIZE_L2_MULTIPLY)
def create_normalize():
    pattern = GraphPattern()
    normalize_l2_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'NORMALIZEL2',
                                            GraphPattern.METATYPE_ATTR: om.OVNormalizeL2Metatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})

    pattern.add_edge(normalize_l2_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_WITH_BIAS)
def create_biased_op():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(linear_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.SCALE_SHIFT)
def create_scale_shift():
    pattern = GraphPattern()
    mul_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                   GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    pattern.add_edge(mul_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.SE_BLOCK)
def create_se_block():
    pattern = GraphPattern()
    reduce_mean_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'REDUCE_MEAN',
                                           GraphPattern.METATYPE_ATTR: om.OVReduceMeanMetatype})
    linear_node_1 = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'RELU, PRELU, SWISH',
                                            GraphPattern.METATYPE_ATTR: [om.OVReluMetatype,
                                                                         om.OVPReluMetatype,
                                                                         om.OVSwishMetatype]})
    linear_node_2 = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SIGMOID',
                                            GraphPattern.METATYPE_ATTR: om.OVSigmoidMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})

    pattern.add_edge(reduce_mean_node, linear_node_1)
    pattern.add_edge(linear_node_1, add_node_1)
    pattern.add_edge(add_node_1, activation_node_1)
    pattern.add_edge(activation_node_1, linear_node_2)
    pattern.add_edge(linear_node_2, add_node_2)
    pattern.add_edge(add_node_2, activation_node_2)
    pattern.add_edge(activation_node_2, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.STABLE_DIFFUSION)
def create_stable_diffusion():
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


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.SOFTMAX_DIV)
def create_softmax_div():
    pattern = GraphPattern()
    exp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'EXP',
                                   GraphPattern.METATYPE_ATTR: om.OVExpMetatype})
    sum_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'REDUCE_SUM',
                                   GraphPattern.METATYPE_ATTR: om.OVSumMetatype})
    divide_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'DIVIDE',
                                      GraphPattern.METATYPE_ATTR: om.OVDivideMetatype})

    pattern.add_edge(exp_node, sum_node)
    pattern.add_edge(exp_node, divide_node)
    pattern.add_edge(sum_node, divide_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.SOFTMAX_RESHAPE_MATMUL)
def create_softmax_reshape_matmul():
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


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.SOFTMAX_RESHAPE_TRANSPOSE_MATMUL)
def create_softmax_reshape_transpose_matmul():
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


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.SOFTMAX_RESHAPE_TRANSPOSE_GATHER_MATMUL)
def create_softmax_reshape_transpose_gather_matmul():
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


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.EQUAL_LOGICALNOT)
def create_equal_logicalnot():
    pattern = GraphPattern()
    equal_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'EQUAL',
                                     GraphPattern.METATYPE_ATTR: om.OVEqualMetatype})
    logical_not_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'LOGICAL_NOT',
                                        GraphPattern.METATYPE_ATTR: om.OVLogicalNotMetatype})

    pattern.add_edge(equal_node, logical_not_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.FC_BN_HSWISH_ACTIVATION)
def create_fc_bn_hswish():
    pattern = GraphPattern()
    unsqueeze_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'UNSQUEEZE',
                                         GraphPattern.METATYPE_ATTR: om.OVUnsqueezeMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    squeeze_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SQUEEZE',
                                       GraphPattern.METATYPE_ATTR: om.OVSqueezeMetatype})

    pattern.add_edge(unsqueeze_node, multiply_node)
    pattern.add_edge(multiply_node, add_node)
    pattern.add_edge(add_node, squeeze_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.MATMUL_SOFTMAX_MATMUL)
def create_matmul_softmax_matmul():
    pattern = GraphPattern()
    softmax_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SOFTMAX',
                                    GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    mat_mul_1_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL_1',
                                      GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})
    mat_mul_2_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL_2',
                                      GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})

    any_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ANY',
                                GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})

    pattern.add_edge(mat_mul_1_1, softmax_1)
    pattern.add_edge(softmax_1, mat_mul_2_1)
    pattern.add_edge(any_1, mat_mul_2_1)

    softmax_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SOFTMAX',
                                    GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    add_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    mat_mul_1_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL_1',
                                      GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})
    mat_mul_2_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MATMUL_2',
                                      GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})

    any_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ANY',
                                GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE})

    pattern.add_edge(mat_mul_1_2, add_2)
    pattern.add_edge(add_2, softmax_2)
    pattern.add_edge(softmax_2, mat_mul_2_2)
    pattern.add_edge(any_2, mat_mul_2_2)

    return pattern


# ACTIVATIONS


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.HSWISH_ACTIVATION)
def create_hswish():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CLAMP',
                                     GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})
    multiply_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})

    pattern.add_edge(linear_node, add_node_1)
    pattern.add_edge(add_node_1, add_node_2)
    pattern.add_edge(add_node_2, clamp_node)
    pattern.add_edge(add_node_1, multiply_node_1)
    pattern.add_edge(clamp_node, multiply_node_1)
    pattern.add_edge(multiply_node_1, multiply_node_2)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.HSWISH_ACTIVATION_V2)
def create_hswish_pattern_2():
    pattern = GraphPattern()
    input_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD, MULTIPLY, REDUCE_MEAN, SQUEEZE',
                                     GraphPattern.METATYPE_ATTR: [om.OVAddMetatype,
                                                                  om.OVMultiplyMetatype,
                                                                  om.OVReduceMeanMetatype,
                                                                  om.OVSqueezeMetatype]})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CLAMP',
                                     GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})
    multiply_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                          GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})

    pattern.add_edge(input_node, add_node)
    pattern.add_edge(add_node, clamp_node)
    pattern.add_edge(clamp_node, multiply_node_1)
    pattern.add_edge(input_node, multiply_node_2)
    pattern.add_edge(multiply_node_1, multiply_node_2)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.HSWISH_ACTIVATION_WITHOUT_DENOMINATOR)
def create_hswish_without_denominator():
    pattern = GraphPattern()
    linear_node = pattern.add_node(**LINEAR_OPERATIONS)
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD_BIAS',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                     GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CLAMP',
                                     GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})

    pattern.add_edge(linear_node, add_node_1)
    pattern.add_edge(add_node_1, add_node_2)
    pattern.add_edge(add_node_2, clamp_node)
    pattern.add_edge(add_node_1, multiply_node)
    pattern.add_edge(clamp_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.SOFTMAX)
def create_softmax():
    pattern = GraphPattern()
    exp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'EXP',
                                   GraphPattern.METATYPE_ATTR: om.OVExpMetatype})
    sum_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'REDUCE_SUM',
                                   GraphPattern.METATYPE_ATTR: om.OVSumMetatype})
    power_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'POWER',
                                     GraphPattern.METATYPE_ATTR: om.OVPowerMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})

    pattern.add_edge(exp_node, sum_node)
    pattern.add_edge(sum_node, power_node)
    pattern.add_edge(exp_node, multiply_node)
    pattern.add_edge(power_node, multiply_node)
    return pattern


# INPUT PROCESSING


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_SHIFT_SCALE)
def create_input_shift_scale():
    pattern = GraphPattern()
    model_input = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                      GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD, SUBTRACT',
                                   GraphPattern.METATYPE_ATTR: [om.OVAddMetatype, om.OVSubtractMetatype]})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})

    pattern.add_edge(model_input, add_node)
    pattern.add_edge(add_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_CONVERT_TRANSPOSE_PROCESSING)
def create_input_convert_transpose_processing():
    input_convert_transpose = create_input_convert_transpose()
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD, MULTIPLY, SUBTRACT',
                        GraphPattern.METATYPE_ATTR: [om.OVAddMetatype,
                                                     om.OVMultiplyMetatype,
                                                     om.OVSubtractMetatype]})

    input_convert_transpose.join_patterns(pattern)
    return input_convert_transpose


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_ADD)
def create_input_convert_transpose_reverse_add():
    input_convert_transpose = create_input_convert_transpose()
    pattern = GraphPattern()
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SPLIT',
                                     GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                      GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(split_node, concat_node)
    pattern.add_edge(concat_node, add_node)
    input_convert_transpose.join_patterns(pattern)
    return input_convert_transpose


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_SCALE_SHIFT)
def create_input_convert_transpose_reverse_scale_shift():
    pattern = GraphPattern()
    model_input = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
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
    scale_shift = create_scale_shift()

    pattern.add_edge(model_input, convert_node)
    pattern.add_edge(convert_node, transpose_node)
    pattern.add_edge(transpose_node, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT)
def create_input_convert_transpose_scale_shift():
    input_convert_transpose = create_input_convert_transpose()
    scale_shift = create_scale_shift()
    input_convert_transpose.join_patterns(scale_shift)
    return input_convert_transpose


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_PROCESSING)
def create_input_processing():
    pattern = GraphPattern()
    model_input = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                      GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    processing_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SUBTRACT, MULTIPLY, ADD',
                                          GraphPattern.METATYPE_ATTR: [om.OVSubtractMetatype,
                                                                       om.OVMultiplyMetatype,
                                                                       om.OVAddMetatype]})

    pattern.add_edge(model_input, processing_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_REVERSE_ADD)
def create_input_reverse_add():
    pattern = GraphPattern()
    model_input = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                      GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SPLIT',
                                     GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                      GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD',
                                   GraphPattern.METATYPE_ATTR: om.OVAddMetatype})

    pattern.add_edge(model_input, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.add_edge(concat_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_REVERSE_SCALE_SHIFT)
def create_input_reverse_scale_shift():
    pattern = GraphPattern()
    model_input = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                      GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    split_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'SPLIT',
                                     GraphPattern.METATYPE_ATTR: om.OVSplitMetatype})
    # TODO (KodiaqQ): Check the pattern on real case
    concat_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONCAT',
                                      GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})
    scale_shift = create_scale_shift()

    pattern.add_edge(model_input, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_SCALE_SHIFT)
def create_input_scale_shift():
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                        GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    scale_shift = create_scale_shift()

    pattern.join_patterns(scale_shift)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_TRANSPOSE_PROCESSING)
def create_input_transpose_processing():
    pattern = GraphPattern()
    model_input = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                      GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    processing_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'ADD, MULTIPLY, SUBTRACT',
                                          GraphPattern.METATYPE_ATTR: [om.OVAddMetatype,
                                                                       om.OVMultiplyMetatype,
                                                                       om.OVSubtractMetatype]})

    pattern.add_edge(model_input, transpose_node)
    pattern.add_edge(transpose_node, processing_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_TRANSPOSE_REVERSE_ADD)
def create_input_transpose_reverse_add():
    pattern = GraphPattern()
    model_input = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
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

    pattern.add_edge(model_input, transpose_node)
    pattern.add_edge(transpose_node, split_node)
    pattern.add_edge(split_node, concat_node)
    pattern.add_edge(concat_node, add_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.INPUT_TRANSPOSE_SCALE_SHIFT)
def create_input_transpose_scale_shift():
    pattern = GraphPattern()
    model_input = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                      GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    scale_shift = create_scale_shift()

    pattern.add_edge(model_input, transpose_node)
    pattern.join_patterns(scale_shift)
    return pattern


# COMBINATIONS

@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.ACTIVATIONS_BATCH_NORM)
def create_activations_batch_norm():
    activations = atomic_activations_operations()
    batch_norm = batch_normalization_operations()
    activations.join_patterns(batch_norm)
    return activations


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_ACTIVATIONS)
def create_arithmetic_activations():
    arithmetic = arithmetic_operations()
    activations = atomic_activations_operations()
    arithmetic.join_patterns(activations)
    return arithmetic


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.BATCH_NORM_ACTIVATIONS)
def create_batch_norm_activations():
    batch_norm = batch_normalization_operations()
    activations = atomic_activations_operations()
    batch_norm.join_patterns(activations)
    return batch_norm


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ACTIVATIONS)
def create_linear_activations():
    linear = linear_operations()
    activations = atomic_activations_operations()
    linear.join_patterns(activations)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ARITHMETIC)
def create_linear_arithmetic():
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    linear.join_patterns(arithmetic)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ARITHMETIC_ACTIVATIONS)
def create_linear_arithmetic_activations():
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    activations = atomic_activations_operations()

    linear.join_patterns(arithmetic)
    linear.join_patterns(activations)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.MVN_SCALE_SHIFT_ACTIVATIONS)
def create_mvn_scale_shift_activations():
    pattern = GraphPattern()
    pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MVN',
                        GraphPattern.METATYPE_ATTR: om.OVMVNMetatype})
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()

    pattern.join_patterns(scale_shift)
    pattern.join_patterns(activations)
    return pattern

# DEVICE PATTERNS


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.HSWISH_ACTIVATION_CLAMP_MULTIPLY)
def create_clamp_mult_const():
    pattern = GraphPattern()
    clamp_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CLAMP',
                                     GraphPattern.METATYPE_ATTR: om.OVClampMetatype})
    multiply_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MULTIPLY',
                                        GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype})

    pattern.add_edge(clamp_node, multiply_node)
    return pattern


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.SCALE_SHIFT_ACTIVATIONS)
def create_scale_shift_activations():
    scale_shift = create_scale_shift()
    activations = atomic_activations_operations()
    scale_shift.join_patterns(activations)
    return scale_shift


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_SCALE_SHIFT)
def create_linear_scale_shift():
    linear = linear_operations()
    scale_shift = create_scale_shift()
    linear.join_patterns(scale_shift)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_BIASED_SCALE_SHIFT)
def create_linear_biased_scale_shift():
    linear_biased = create_biased_op()
    scale_shift = create_scale_shift()
    linear_biased.join_patterns(scale_shift)
    return linear_biased


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ACTIVATION_SCALE_SHIFT)
def create_linear_activation_scale_shift():
    linear_activations = create_linear_activations()
    scale_shift = create_scale_shift()

    linear_activations.join_patterns(scale_shift)
    return linear_activations


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_BIASED_ACTIVATION_SCALE_SHIFT)
def create_linear_biased_activation_scale_shift():
    linear_biased = create_biased_op()
    activations = atomic_activations_operations()
    scale_shift = create_scale_shift()

    linear_biased.join_patterns(activations)
    linear_biased.join_patterns(scale_shift)
    return linear_biased


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ELEMENTWISE)
def create_linear_elementwise():
    linear = linear_operations()
    elementwise = elementwise_operations()
    linear.join_patterns(elementwise)
    return linear


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_BIASED_ELEMENTWISE)
def create_linear_biased_elementwise():
    linear_biased = create_biased_op()
    elementwise = elementwise_operations()
    linear_biased.join_patterns(elementwise)
    return linear_biased


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ACTIVATION_ELEMENTWISE)
def create_linear_activation_elementwise():
    linear_activations = create_linear_activations()
    elementwise = elementwise_operations()

    linear_activations.join_patterns(elementwise)
    return linear_activations


@OPENVINO_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_BIASED_ACTIVATION_ELEMENTWISE)
def create_linear_biased_activation_elementwise():
    linear_biased = create_biased_op()
    activations = atomic_activations_operations()
    elementwise = elementwise_operations()

    linear_biased.join_patterns(activations)
    linear_biased.join_patterns(elementwise)
    return linear_biased


def elementwise_operations():
    pattern = GraphPattern()
    pattern.add_node(**ELEMENTWISE_OPERATIONS)
    return pattern


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
    return pattern


def arithmetic_operations():
    pattern = GraphPattern()
    pattern.add_node(**ARITHMETIC_OPERATIONS)
    return pattern


def create_input_convert_transpose():
    pattern = GraphPattern()
    model_input = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'MODEL_INPUT',
                                      GraphPattern.METATYPE_ATTR: om.OVParameterMetatype})
    convert_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'CONVERT',
                                       GraphPattern.METATYPE_ATTR: om.OVConvertMetatype})
    transpose_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: 'TRANSPOSE',
                                         GraphPattern.METATYPE_ATTR: om.OVTransposeMetatype})
    pattern.add_edge(model_input, convert_node)
    pattern.add_edge(convert_node, transpose_node)
    return pattern
