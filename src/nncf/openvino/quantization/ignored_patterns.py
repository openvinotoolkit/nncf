# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.patterns.patterns import IgnoredPatternNames
from nncf.common.utils.registry import Registry
from nncf.openvino.graph.metatypes import openvino_metatypes as om
from nncf.openvino.graph.metatypes.groups import LINEAR_OPERATIONS
from nncf.quantization.ignored_patterns import create_rope_pattern

OPENVINO_IGNORED_PATTERNS = Registry("IGNORED_PATTERNS")


def _add_softmax_matmul(pattern: GraphPattern, branch_matmul_nodes: list[om.OperatorMetatype]) -> None:
    #       SOFTMAX  READVALUE||RESHAPE||TRANSPOSE||GATHER||SQUEEZE||CONCAT
    #           \              /
    #            \            /
    #             \          /
    #              \        /
    #               \      /
    #                MATMUL
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})
    matmul_branch_nodes = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "READVALUE||RESHAPE||TRANSPOSE||GATHER||SQUEEZE||CONCAT",
            GraphPattern.METATYPE_ATTR: branch_matmul_nodes,
        }
    )
    pattern.add_edge(softmax, matmul)
    pattern.add_edge(matmul_branch_nodes, matmul)


def _add_softmax_reshape_matmul(pattern: GraphPattern, branch_matmul_nodes: list[om.OperatorMetatype]) -> None:
    #       SOFTMAX
    #           \
    #            \
    #             \
    #             RESHAPE   READVALUE||RESHAPE||TRANSPOSE||GATHER||SQUEEZE||CONCAT
    #                 \                 /
    #                  \               /
    #                   \             /
    #                    \           /
    #                     \         /
    #                      \       /
    #                        MATMUL
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: om.OVSoftmaxMetatype})
    reshape = pattern.add_node(**{GraphPattern.LABEL_ATTR: "RESHAPE", GraphPattern.METATYPE_ATTR: om.OVReshapeMetatype})
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype})
    matmul_branch_nodes = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "READVALUE||RESHAPE||TRANSPOSE||GATHER||SQUEEZE||CONCAT",
            GraphPattern.METATYPE_ATTR: branch_matmul_nodes,
        }
    )
    pattern.add_edge(softmax, reshape)
    pattern.add_edge(reshape, matmul)
    pattern.add_edge(matmul_branch_nodes, matmul)


@OPENVINO_IGNORED_PATTERNS.register(IgnoredPatternNames.MULTIHEAD_ATTENTION_OUTPUT)
def create_multihead_attention_output() -> GraphPattern:
    pattern = GraphPattern()
    branch_matmul_nodes = (
        om.OVReadValueMetatype,
        om.OVReshapeMetatype,
        om.OVTransposeMetatype,
        om.OVGatherMetatype,
        om.OVSqueezeMetatype,
        om.OVConcatMetatype,
    )

    _add_softmax_matmul(pattern, list(branch_matmul_nodes))
    _add_softmax_reshape_matmul(pattern, list(branch_matmul_nodes))
    return pattern


@OPENVINO_IGNORED_PATTERNS.register(IgnoredPatternNames.FC_BN_HSWISH_ACTIVATION)
def create_fc_bn_hswish() -> GraphPattern:
    pattern = GraphPattern()
    unsqueeze_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "UNSQUEEZE", GraphPattern.METATYPE_ATTR: om.OVUnsqueezeMetatype}
    )
    multiply_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )
    add_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    squeeze_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "SQUEEZE", GraphPattern.METATYPE_ATTR: om.OVSqueezeMetatype}
    )

    pattern.add_edge(unsqueeze_node, multiply_node)
    pattern.add_edge(multiply_node, add_node)
    pattern.add_edge(add_node, squeeze_node)
    return pattern


@OPENVINO_IGNORED_PATTERNS.register(IgnoredPatternNames.EQUAL_LOGICALNOT)
def create_equal_logicalnot() -> GraphPattern:
    pattern = GraphPattern()
    equal_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "EQUAL", GraphPattern.METATYPE_ATTR: om.OVEqualMetatype})
    logical_not_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "LOGICAL_NOT", GraphPattern.METATYPE_ATTR: om.OVLogicalNotMetatype}
    )

    pattern.add_edge(equal_node, logical_not_node)
    return pattern


@OPENVINO_IGNORED_PATTERNS.register(IgnoredPatternNames.SE_BLOCK)
def create_se_block() -> GraphPattern:
    pattern = GraphPattern()
    any_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "ANY", GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE}
    )
    reduce_mean_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "REDUCE_MEAN",
            GraphPattern.METATYPE_ATTR: om.OVReduceMeanMetatype,
            GraphPattern.PATTERN_NODE_TO_EXCLUDE: True,
        }
    )
    linear_node_1 = pattern.add_node(
        **{GraphPattern.METATYPE_ATTR: LINEAR_OPERATIONS, GraphPattern.LABEL_ATTR: "LINEAR"}
    )
    add_node_1 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD_BIAS", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node_1 = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "RELU, PRELU, SWISH",
            GraphPattern.METATYPE_ATTR: [om.OVReluMetatype, om.OVPReluMetatype, om.OVSwishMetatype],
        }
    )
    linear_node_2 = pattern.add_node(
        **{GraphPattern.METATYPE_ATTR: LINEAR_OPERATIONS, GraphPattern.LABEL_ATTR: "LINEAR"}
    )
    add_node_2 = pattern.add_node(**{GraphPattern.LABEL_ATTR: "ADD_BIAS", GraphPattern.METATYPE_ATTR: om.OVAddMetatype})
    activation_node_2 = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "SIGMOID",
            GraphPattern.METATYPE_ATTR: [om.OVSigmoidMetatype, om.OVHSigmoidMetatype],
        }
    )
    multiply_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "MULTIPLY",
            GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype,
            GraphPattern.PATTERN_NODE_TO_EXCLUDE: True,
        }
    )

    pattern.add_edge(any_node, reduce_mean_node)
    pattern.add_edge(reduce_mean_node, linear_node_1)
    pattern.add_edge(linear_node_1, add_node_1)
    pattern.add_edge(add_node_1, activation_node_1)
    pattern.add_edge(activation_node_1, linear_node_2)
    pattern.add_edge(linear_node_2, add_node_2)
    pattern.add_edge(add_node_2, activation_node_2)
    pattern.add_edge(activation_node_2, multiply_node)
    pattern.add_edge(any_node, multiply_node)
    return pattern


@OPENVINO_IGNORED_PATTERNS.register(IgnoredPatternNames.ROPE)
def create_rope() -> GraphPattern:
    return create_rope_pattern(
        mm_metatype=om.OVMatMulMetatype,
        transpose_metatype=om.OVTransposeMetatype,
        concat_metatype=om.OVConcatMetatype,
        cos_metatype=om.OVCosMetatype,
        sin_metatype=om.OVSinMetatype,
    )


@OPENVINO_IGNORED_PATTERNS.register(IgnoredPatternNames.SAM_PE)
def create_sam_pe() -> GraphPattern:
    """
    Positional Embedding from Segment Anything Model (SAM).
    """
    pattern = GraphPattern()

    matmul_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: om.OVMatMulMetatype}
    )
    mul_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MULTIPLY", GraphPattern.METATYPE_ATTR: om.OVMultiplyMetatype}
    )
    cos_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "COS", GraphPattern.METATYPE_ATTR: om.OVCosMetatype})
    sin_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SIN", GraphPattern.METATYPE_ATTR: om.OVSinMetatype})
    concat = pattern.add_node(**{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: om.OVConcatMetatype})

    pattern.add_edge(matmul_node, mul_node)
    pattern.add_edge(mul_node, cos_node)
    pattern.add_edge(mul_node, sin_node)
    pattern.add_edge(cos_node, concat)
    pattern.add_edge(sin_node, concat)

    return pattern
