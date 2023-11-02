# Copyright (c) 2023 Intel Corporation
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
from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.graph.metatypes.groups import MATMUL_METATYPES
from nncf.onnx.hardware.fused_patterns import atomic_activations_operations

ONNX_IGNORED_PATTERNS = Registry("IGNORED_PATTERNS")


def _add_softmax_matmul(pattern: GraphPattern) -> None:
    #       SOFTMAX  RESHAPE||TRANSPOSE||GATHER||SQUEEZE
    #           \              /
    #            \            /
    #             \          /
    #              \        /
    #               \      /
    #                MATMUL
    reshape_transpose_gather_squeeze = [
        om.ONNXReshapeMetatype,
        om.ONNXTransposeMetatype,
        om.ONNXGatherMetatype,
        om.ONNXSqueezeMetatype,
    ]
    softmax = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: om.ONNXSoftmaxMetatype}
    )
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: MATMUL_METATYPES})
    matmul_branch_nodes = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "RESHAPE||TRANSPOSE||GATHER||SQUEEZE",
            GraphPattern.METATYPE_ATTR: reshape_transpose_gather_squeeze,
        }
    )
    pattern.add_edge(softmax, matmul)
    pattern.add_edge(matmul_branch_nodes, matmul)


def _add_softmax_reshape_matmul(pattern: GraphPattern) -> None:
    #       SOFTMAX
    #           \
    #            \
    #             \
    #             RESHAPE   RESHAPE||TRANSPOSE||GATHER||SQUEEZE
    #                 \                 /
    #                  \               /
    #                   \             /
    #                    \           /
    #                     \         /
    #                      \       /
    #                        MATMUL
    reshape_transpose_gather_squeeze = [
        om.ONNXReshapeMetatype,
        om.ONNXTransposeMetatype,
        om.ONNXGatherMetatype,
        om.ONNXSqueezeMetatype,
    ]
    softmax = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: om.ONNXSoftmaxMetatype}
    )
    reshape = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "RESHAPE", GraphPattern.METATYPE_ATTR: om.ONNXReshapeMetatype}
    )
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: MATMUL_METATYPES})
    matmul_branch_nodes = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "RESHAPE||TRANSPOSE||GATHER||SQUEEZE",
            GraphPattern.METATYPE_ATTR: reshape_transpose_gather_squeeze,
        }
    )
    pattern.add_edge(softmax, reshape)
    pattern.add_edge(reshape, matmul)
    pattern.add_edge(matmul_branch_nodes, matmul)


@ONNX_IGNORED_PATTERNS.register(IgnoredPatternNames.MULTIHEAD_ATTENTION_OUTPUT)
def create_multihead_attention_output() -> GraphPattern:
    pattern = GraphPattern()
    _add_softmax_matmul(pattern)
    _add_softmax_reshape_matmul(pattern)
    return pattern


@ONNX_IGNORED_PATTERNS.register(IgnoredPatternNames.SE_BLOCK)
def create_se_block() -> GraphPattern:
    #       NON_PATTERN_NODE--------
    #    (PATTERN_NODE_TO_EXCLUDE) |
    #              |               |
    #         REDUCE_MEAN          |
    #              |               |
    #         CONVOLUTION          |
    #              |               |
    #          ACTIVATION          |
    #              |               |
    #         CONVOLUTION          |
    #              |               |
    #     SIGMOID||HARDSIGMOID     |
    #              |               |
    #              |               |
    #             MUL---------------
    #   (PATTERN_NODE_TO_EXCLUDE)
    pattern = GraphPattern()
    non_pattern_node = pattern.add_node(label="NON_PATTERN_NODE", type=GraphPattern.NON_PATTERN_NODE_TYPE)
    reduce_mean_node = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "REDUCE_MEAN",
            GraphPattern.METATYPE_ATTR: om.ONNXReduceMeanMetatype,
            GraphPattern.PATTERN_NODE_TO_EXCLUDE: True,
        }
    )
    conv_node_1 = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONVOLUTION", GraphPattern.METATYPE_ATTR: om.ONNXConvolutionMetatype}
    )

    pattern.add_edge(non_pattern_node, reduce_mean_node)
    pattern.add_edge(reduce_mean_node, conv_node_1)
    pattern.join_patterns(atomic_activations_operations())

    rest_pattern = GraphPattern()
    conv_node_2 = rest_pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "CONVOLUTION", GraphPattern.METATYPE_ATTR: om.ONNXConvolutionMetatype}
    )

    sigmoid_node = rest_pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "SIGMOID",
            GraphPattern.METATYPE_ATTR: [om.ONNXSigmoidMetatype, om.ONNXHardSigmoidMetatype],
        }
    )
    multiply_node = rest_pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "LAST_MULTIPLY",
            GraphPattern.METATYPE_ATTR: om.ONNXMulLayerMetatype,
            GraphPattern.PATTERN_NODE_TO_EXCLUDE: True,
        }
    )
    rest_pattern.add_edge(conv_node_2, sigmoid_node)
    rest_pattern.add_edge(sigmoid_node, multiply_node)
    pattern.join_patterns(rest_pattern)
    # Connect all NON_PATTERN_NODE with all MULTIPLY
    for component in pattern.get_weakly_connected_subgraphs():
        for node_id, attrs in component.nodes(data=True):
            if attrs[GraphPattern.LABEL_ATTR] == "NON_PATTERN_NODE":
                non_pattern_node = node_id
            if attrs[GraphPattern.LABEL_ATTR] == "LAST_MULTIPLY":
                multiply_node = node_id
        pattern.add_edge(non_pattern_node, multiply_node)
    return pattern
