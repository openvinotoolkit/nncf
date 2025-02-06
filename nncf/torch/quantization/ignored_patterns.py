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
from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.patterns.patterns import IgnoredPatternNames
from nncf.common.utils.registry import Registry
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.torch.graph.pattern_operations import LINEAR_OPERATIONS

PT_IGNORED_PATTERNS = Registry("IGNORED_PATTERNS")


def _add_softmax_matmul(
    pattern: GraphPattern,
    matmul_metatypes,
    reshape_squeeze_metatypes,
    gather_metatypes,
    transpose_metatypes,
    concat_metatypes,
) -> None:
    #       SOFTMAX  RESHAPE||TRANSPOSE||GATHER||SQUEEZE||CONCAT
    #           \              /
    #            \            /
    #             \          /
    #              \        /
    #               \      /
    #                MATMUL
    branch_matmul_nodes = reshape_squeeze_metatypes + gather_metatypes + transpose_metatypes + concat_metatypes
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: om.PTSoftmaxMetatype})
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: matmul_metatypes})
    matmul_branch_nodes = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "NON_PATTERN", GraphPattern.METATYPE_ATTR: branch_matmul_nodes}
    )
    pattern.add_edge(softmax, matmul)
    pattern.add_edge(matmul_branch_nodes, matmul)


def _add_softmax_reshape_matmul(
    pattern: GraphPattern,
    matmul_metatypes,
    reshape_squeeze_metatypes,
    gather_metatypes,
    transpose_metatypes,
    concat_metatypes,
) -> None:
    #       SOFTMAX
    #           \
    #            \
    #             \
    #    RESHAPE || TRANSPOSE   RESHAPE||TRANSPOSE||GATHER||SQUEEZE||CONCAT
    #                 \                 /
    #                  \               /
    #                   \             /
    #                    \           /
    #                     \         /
    #                      \       /
    #                        MATMUL
    branch_matmul_nodes = reshape_squeeze_metatypes + gather_metatypes + transpose_metatypes + concat_metatypes
    softmax = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SOFTMAX", GraphPattern.METATYPE_ATTR: om.PTSoftmaxMetatype})
    reshape = pattern.add_node(
        **{
            GraphPattern.LABEL_ATTR: "RESHAPE",
            GraphPattern.METATYPE_ATTR: reshape_squeeze_metatypes + transpose_metatypes,
        }
    )
    matmul = pattern.add_node(**{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: matmul_metatypes})
    matmul_branch_nodes = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "RESHAPE||TRANSPOSE||GATHER", GraphPattern.METATYPE_ATTR: branch_matmul_nodes}
    )
    pattern.add_edge(softmax, reshape)
    pattern.add_edge(reshape, matmul)
    pattern.add_edge(matmul_branch_nodes, matmul)
    return pattern


@PT_IGNORED_PATTERNS.register(IgnoredPatternNames.MULTIHEAD_ATTENTION_OUTPUT)
def create_multihead_attention_output() -> GraphPattern:
    matmul_metatypes = [om.PTLinearMetatype, om.PTAddmmMetatype, om.PTMatMulMetatype]
    reshape_squeeze_metatypes = [om.PTReshapeMetatype, om.PTSqueezeMetatype, om.PTSplitMetatype]
    gather_metatypes = [om.PTGatherMetatype]
    transpose_metatypes = [om.PTTransposeMetatype]
    concat_metatypes = [om.PTCatMetatype]

    pattern = GraphPattern()
    _add_softmax_matmul(
        pattern,
        matmul_metatypes=matmul_metatypes,
        reshape_squeeze_metatypes=reshape_squeeze_metatypes,
        gather_metatypes=gather_metatypes,
        transpose_metatypes=transpose_metatypes,
        concat_metatypes=concat_metatypes,
    )
    _add_softmax_reshape_matmul(
        pattern,
        matmul_metatypes=matmul_metatypes,
        reshape_squeeze_metatypes=reshape_squeeze_metatypes,
        gather_metatypes=gather_metatypes,
        transpose_metatypes=transpose_metatypes,
        concat_metatypes=concat_metatypes,
    )
    return pattern


# pylint:disable=too-many-statements
@PT_IGNORED_PATTERNS.register(IgnoredPatternNames.SE_BLOCK)
def create_se_block() -> GraphPattern:
    MEAN_OPERATIONS = {
        GraphPattern.LABEL_ATTR: "REDUCE_MEAN",
        GraphPattern.METATYPE_ATTR: [om.PTAvgPool2dMetatype, om.PTAvgPool3dMetatype, om.PTMeanMetatype],
        GraphPattern.PATTERN_NODE_TO_EXCLUDE: True,
    }
    SYGMOID_OPERATIONS = {
        GraphPattern.LABEL_ATTR: "SIGMOID",
        GraphPattern.METATYPE_ATTR: [om.PTSigmoidMetatype, om.PTHardSigmoidMetatype],
    }
    MUL_OPERATION = {
        GraphPattern.LABEL_ATTR: "MUL",
        GraphPattern.METATYPE_ATTR: om.PTMulMetatype,
        GraphPattern.PATTERN_NODE_TO_EXCLUDE: True,
    }

    def get_se_block_pattern() -> GraphPattern:
        pattern = GraphPattern()
        any_node = pattern.add_node(label="NON_PATTERN_NODE", type=GraphPattern.NON_PATTERN_NODE_TYPE)
        reduce_mean_node = pattern.add_node(**MEAN_OPERATIONS)
        linear_node_1 = pattern.add_node(**LINEAR_OPERATIONS)
        activation_node_1 = pattern.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
        linear_node_2 = pattern.add_node(**LINEAR_OPERATIONS)
        activation_node_2 = pattern.add_node(**SYGMOID_OPERATIONS)
        multiply_node = pattern.add_node(**MUL_OPERATION)

        pattern.add_edge(any_node, reduce_mean_node)
        pattern.add_edge(reduce_mean_node, linear_node_1)
        pattern.add_edge(linear_node_1, activation_node_1)
        pattern.add_edge(activation_node_1, linear_node_2)
        pattern.add_edge(linear_node_2, activation_node_2)
        pattern.add_edge(activation_node_2, multiply_node)
        pattern.add_edge(any_node, multiply_node)
        return pattern

    def get_se_block_with_bias_pattern() -> GraphPattern:
        pattern = GraphPattern()
        any_node = pattern.add_node(label="NON_PATTERN_NODE", type=GraphPattern.NON_PATTERN_NODE_TYPE)
        reduce_mean_node = pattern.add_node(**MEAN_OPERATIONS)
        linear_node_1 = pattern.add_node(**LINEAR_OPERATIONS)
        add_node_1 = pattern.add_node(label="ADD_BIAS", type=[om.PTAddMetatype, om.PTSubMetatype])
        activation_node_1 = pattern.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
        linear_node_2 = pattern.add_node(**LINEAR_OPERATIONS)
        add_node_2 = pattern.add_node(label="ADD_BIAS", type=[om.PTAddMetatype, om.PTSubMetatype])
        activation_node_2 = pattern.add_node(**SYGMOID_OPERATIONS)
        multiply_node = pattern.add_node(**MUL_OPERATION)

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

    RESHAPE_NODES = {
        GraphPattern.LABEL_ATTR: "RESHAPE",
        GraphPattern.METATYPE_ATTR: om.PTReshapeMetatype,
    }

    def get_se_block_with_reshape() -> GraphPattern:
        pattern = GraphPattern()
        any_node = pattern.add_node(label="NON_PATTERN_NODE", type=GraphPattern.NON_PATTERN_NODE_TYPE)
        reduce_mean_node = pattern.add_node(**MEAN_OPERATIONS)
        reshape_node_1 = pattern.add_node(**RESHAPE_NODES)
        linear_node_1 = pattern.add_node(**LINEAR_OPERATIONS)
        activation_node_1 = pattern.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
        linear_node_2 = pattern.add_node(**LINEAR_OPERATIONS)
        activation_node_2 = pattern.add_node(**SYGMOID_OPERATIONS)
        reshape_node_2 = pattern.add_node(**RESHAPE_NODES)
        multiply_node = pattern.add_node(**MUL_OPERATION)

        pattern.add_edge(any_node, reduce_mean_node)
        pattern.add_edge(reduce_mean_node, reshape_node_1)
        pattern.add_edge(reshape_node_1, linear_node_1)
        pattern.add_edge(linear_node_1, activation_node_1)
        pattern.add_edge(activation_node_1, linear_node_2)
        pattern.add_edge(linear_node_2, activation_node_2)
        pattern.add_edge(activation_node_2, reshape_node_2)
        pattern.add_edge(reshape_node_2, multiply_node)
        pattern.add_edge(any_node, multiply_node)
        return pattern

    def get_se_block_with_bias_and_reshape() -> GraphPattern:
        pattern = GraphPattern()
        any_node = pattern.add_node(label="NON_PATTERN_NODE", type=GraphPattern.NON_PATTERN_NODE_TYPE)
        reduce_mean_node = pattern.add_node(**MEAN_OPERATIONS)
        reshape_node_1 = pattern.add_node(**RESHAPE_NODES)
        linear_node_1 = pattern.add_node(**LINEAR_OPERATIONS)
        add_node_1 = pattern.add_node(label="ADD_BIAS", type=[om.PTAddMetatype, om.PTSubMetatype])
        activation_node_1 = pattern.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
        linear_node_2 = pattern.add_node(**LINEAR_OPERATIONS)
        add_node_2 = pattern.add_node(label="ADD_BIAS", type=[om.PTAddMetatype, om.PTSubMetatype])
        activation_node_2 = pattern.add_node(**SYGMOID_OPERATIONS)
        reshape_node_2 = pattern.add_node(**RESHAPE_NODES)
        multiply_node = pattern.add_node(**MUL_OPERATION)

        pattern.add_edge(any_node, reduce_mean_node)
        pattern.add_edge(reduce_mean_node, reshape_node_1)
        pattern.add_edge(reshape_node_1, linear_node_1)
        pattern.add_edge(linear_node_1, add_node_1)
        pattern.add_edge(add_node_1, activation_node_1)
        pattern.add_edge(activation_node_1, linear_node_2)
        pattern.add_edge(linear_node_2, add_node_2)
        pattern.add_edge(add_node_2, activation_node_2)
        pattern.add_edge(activation_node_2, reshape_node_2)
        pattern.add_edge(reshape_node_2, multiply_node)
        pattern.add_edge(any_node, multiply_node)
        return pattern

    main_pattern = GraphPattern()
    main_pattern.add_pattern_alternative(get_se_block_pattern())
    main_pattern.add_pattern_alternative(get_se_block_with_bias_pattern())
    main_pattern.add_pattern_alternative(get_se_block_with_reshape())
    main_pattern.add_pattern_alternative(get_se_block_with_bias_and_reshape())
    return main_pattern


@PT_IGNORED_PATTERNS.register(IgnoredPatternNames.ROPE)
def create_rope() -> GraphPattern:
    pattern = GraphPattern()
    matmul_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "MATMUL", GraphPattern.METATYPE_ATTR: om.PTMatMulMetatype}
    )
    transpose_node = pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "TRANSPOSE", GraphPattern.METATYPE_ATTR: om.PTTransposeMetatype}
    )
    concat_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "CONCAT", GraphPattern.METATYPE_ATTR: om.PTCatMetatype})
    cos_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "COS", GraphPattern.METATYPE_ATTR: om.PTCosMetatype})
    sin_node = pattern.add_node(**{GraphPattern.LABEL_ATTR: "SIN", GraphPattern.METATYPE_ATTR: om.PTSinMetatype})

    pattern.add_edge(matmul_node, transpose_node)
    pattern.add_edge(transpose_node, concat_node)
    pattern.add_edge(concat_node, cos_node)
    pattern.add_edge(concat_node, sin_node)
    return pattern
