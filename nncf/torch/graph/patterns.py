"""
 Copyright (c) 2019-2020 Intel Corporation
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
from nncf.common.utils.registry import Registry

from nncf.torch.graph.pattern_operations import LINEAR_OPERATIONS
from nncf.torch.graph.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import ACTIVATIONS_OPERATIONS
from nncf.torch.graph.pattern_operations import ARITHMETIC_OPERATIONS

PATTERN_GRAPH = Registry('pattern_graph')


def create_linear_swish_act() -> GraphPattern:
    main_pattern = GraphPattern()

    pattern = GraphPattern()
    linear_ops_node = pattern.add_node(LINEAR_OPERATIONS)
    bn_node = pattern.add_node(BATCH_NORMALIZATION_OPERATIONS)
    sigmoid_node = pattern.add_node(['sigmoid'])
    mul_node = pattern.add_node(['__mul__'])

    pattern.add_edge(linear_ops_node, bn_node)
    pattern.add_edge(bn_node, sigmoid_node)
    pattern.add_edge(linear_ops_node, mul_node)

    GraphPattern.add_subgraph_to_graph(main_pattern.graph, pattern.graph)

    pattern = GraphPattern()
    linear_ops_node = pattern.add_node(LINEAR_OPERATIONS)
    sigmoid_node = pattern.add_node(['sigmoid'])
    mul_node = pattern.add_node(['__mul__'])

    pattern.add_edge(linear_ops_node, sigmoid_node)
    pattern.add_edge(linear_ops_node, mul_node)

    GraphPattern.add_subgraph_to_graph(main_pattern.graph, pattern.graph)

    return main_pattern


@PATTERN_GRAPH.register('graph_pattern_factory')
class GraphPatternFactory:
    FULL_PATTERN_GRAPH = None

    @staticmethod
    def get_graph():
        if GraphPatternFactory.FULL_PATTERN_GRAPH is None:
            GraphPatternFactory.FULL_PATTERN_GRAPH = GraphPatternFactory._generate_full_pattern_graph()
        return GraphPatternFactory.FULL_PATTERN_GRAPH

    @staticmethod
    def _generate_full_pattern_graph():
        LINEAR_OPS = GraphPattern(LINEAR_OPERATIONS)

        BN = GraphPattern(BATCH_NORMALIZATION_OPERATIONS)

        ACTIVATIONS = GraphPattern(ACTIVATIONS_OPERATIONS)

        ARITHMETIC = GraphPattern(ARITHMETIC_OPERATIONS)

        ANY_BN_ACT_COMBO = BN + ACTIVATIONS | ACTIVATIONS + BN | BN | ACTIVATIONS

        LINEAR_OPS_SWISH_ACTIVATION = create_linear_swish_act()

        FULL_PATTERN_GRAPH = LINEAR_OPS + ANY_BN_ACT_COMBO | ANY_BN_ACT_COMBO | \
                             ARITHMETIC + ANY_BN_ACT_COMBO | LINEAR_OPS_SWISH_ACTIVATION

        return FULL_PATTERN_GRAPH


def get_full_pattern_graph():
    return PATTERN_GRAPH.get("graph_pattern_factory").get_graph()

# def get_full_pattern_graph():
#     p1 = create_linear_any_bn_act_combo()
#     p2 = create_any_bn_act_combo()
#     p3 = create_arithmetic_any_bn_act_combo()
#     p4 = create_linear_swish_act()
#     return p1 | p2 | p3 | p4
#
# def create_linear_any_bn_act_combo() -> GraphPattern:
#     main_pattern = GraphPattern('linear_combination_batchnorm_activation')
#     for combo in [(BN_type, ACTIVATIONS_type), (ACTIVATIONS_type, BN_type),
#                   (BN_type), (ACTIVATIONS_type)]:
#
#         pattern = GraphPattern('test')
#         linear_ops_node = pattern.add_node(LINEAR_OPS_type)
#         bn_node = pattern.add_node(BN_type)
#         activation_node = pattern.add_node(ACTIVATIONS_type)
#
#         name_mapping = {BN_type: bn_node, ACTIVATIONS_type: activation_node}
#
#         pattern.add_edge(linear_ops_node, name_mapping[combo[0]])
#         if len(combo) > 1:
#             pattern.add_edge(name_mapping[combo[0]], name_mapping[combo[1]])
#
#         GraphPattern.add_subgraph_to_graph(main_pattern, pattern)
#
#     return main_pattern
#
#
# def create_any_bn_act_combo() -> GraphPattern:
#     main_pattern = GraphPattern('combination_batchnorm_activation')
#     for combo in [(BN_type, ACTIVATIONS_type), (ACTIVATIONS_type, BN_type),
#                   (BN_type), (ACTIVATIONS_type)]:
#
#         pattern = GraphPattern('test')
#         bn_node = pattern.add_node(BN_type)
#         activation_node = pattern.add_node(ACTIVATIONS_type)
#
#         name_mapping = {BN_type: bn_node, ACTIVATIONS_type: activation_node}
#
#         if len(combo) > 1:
#             pattern.add_edge(name_mapping[combo[0]], name_mapping[combo[1]])
#
#         GraphPattern.add_subgraph_to_graph(main_pattern, pattern)
#
#     return main_pattern
#
#
# def create_arithmetic_any_bn_act_combo() -> GraphPattern:
#     main_pattern = GraphPattern('arithmetic_combination_batchnorm_activation')
#     for combo in [(BN_type, ACTIVATIONS_type), (ACTIVATIONS_type, BN_type),
#                   (BN_type), (ACTIVATIONS_type)]:
#
#         pattern = GraphPattern('test')
#         arithmetic_node = pattern.add_node(ARITHMETIC_type)
#         bn_node = pattern.add_node(BN_type)
#         activation_node = pattern.add_node(ACTIVATIONS_type)
#
#         name_mapping = {BN_type: bn_node, ACTIVATIONS_type: activation_node}
#
#         pattern.add_edge(arithmetic_node, name_mapping[combo[0]])
#         if len(combo) > 1:
#             pattern.add_edge(name_mapping[combo[0]], name_mapping[combo[1]])
#
#         GraphPattern.add_subgraph_to_graph(main_pattern, pattern)
#     return pattern
#
#
