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

    main_pattern.add_pattern_alternative(pattern)

    pattern = GraphPattern()
    linear_ops_node = pattern.add_node(LINEAR_OPERATIONS)
    sigmoid_node = pattern.add_node(['sigmoid'])
    mul_node = pattern.add_node(['__mul__'])

    pattern.add_edge(linear_ops_node, sigmoid_node)
    pattern.add_edge(linear_ops_node, mul_node)

    main_pattern.add_pattern_alternative(pattern)

    return main_pattern


def create_linear_h_swish_act() -> GraphPattern:
    main_pattern = GraphPattern()

    pattern = GraphPattern()
    linear_ops_node = pattern.add_node(LINEAR_OPERATIONS)
    bn_node = pattern.add_node(BATCH_NORMALIZATION_OPERATIONS)
    __add__node = pattern.add_node(['__add__'])
    hardtanh_node = pattern.add_node(['hardtanh'])
    __truediv__node = pattern.add_node(['__truediv__'])
    mul_node = pattern.add_node(['__mul__'])

    pattern.add_edge(linear_ops_node, bn_node)
    pattern.add_edge(bn_node, __add__node)
    pattern.add_edge(bn_node, mul_node)
    pattern.add_edge(__add__node, hardtanh_node)
    pattern.add_edge(hardtanh_node, __truediv__node)
    pattern.add_edge(__truediv__node, mul_node)

    main_pattern.add_pattern_alternative(pattern)

    pattern = GraphPattern()
    linear_ops_node = pattern.add_node(LINEAR_OPERATIONS)
    __add__node = pattern.add_node(['__add__'])
    hardtanh_node = pattern.add_node(['hardtanh'])
    __truediv__node = pattern.add_node(['__truediv__'])
    mul_node = pattern.add_node(['__mul__'])

    pattern.add_edge(linear_ops_node, __add__node)
    pattern.add_edge(linear_ops_node, mul_node)
    pattern.add_edge(__add__node, hardtanh_node)
    pattern.add_edge(hardtanh_node, __truediv__node)
    pattern.add_edge(__truediv__node, mul_node)

    main_pattern.add_pattern_alternative(pattern)

    return main_pattern


def create_linear_h_sigmoid_act() -> GraphPattern:
    pattern = GraphPattern()
    linear_ops_node = pattern.add_node(LINEAR_OPERATIONS)
    __add__node = pattern.add_node(['__add__'])
    hardtanh_node = pattern.add_node(['hardtanh'])
    __truediv__node = pattern.add_node(['__truediv__'])

    pattern.add_edge(linear_ops_node, __add__node)
    pattern.add_edge(__add__node, hardtanh_node)
    pattern.add_edge(hardtanh_node, __truediv__node)

    return pattern


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

        LINEAR_OPS_H_SWISH_ACTIVATION = create_linear_h_swish_act()

        LINEAR_OPS_H_SIGMOID_ACTIVATION = create_linear_h_sigmoid_act()

        FULL_PATTERN_GRAPH = LINEAR_OPS + ANY_BN_ACT_COMBO | ANY_BN_ACT_COMBO | \
                             ARITHMETIC + ANY_BN_ACT_COMBO | LINEAR_OPS_SWISH_ACTIVATION | \
                             LINEAR_OPS_H_SWISH_ACTIVATION | LINEAR_OPS_H_SIGMOID_ACTIVATION

        return FULL_PATTERN_GRAPH


def get_full_pattern_graph():
    return PATTERN_GRAPH.get("graph_pattern_factory").get_graph()
