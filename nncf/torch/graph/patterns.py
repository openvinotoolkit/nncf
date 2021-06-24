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


def create_swish_act() -> GraphPattern:
    pattern = GraphPattern('SWISH')
    input_pattern_node = pattern.add_node('*INPUT_NODE*', GraphPattern.INPUT_NODE_TYPE)
    sigmoid_node = pattern.add_node('SIGMOID', 'sigmoid')
    mul_node = pattern.add_node('MUL', '__mul__')

    pattern.add_edge(input_pattern_node, sigmoid_node)
    pattern.add_edge(sigmoid_node, mul_node)
    pattern.add_edge(input_pattern_node, mul_node)
    return pattern


def create_h_swish_act() -> GraphPattern:
    pattern = GraphPattern('H_SWISH')

    input_pattern_node = pattern.add_node('*INPUT_NODE*', GraphPattern.INPUT_NODE_TYPE)
    add_node = pattern.add_node('ADD', '__add__')
    hardtanh_node = pattern.add_node('HARDTANH', 'hardtanh')
    truediv_node = pattern.add_node('DIV', '__truediv__')
    mul_node = pattern.add_node('MUL', '__mul__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, truediv_node)
    pattern.add_edge(truediv_node, mul_node)
    return pattern


def create_h_sigmoid_act() -> GraphPattern:
    pattern = GraphPattern('H_SIGMOID')
    input_pattern_node = pattern.add_node('*INPUT_NODE*', GraphPattern.INPUT_NODE_TYPE)
    add_node = pattern.add_node('ADD', '__add__')
    hardtanh_node = pattern.add_node('HARTANH', 'hardtanh')
    truediv_node = pattern.add_node('DIV', '__truediv__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, truediv_node)
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
        LINEAR_OPS = GraphPattern('LINEAR', LINEAR_OPERATIONS)

        BN = GraphPattern('BATCH_NORM', BATCH_NORMALIZATION_OPERATIONS)

        ATOMIC_ACTIVATIONS = GraphPattern('ACTIVATIONS', ACTIVATIONS_OPERATIONS)

        SWISH = create_swish_act()

        H_SIGMOID = create_h_sigmoid_act()

        H_SWISH = create_h_swish_act()

        ACTIVATIONS = ATOMIC_ACTIVATIONS | SWISH | H_SWISH | H_SIGMOID

        ARITHMETIC = GraphPattern('ARITHMETIC', ARITHMETIC_OPERATIONS)

        ANY_BN_ACT_COMBO = BN + ACTIVATIONS | ACTIVATIONS + BN | BN | ACTIVATIONS

        FULL_PATTERN_GRAPH = LINEAR_OPS + ANY_BN_ACT_COMBO | BN + ACTIVATIONS | ACTIVATIONS + BN | \
                             ARITHMETIC + ANY_BN_ACT_COMBO

        return FULL_PATTERN_GRAPH


def get_full_pattern_graph():
    return PATTERN_GRAPH.get("graph_pattern_factory").get_graph()
