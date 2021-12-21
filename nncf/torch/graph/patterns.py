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


def create_swish_act() -> GraphPattern:
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    sigmoid_node = pattern.add_node(label='SIGMOID', type='sigmoid')
    mul_node = pattern.add_node(label='MUL', type='__mul__')

    pattern.add_edge(input_pattern_node, sigmoid_node)
    pattern.add_edge(sigmoid_node, mul_node)
    pattern.add_edge(input_pattern_node, mul_node)
    return pattern


def create_h_swish_act() -> GraphPattern:
    main_pattern = GraphPattern()

    # Mul -> Div version
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type='__add__')
    hardtanh_node = pattern.add_node(label='HARDTANH', type='hardtanh')
    truediv_node = pattern.add_node(label='DIV', type='__truediv__')
    mul_node = pattern.add_node(label='MUL', type='__mul__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, truediv_node)
    pattern.add_edge(truediv_node, mul_node)
    main_pattern.add_pattern_alternative(pattern)

    # Div -> Mul version
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type='__add__')
    hardtanh_node = pattern.add_node(label='HARDTANH', type='hardtanh')
    mul_node = pattern.add_node(label='MUL', type='__mul__')
    truediv_node = pattern.add_node(label='DIV', type='__truediv__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, mul_node)
    pattern.add_edge(mul_node, truediv_node)
    main_pattern.add_pattern_alternative(pattern)

    # ReLU6 version - Mul -> Div
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type='__add__')
    relu6_node = pattern.add_node(label='RELU6', type='relu6')
    mul_node = pattern.add_node(label='MUL', type='__mul__')
    truediv_node = pattern.add_node(label='DIV', type='__truediv__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(add_node, relu6_node)
    pattern.add_edge(hardtanh_node, mul_node)
    pattern.add_edge(mul_node, truediv_node)
    main_pattern.add_pattern_alternative(pattern)

    # ReLU6 version - Div -> Mul
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type='__add__')
    relu6_node = pattern.add_node(label='RELU6', type='relu6')
    truediv_node = pattern.add_node(label='DIV', type='__truediv__')
    mul_node = pattern.add_node(label='MUL', type='__mul__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(add_node, relu6_node)
    pattern.add_edge(relu6_node, truediv_node)
    pattern.add_edge(truediv_node, mul_node)

    main_pattern.add_pattern_alternative(pattern)

    return main_pattern


def create_h_sigmoid_act() -> GraphPattern:
    main_pattern = GraphPattern()

    # ReLU version:
    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type='__add__')
    hardtanh_node = pattern.add_node(label='HARDTANH', type='hardtanh')
    truediv_node = pattern.add_node(label='DIV', type='__truediv__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, truediv_node)

    main_pattern.add_pattern_alternative(pattern)

    # ReLU6 version
    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type='__add__')
    relu6_node = pattern.add_node(label='RELU6', type='relu6')
    truediv_node = pattern.add_node(label='DIV', type='__truediv__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, relu6_node)
    pattern.add_edge(hardtanh_node, truediv_node)

    main_pattern.add_pattern_alternative(pattern)

    return main_pattern


def create_l2_norm() -> GraphPattern:
    pattern = GraphPattern()

    outside_pattern_node = pattern.add_node(label='*OUTSIDE_PATTERN_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    pow_node = pattern.add_node(label='POW', type='pow')
    sum_node = pattern.add_node(label='SUM', type='sum')
    sqrt_node = pattern.add_node(label='SQRT', type='sqrt')
    add_node = pattern.add_node(label='ADD', type='__add__')
    div_node = pattern.add_node(label='DIV', type='div')
    mul_node = pattern.add_node(label='MUL', type='__rmul__')

    pattern.add_edge(outside_pattern_node, pow_node)
    pattern.add_edge(pow_node, sum_node)
    pattern.add_edge(sum_node, sqrt_node)
    pattern.add_edge(sqrt_node, add_node)
    pattern.add_edge(add_node, div_node)
    pattern.add_edge(div_node, mul_node)
    pattern.add_edge(outside_pattern_node, div_node)
    return pattern
