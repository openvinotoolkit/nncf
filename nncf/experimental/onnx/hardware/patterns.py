"""
 Copyright (c) 2022 Intel Corporation
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

from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.patterns import GraphPattern


def create_swish_activation() -> GraphPattern:
    pattern = GraphPattern()

    input_pattern_node_1 = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    sigmoid_node_1 = pattern.add_node(label='SIGMOID', type='Sigmoid')
    mul_node_1 = pattern.add_node(label='MUL', type='Mul')

    pattern.add_edge(input_pattern_node_1, sigmoid_node_1)
    pattern.add_edge(input_pattern_node_1, mul_node_1)
    pattern.add_edge(sigmoid_node_1, mul_node_1)

    input_pattern_node_2 = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    sigmoid_node_2 = pattern.add_node(label='HARDSIGMOID', type='HardSigmoid')
    mul_node_2 = pattern.add_node(label='MUL', type='Mul')

    pattern.add_edge(input_pattern_node_2, sigmoid_node_2)
    pattern.add_edge(input_pattern_node_2, mul_node_2)
    pattern.add_edge(sigmoid_node_2, mul_node_2)

    return pattern


def create_input_preprocessing_pattern() -> GraphPattern:
    pattern = GraphPattern()

    model_input_node_1 = pattern.add_node(label='MODEL_INPUT', type=NNCFGraphNodeType.INPUT_NODE)
    add_node_1 = pattern.add_node(label='ADD', type='Add')
    mul_node_1 = pattern.add_node(label='MUL', type='Mul')

    pattern.add_edge(model_input_node_1, add_node_1)
    pattern.add_edge(add_node_1, mul_node_1)

    model_input_node_2 = pattern.add_node(label='MODEL_INPUT', type=NNCFGraphNodeType.INPUT_NODE)
    mul_node_2 = pattern.add_node(label='MUL', type='Mul')
    add_node_2 = pattern.add_node(label='ADD', type='Add')

    pattern.add_edge(model_input_node_2, mul_node_2)
    pattern.add_edge(mul_node_2, add_node_2)

    model_input_node_3 = pattern.add_node(label='MODEL_INPUT', type=NNCFGraphNodeType.INPUT_NODE)
    add_node_3 = pattern.add_node(label='ADD', type='Add')

    pattern.add_edge(model_input_node_3, add_node_3)

    model_input_node_4 = pattern.add_node(label='MODEL_INPUT', type=NNCFGraphNodeType.INPUT_NODE)
    mul_node_4 = pattern.add_node(label='MUL', type='Mul')

    pattern.add_edge(model_input_node_4, mul_node_4)

    return pattern
