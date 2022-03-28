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

from nncf.common.graph.patterns import GraphPattern


def create_h_sigmoid_act() -> GraphPattern:
    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.NON_PATTERN_NODE_TYPE)
    sigmoid_node = pattern.add_node(label='SIGMOID', type='Sigmoid')
    mul_node = pattern.add_node(label='MUL', type='Mul')

    pattern.add_edge(input_pattern_node, sigmoid_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(sigmoid_node, mul_node)
    return pattern
