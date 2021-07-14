"""
 Copyright (c) 2020 Intel Corporation
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

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.PATTERN_INPUT_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type='AddV2')
    relu_node = pattern.add_node(label='RELU', type='ReLU')
    mul_node = pattern.add_node(label='TF_OP_MUL', type='Mul')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, relu_node)
    pattern.add_edge(relu_node, mul_node)

    return pattern


def create_h_swish_act() -> GraphPattern:
    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.PATTERN_INPUT_NODE_TYPE)

    # TODO (vshampor): current approach with join_patterns is deficient since it does not allow to reliably
    #  connect nodes after a pattern has been joined. Along with the label and the type, the nodes created
    #  in the pattern must allow a "name" or "address" attribute, which must be a unique human readable
    #  string identifier of the node even if it has been joined multiple times, or perhaps each pattern
    #  after joining must return a list of output nodes so that these can be joined to later.
    #  Currently cannot specify h_swish in terms of h_sigmoid due to this.
    add_node = pattern.add_node(label='ADD', type='AddV2')
    relu_node = pattern.add_node(label='RELU', type='ReLU')
    mul_node = pattern.add_node(label='TF_OP_MUL', type='Mul')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, relu_node)
    pattern.add_edge(relu_node, mul_node)

    mul_2_node = pattern.add_node(label='MULTIPLY', type='Multiply')
    pattern.add_edge(input_pattern_node, mul_2_node)
    pattern.add_edge(mul_node, mul_2_node)

    return pattern
