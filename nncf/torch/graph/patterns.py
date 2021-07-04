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
from nncf.common.graph.patterns import HWFusedPatterns

from nncf.torch.graph.pattern_operations import ACTIVATIONS_OPERATIONS
from nncf.torch.graph.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.torch.graph.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import GROUP_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import RELU_OPERATIONS
from nncf.torch.graph.pattern_operations import LINEAR_OPERATIONS

QUANTIZATION_IGNORE_PATTERNS = HWFusedPatterns()


def create_swish_act() -> GraphPattern:
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.PATTERN_INPUT_NODE_TYPE)
    sigmoid_node = pattern.add_node(label='SIGMOID', type='sigmoid')
    mul_node = pattern.add_node(label='MUL', type='__mul__')

    pattern.add_edge(input_pattern_node, sigmoid_node)
    pattern.add_edge(sigmoid_node, mul_node)
    pattern.add_edge(input_pattern_node, mul_node)
    return pattern


def create_h_swish_act() -> GraphPattern:
    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.PATTERN_INPUT_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type='__add__')
    hardtanh_node = pattern.add_node(label='HARDTANH', type='hardtanh')
    truediv_node = pattern.add_node(label='DIV', type='__truediv__')
    mul_node = pattern.add_node(label='MUL', type='__mul__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, truediv_node)
    pattern.add_edge(truediv_node, mul_node)
    return pattern


def create_h_sigmoid_act() -> GraphPattern:
    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label='*INPUT_NODE*', type=GraphPattern.PATTERN_INPUT_NODE_TYPE)
    add_node = pattern.add_node(label='ADD', type='__add__')
    hardtanh_node = pattern.add_node(label='HARTANH', type='hardtanh')
    truediv_node = pattern.add_node(label='DIV', type='__truediv__')

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, truediv_node)
    return pattern


def register_all_patterns():
    linear_ops = GraphPattern()
    linear_ops.add_node(**LINEAR_OPERATIONS)
    QUANTIZATION_IGNORE_PATTERNS.register(linear_ops, LINEAR_OPERATIONS['label'], match=False)

    batch_norm = GraphPattern()
    batch_norm.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    QUANTIZATION_IGNORE_PATTERNS.register(batch_norm, BATCH_NORMALIZATION_OPERATIONS['label'], match=False)

    atomic_activations = GraphPattern()
    atomic_activations.add_node(**ACTIVATIONS_OPERATIONS)
    swish = create_swish_act()
    h_sigmoid = create_h_sigmoid_act()
    h_swish = create_h_swish_act()
    activations = atomic_activations | swish | h_swish | h_sigmoid
    QUANTIZATION_IGNORE_PATTERNS.register(activations, 'ACTIVATIONS', match=False)

    arithmetic_ops = GraphPattern()
    arithmetic_ops.add_node(**ARITHMETIC_OPERATIONS)
    QUANTIZATION_IGNORE_PATTERNS.register(arithmetic_ops, ARITHMETIC_OPERATIONS['label'], match=False)

    batch_norm_activations_permutation = batch_norm + activations | activations + batch_norm | batch_norm | activations

    QUANTIZATION_IGNORE_PATTERNS.register(linear_ops + batch_norm_activations_permutation, 'LINEAR + BN_ACT_PERM',
                                          match=True)
    QUANTIZATION_IGNORE_PATTERNS.register(batch_norm + activations, 'BN + ACTIVATIONS', match=True)
    QUANTIZATION_IGNORE_PATTERNS.register(activations + batch_norm, 'ACTIVATIONS + BN', match=True)
    QUANTIZATION_IGNORE_PATTERNS.register(arithmetic_ops + batch_norm_activations_permutation,
                                          'ARITHMETIC + BN_ACT_PERM', match=True)

    group_norm = GraphPattern()
    group_norm.add_node(**GROUP_NORMALIZATION_OPERATIONS)
    relu = GraphPattern()
    relu.add_node(**RELU_OPERATIONS)
    QUANTIZATION_IGNORE_PATTERNS.register(group_norm + relu, 'GROUP_NORM + RELU', match=True)


register_all_patterns()


def get_full_pattern_graph():
    return QUANTIZATION_IGNORE_PATTERNS.get_full_pattern_graph()
