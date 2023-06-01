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

from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import HWFusedPatternNames
from nncf.common.utils.registry import Registry
from nncf.torch.graph.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.torch.graph.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.torch.graph.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import GROUP_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import LINEAR_OPERATIONS
from nncf.torch.graph.pattern_operations import RELU_OPERATIONS

PT_HW_FUSED_PATTERNS = Registry("torch")

# ATOMIC OPERATIONS


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.L2_NORM)
def create_l2_norm_operations() -> GraphPattern:
    pattern = GraphPattern()

    outside_pattern_node = pattern.add_node(label="*OUTSIDE_PATTERN_NODE*", type=GraphPattern.NON_PATTERN_NODE_TYPE)
    pow_node = pattern.add_node(label="POW", type="pow")
    sum_node = pattern.add_node(label="SUM", type="sum")
    sqrt_node = pattern.add_node(label="SQRT", type="sqrt")
    add_node = pattern.add_node(label="ADD", type="__add__")
    div_node = pattern.add_node(label="DIV", type="div")
    mul_node = pattern.add_node(label="MUL", type="__rmul__")

    pattern.add_edge(outside_pattern_node, pow_node)
    pattern.add_edge(pow_node, sum_node)
    pattern.add_edge(sum_node, sqrt_node)
    pattern.add_edge(sqrt_node, add_node)
    pattern.add_edge(add_node, div_node)
    pattern.add_edge(div_node, mul_node)
    pattern.add_edge(outside_pattern_node, div_node)
    return pattern


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.MATMUL_SOFTMAX_MATMUL)
def create_matmul_softmax_matmul() -> GraphPattern:
    matmul_aliases = ["linear", "addmm", "matmul", "bmm", "mm", "baddbmm"]
    pattern = GraphPattern()
    softmax_1 = pattern.add_node(label="SOFTMAX", type="softmax")
    mat_mul_1_1 = pattern.add_node(label="MATMUL_1", type=matmul_aliases)
    mat_mul_2_1 = pattern.add_node(label="MATMUL_2", type=matmul_aliases)

    any_1 = pattern.add_node(label="ANY", type=GraphPattern.NON_PATTERN_NODE_TYPE)

    pattern.add_edge(mat_mul_1_1, softmax_1)
    pattern.add_edge(softmax_1, mat_mul_2_1)
    pattern.add_edge(any_1, mat_mul_2_1)

    softmax_2 = pattern.add_node(label="SOFTMAX", type="softmax")
    add_2 = pattern.add_node(label="ADD", type=["add", "__add__", "__iadd__", "__radd__"])
    mat_mul_1_2 = pattern.add_node(label="MATMUL_1", type=matmul_aliases)
    mat_mul_2_2 = pattern.add_node(label="MATMUL_2", type=matmul_aliases)

    any_2 = pattern.add_node(label="ANY", type=GraphPattern.NON_PATTERN_NODE_TYPE)

    pattern.add_edge(mat_mul_1_2, add_2)
    pattern.add_edge(add_2, softmax_2)
    pattern.add_edge(softmax_2, mat_mul_2_2)
    pattern.add_edge(any_2, mat_mul_2_2)

    return pattern


# COMBINATIONS


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ARITHMETIC)
def create_linear_arithmetic_operations() -> GraphPattern:
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    linear.join_patterns(arithmetic)
    return linear


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.BATCH_NORM_ACTIVATIONS)
def create_batch_norm_activations_operations() -> GraphPattern:
    batch_norm = batch_norm_operations()
    activations = activation_operations()
    batch_norm.join_patterns(activations)
    return batch_norm


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ACTIVATIONS_BATCH_NORM)
def create_activations_batch_norm_operations() -> GraphPattern:
    batch_norm = batch_norm_operations()
    activations = activation_operations()
    activations.join_patterns(batch_norm)
    return activations


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BATCH_NORM)
def create_linear_batch_norm_operations() -> GraphPattern:
    linear = linear_operations()
    batch_norm = batch_norm_operations()
    linear.join_patterns(batch_norm)
    return linear


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ACTIVATIONS)
def create_linear_activation_operations() -> GraphPattern:
    linear = linear_operations()
    activation = activation_operations()
    linear.join_patterns(activation)
    return linear


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_BATCH_NORM_ACTIVATIONS)
def create_linear_batch_norm_activation_operations() -> GraphPattern:
    linear_bn = create_linear_batch_norm_operations()
    activations = activation_operations()
    linear_bn.join_patterns(activations)
    return linear_bn


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_ACTIVATIONS_BATCH_NORM)
def create_linear_activation_batch_norm_activations() -> GraphPattern:
    linear_act = create_linear_activation_operations()
    batch_norm = batch_norm_operations()
    linear_act.join_patterns(batch_norm)
    return linear_act


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_BATCH_NORM)
def create_arithmetic_batch_norm_operations() -> GraphPattern:
    arithmetic = arithmetic_operations()
    batch_norm = batch_norm_operations()
    arithmetic.join_patterns(batch_norm)
    return arithmetic


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_ACTIVATIONS)
def create_arithmetic_activations_operations() -> GraphPattern:
    arithmetic = arithmetic_operations()
    activation = activation_operations()
    arithmetic.join_patterns(activation)
    return arithmetic


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_BATCH_NORM_ACTIVATIONS)
def create_arithmetic_batch_norm_activations_operations() -> GraphPattern:
    arithmetic_bn = create_arithmetic_batch_norm_operations()
    activation = activation_operations()
    arithmetic_bn.join_patterns(activation)
    return arithmetic_bn


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.ARITHMETIC_ACTIVATIONS_BATCH_NORM)
def create_arithmetic_activations_batch_norm_operations() -> GraphPattern:
    arithmetic_act = create_arithmetic_activations_operations()
    batch_norm = batch_norm_operations()
    arithmetic_act.join_patterns(batch_norm)
    return arithmetic_act


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.GROUP_NORM_RELU)
def create_group_norm_relu_operations() -> GraphPattern:
    group_norm = GraphPattern()
    group_norm.add_node(**GROUP_NORMALIZATION_OPERATIONS)
    relu = GraphPattern()
    relu.add_node(**RELU_OPERATIONS)
    group_norm.join_patterns(relu)
    return group_norm


@PT_HW_FUSED_PATTERNS.register(HWFusedPatternNames.LINEAR_CONST_MULTIPLY)
def create_linear_const_multiply() -> GraphPattern:
    pattern = GraphPattern()
    linear_node = pattern.add_node(label="linear", type="linear")
    mul_node = pattern.add_node(label="MUL", type="__mul__")
    pattern.add_edge(linear_node, mul_node)

    return pattern


def linear_operations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**LINEAR_OPERATIONS)
    return pattern


def arithmetic_operations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**ARITHMETIC_OPERATIONS)
    return pattern


def batch_norm_operations() -> GraphPattern:
    pattern = GraphPattern()
    pattern.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    return pattern


def activation_operations() -> GraphPattern:
    atomic_activations = GraphPattern()
    atomic_activations.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
    swish = create_swish_act()
    h_sigmoid = create_h_sigmoid_act()
    h_swish = create_h_swish_act()

    pattern = GraphPattern()
    pattern.add_pattern_alternative(atomic_activations)
    pattern.add_pattern_alternative(swish)
    pattern.add_pattern_alternative(h_swish)
    pattern.add_pattern_alternative(h_sigmoid)
    return pattern


def create_swish_act() -> GraphPattern:
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label="*INPUT_NODE*", type=GraphPattern.NON_PATTERN_NODE_TYPE)
    sigmoid_node = pattern.add_node(label="SIGMOID", type="sigmoid")
    mul_node = pattern.add_node(label="MUL", type="__mul__")

    pattern.add_edge(input_pattern_node, sigmoid_node)
    pattern.add_edge(sigmoid_node, mul_node)
    pattern.add_edge(input_pattern_node, mul_node)
    return pattern


def create_h_swish_act() -> GraphPattern:
    main_pattern = GraphPattern()

    # Mul -> Div version
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label="*INPUT_NODE*", type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label="ADD", type="__add__")
    hardtanh_node = pattern.add_node(label="HARDTANH", type="hardtanh")
    truediv_node = pattern.add_node(label="DIV", type="__truediv__")
    mul_node = pattern.add_node(label="MUL", type="__mul__")

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, truediv_node)
    pattern.add_edge(truediv_node, mul_node)
    main_pattern.add_pattern_alternative(pattern)

    # Div -> Mul version
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label="*INPUT_NODE*", type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label="ADD", type="__add__")
    hardtanh_node = pattern.add_node(label="HARDTANH", type="hardtanh")
    mul_node = pattern.add_node(label="MUL", type="__mul__")
    truediv_node = pattern.add_node(label="DIV", type="__truediv__")

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, mul_node)
    pattern.add_edge(mul_node, truediv_node)
    main_pattern.add_pattern_alternative(pattern)

    # ReLU6 version - Mul -> Div
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label="*INPUT_NODE*", type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label="ADD", type="__add__")
    relu6_node = pattern.add_node(label="RELU6", type="relu6")
    mul_node = pattern.add_node(label="MUL", type="__mul__")
    truediv_node = pattern.add_node(label="DIV", type="__truediv__")

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(input_pattern_node, mul_node)
    pattern.add_edge(add_node, relu6_node)
    pattern.add_edge(hardtanh_node, mul_node)
    pattern.add_edge(mul_node, truediv_node)
    main_pattern.add_pattern_alternative(pattern)

    # ReLU6 version - Div -> Mul
    pattern = GraphPattern()
    input_pattern_node = pattern.add_node(label="*INPUT_NODE*", type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label="ADD", type="__add__")
    relu6_node = pattern.add_node(label="RELU6", type="relu6")
    truediv_node = pattern.add_node(label="DIV", type="__truediv__")
    mul_node = pattern.add_node(label="MUL", type="__mul__")

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

    input_pattern_node = pattern.add_node(label="*INPUT_NODE*", type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label="ADD", type="__add__")
    hardtanh_node = pattern.add_node(label="HARDTANH", type="hardtanh")
    truediv_node = pattern.add_node(label="DIV", type="__truediv__")

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, hardtanh_node)
    pattern.add_edge(hardtanh_node, truediv_node)

    main_pattern.add_pattern_alternative(pattern)

    # ReLU6 version
    pattern = GraphPattern()

    input_pattern_node = pattern.add_node(label="*INPUT_NODE*", type=GraphPattern.NON_PATTERN_NODE_TYPE)
    add_node = pattern.add_node(label="ADD", type="__add__")
    relu6_node = pattern.add_node(label="RELU6", type="relu6")
    truediv_node = pattern.add_node(label="DIV", type="__truediv__")

    pattern.add_edge(input_pattern_node, add_node)
    pattern.add_edge(add_node, relu6_node)
    pattern.add_edge(hardtanh_node, truediv_node)

    main_pattern.add_pattern_alternative(pattern)

    return main_pattern
