"""
 Copyright (c) 2023 Intel Corporation
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

from nncf.common.utils.backend import BackendType
from nncf.common.utils.registry import Registry
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import PatternNames
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.patterns.manager import TargetDevice
from nncf.torch.graph.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.torch.graph.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.torch.graph.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import GROUP_NORMALIZATION_OPERATIONS
from nncf.torch.graph.pattern_operations import LINEAR_OPERATIONS
from nncf.torch.graph.pattern_operations import RELU_OPERATIONS
from nncf.torch.graph.patterns import create_fc_conv_mul
from nncf.torch.graph.patterns import create_h_sigmoid_act
from nncf.torch.graph.patterns import create_h_swish_act
from nncf.torch.graph.patterns import create_swish_act
from nncf.torch.graph.patterns import create_l2_norm


PT_HW_FUSED_PATTERNS = Registry('torch')

# ATOMIC OPERATIONS


@PT_HW_FUSED_PATTERNS.register(PatternNames.L2_NORM)
def create_l2_norm_operations():
    return create_l2_norm()

# COMBINATIONS


@PT_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ARITHMETIC)
def create_linear_arithmetic_operations():
    linear = linear_operations()
    arithmetic = arithmetic_operations()
    linear.join_patterns(arithmetic)
    return linear


@PT_HW_FUSED_PATTERNS.register(PatternNames.BATCH_NORM_ACTIVATIONS)
def create_batch_norm_activations_operations():
    batch_norm = batch_norm_operations()
    activations = activation_operations()
    batch_norm.join_patterns(activations)
    return batch_norm


@PT_HW_FUSED_PATTERNS.register(PatternNames.ACTIVATIONS_BATCH_NORM)
def create_activations_batch_norm_operations():
    batch_norm = batch_norm_operations()
    activations = activation_operations()
    activations.join_patterns(batch_norm)
    return activations


@PT_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_ACTIVATIONS_BATCH_NORM_PERMUTATIONS)
def create_linear_activation_batch_norm_permutation():
    linear = linear_operations()
    bn_activations_permutations = batch_norm_activations_permutations()
    linear.join_patterns(bn_activations_permutations)
    return linear


@PT_HW_FUSED_PATTERNS.register(PatternNames.ARITHMETIC_BATCH_NORM_ACTIVATIONS_PERMUTATIONS)
def create_arithmetic_batch_norm_activations_permutations():
    arithmetic = arithmetic_operations()
    bn_act_perm = batch_norm_activations_permutations()
    arithmetic.join_patterns(bn_act_perm)
    return arithmetic


@PT_HW_FUSED_PATTERNS.register(PatternNames.GROUP_NORM_RELU)
def create_group_norm_relu_operations():
    group_norm = GraphPattern()
    group_norm.add_node(**GROUP_NORMALIZATION_OPERATIONS)
    relu = GraphPattern()
    relu.add_node(**RELU_OPERATIONS)
    group_norm.join_patterns(relu)
    return group_norm


@PT_HW_FUSED_PATTERNS.register(PatternNames.LINEAR_CONST_MULTIPLY)
def create_linear_const_multiply():
    return create_fc_conv_mul()


def linear_operations():
    pattern = GraphPattern()
    pattern.add_node(**LINEAR_OPERATIONS)
    return pattern

def arithmetic_operations():
    pattern = GraphPattern()
    pattern.add_node(**ARITHMETIC_OPERATIONS)
    return pattern


def batch_norm_operations():
    pattern = GraphPattern()
    pattern.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    return pattern


def activation_operations():
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


def batch_norm_activations_permutations():
    batch_norm = batch_norm_operations()
    activations = activation_operations()

    bn_act = GraphPattern()
    bn_act.add_pattern_alternative(batch_norm)
    bn_act.join_patterns(activations)

    act_bn = GraphPattern()
    act_bn.add_pattern_alternative(activations)
    act_bn.join_patterns(batch_norm)

    pattern = GraphPattern()
    pattern.add_pattern_alternative(bn_act)
    pattern.add_pattern_alternative(act_bn)
    pattern.add_pattern_alternative(batch_norm)
    pattern.add_pattern_alternative(activations)
    return pattern


def get_torch_hw_patterns():
    return PatternsManager().get_full_pattern_graph(BackendType.TORCH, TargetDevice.ANY)
