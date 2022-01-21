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
from nncf.common.graph.patterns import HWFusedPatterns

from nncf.experimental.onnx.graph.pattern_operations import LINEAR_OPERATIONS
from nncf.experimental.onnx.graph.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.experimental.onnx.graph.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.experimental.onnx.graph.pattern_operations import ARITHMETIC_OPERATIONS
from nncf.experimental.onnx.graph.pattern_operations import MATMUL_OPERATIONS

from nncf.experimental.onnx.graph.patterns import create_h_sigmoid_act


def _get_onnx_hw_fused_patterns() -> HWFusedPatterns:
    hw_fused_patterns = HWFusedPatterns()

    linear_ops = GraphPattern()
    linear_ops.add_node(**LINEAR_OPERATIONS)
    hw_fused_patterns.register(linear_ops, LINEAR_OPERATIONS['label'], match=False)

    batch_norm = GraphPattern()
    batch_norm.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    hw_fused_patterns.register(batch_norm, BATCH_NORMALIZATION_OPERATIONS['label'], match=False)

    matmul_ops = GraphPattern()
    matmul_ops.add_node(**MATMUL_OPERATIONS)
    hw_fused_patterns.register(linear_ops, MATMUL_OPERATIONS['label'], match=False)

    atomic_activations = GraphPattern()
    atomic_activations.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
    h_sigmoid = create_h_sigmoid_act()
    activations = ATOMIC_ACTIVATIONS_OPERATIONS | h_sigmoid
    hw_fused_patterns.register(activations, 'ACTIVATIONS', match=False)

    arithmetic_ops = GraphPattern()
    arithmetic_ops.add_node(**ARITHMETIC_OPERATIONS)
    hw_fused_patterns.register(arithmetic_ops, ARITHMETIC_OPERATIONS['label'], match=False)

    batch_norm_activations_permutation = batch_norm + activations | activations + batch_norm | batch_norm | activations

    hw_fused_patterns.register(linear_ops + batch_norm_activations_permutation, 'LINEAR + BN_ACT_PERM',
                               match=True)
    hw_fused_patterns.register(matmul_ops + arithmetic_ops, 'MATMUL + ARITHMETIC',
                               match=True)

    hw_fused_patterns.register(batch_norm + activations, 'BN + ACTIVATIONS', match=True)
    hw_fused_patterns.register(activations + batch_norm, 'ACTIVATIONS + BN', match=True)
    hw_fused_patterns.register(arithmetic_ops + batch_norm_activations_permutation,
                               'ARITHMETIC + BN_ACT_PERM', match=True)

    return hw_fused_patterns


ONNX_HW_FUSED_PATTERNS = _get_onnx_hw_fused_patterns()
