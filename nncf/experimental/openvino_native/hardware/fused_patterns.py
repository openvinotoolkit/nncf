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

from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import HWFusedPatterns

from nncf.experimental.openvino_native.hardware.pattern_operations import LINEAR_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import BATCH_NORMALIZATION_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import ATOMIC_ACTIVATIONS_OPERATIONS
from nncf.experimental.openvino_native.hardware.pattern_operations import ARITHMETIC_OPERATIONS

from nncf.experimental.openvino_native.hardware.patterns import create_input_preprocessing_pattern
from nncf.experimental.openvino_native.hardware.patterns import create_scale_shift


def _get_openvino_hw_fused_patterns() -> HWFusedPatterns:
    hw_fused_patterns = HWFusedPatterns()

    linear_ops = GraphPattern()
    linear_ops.add_node(**LINEAR_OPERATIONS)
    hw_fused_patterns.register(linear_ops, LINEAR_OPERATIONS['label'], match=False)

    batch_norm = GraphPattern()
    batch_norm.add_node(**BATCH_NORMALIZATION_OPERATIONS)
    hw_fused_patterns.register(batch_norm, BATCH_NORMALIZATION_OPERATIONS['label'], match=False)

    activations = GraphPattern()
    activations.add_node(**ATOMIC_ACTIVATIONS_OPERATIONS)
    hw_fused_patterns.register(activations, 'ACTIVATIONS', match=False)

    arithmetic_ops = GraphPattern()
    arithmetic_ops.add_node(**ARITHMETIC_OPERATIONS)
    hw_fused_patterns.register(arithmetic_ops, ARITHMETIC_OPERATIONS['label'], match=False)

    hw_fused_patterns.register(linear_ops + arithmetic_ops, 'LINEAR + ARITHMETIC', match=True)
    hw_fused_patterns.register(batch_norm + activations, 'BN + ACTIVATIONS', match=True)
    hw_fused_patterns.register(activations + batch_norm, 'ACTIVATIONS + BN', match=True)

    input_preprocessing_pattern = create_input_preprocessing_pattern()
    hw_fused_patterns.register(input_preprocessing_pattern,
                               'INPUT_PREPROCESSING', match=True)

    scale_shift = create_scale_shift()
    hw_fused_patterns.register(scale_shift, 'SCALE_SHIFT', match=True)

    return hw_fused_patterns


OPENVINO_HW_FUSED_PATTERNS = _get_openvino_hw_fused_patterns()
