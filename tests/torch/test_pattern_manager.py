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
from nncf.common.graph.patterns import PatternNames
from tests.shared.patterns import check_patterns


IGNORED_PATTERN_REASONS = {
    PatternNames.ADD_SCALE_SHIFT_OUTPUT: 'Not relevant for Torch.',
    PatternNames.BATCH_INDEX: 'Not relevant for Torch.',
    PatternNames.EQUAL_LOGICALNOT: 'Not relevant for Torch.',
    PatternNames.FC_BN_HSWISH_ACTIVATION: 'Not relevant for Torch.',
    PatternNames.LINEAR_WITH_BIAS: 'Not relevant for Torch.',
    PatternNames.MVN_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.NORMALIZE_L2_MULTIPLY: 'Not relevant for Torch.',
    PatternNames.SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.SE_BLOCK: 'Not relevant for Torch.',
    PatternNames.SOFTMAX_DIV: 'Not relevant for Torch.',
    PatternNames.SOFTMAX_RESHAPE_MATMUL: 'Not relevant for Torch.',
    PatternNames.SOFTMAX_RESHAPE_TRANSPOSE_GATHER_MATMUL: 'Not relevant for Torch.',
    PatternNames.SOFTMAX_RESHAPE_TRANSPOSE_MATMUL: 'Not relevant for Torch.',
    PatternNames.STABLE_DIFFUSION: 'Not relevant for Torch.',
    PatternNames.HSWISH_ACTIVATION: 'Not relevant for Torch.',
    PatternNames.HSWISH_ACTIVATION_V2: 'Not relevant for Torch.',
    PatternNames.HSWISH_ACTIVATION_WITHOUT_DENOMINATOR: 'Not relevant for Torch.',
    PatternNames.SOFTMAX: 'Not relevant for Torch.',
    PatternNames.SWISH_WITH_HARD_SIGMOID: 'Not relevant for Torch.',
    PatternNames.SWISH_WITH_SIGMOID: 'Not relevant for Torch.',
    PatternNames.INPUT_CONVERT_TRANSPOSE_PROCESSING: 'Not relevant for Torch.',
    PatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_ADD: 'Not relevant for Torch.',
    PatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.INPUT_PROCESSING: 'Not relevant for Torch.',
    PatternNames.INPUT_REVERSE_ADD: 'Not relevant for Torch.',
    PatternNames.INPUT_REVERSE_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.INPUT_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.INPUT_SHIFT_SCALE: 'Not relevant for Torch.',
    PatternNames.INPUT_TRANSPOSE_PROCESSING: 'Not relevant for Torch.',
    PatternNames.INPUT_TRANSPOSE_REVERSE_ADD: 'Not relevant for Torch.',
    PatternNames.INPUT_TRANSPOSE_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.ACTIVATIONS_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.ARITHMETIC_ACTIVATIONS_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.ARITHMETIC_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.ARITHMETIC_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for Torch.',
    PatternNames.BATCH_NORM_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for Torch.',
    PatternNames.LINEAR_ACTIVATIONS_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.LINEAR_ARITHMETIC_ACTIVATIONS: 'Not relevant for Torch.',
    PatternNames.LINEAR_BATCH_NORM_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for Torch.',
    PatternNames.LINEAR_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for Torch.',
    PatternNames.SCALE_SHIFT_ACTIVATIONS: 'Not relevant for Torch.',
    PatternNames.HSWISH_ACTIVATION_CLAMP_MULTIPLY: 'Not relevant for Torch.',
    PatternNames.LINEAR_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.LINEAR_ACTIVATION_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.LINEAR_BIASED_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.LINEAR_BIASED_ACTIVATION_SCALE_SHIFT: 'Not relevant for Torch.',
    PatternNames.LINEAR_ELEMENTWISE: 'Not relevant for Torch.',
    PatternNames.LINEAR_BIASED_ELEMENTWISE: 'Not relevant for Torch.',
    PatternNames.LINEAR_ACTIVATION_ELEMENTWISE: 'Not relevant for Torch.',
    PatternNames.LINEAR_BIASED_ACTIVATION_ELEMENTWISE: 'Not relevant for Torch.',
    PatternNames.MVN_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for Torch.',
    PatternNames.LINEAR_SQUEEZE_ACTIVATIONS: 'Not relevant for Torch.'
}


def test_pattern_manager():
    check_patterns(BackendType.TORCH, IGNORED_PATTERN_REASONS)
