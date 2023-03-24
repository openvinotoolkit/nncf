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
    PatternNames.L2_NORM:'Not relevant for ONNX.',
    PatternNames.GROUP_NORM_RELU:'Not relevant for ONNX.',
    PatternNames.LINEAR_CONST_MULTIPLY:'Not relevant for ONNX.',
    PatternNames.ADD_SCALE_SHIFT_OUTPUT: 'Not relevant for ONNX.',
    PatternNames.BATCH_INDEX: 'Not relevant for ONNX.',
    PatternNames.MVN_SCALE_SHIFT: 'Not relevant for ONNX.',
    PatternNames.NORMALIZE_L2_MULTIPLY: 'Not relevant for ONNX.',
    PatternNames.LINEAR_WITH_BIAS: 'Linear layers contains biases in ONNX.',
    PatternNames.SE_BLOCK: 'Not relevant for ONNX.',
    PatternNames.STABLE_DIFFUSION: 'Not relevant for ONNX.',
    PatternNames.SOFTMAX_DIV: 'Not relevant for ONNX.',
    PatternNames.SOFTMAX_RESHAPE_MATMUL: 'Not relevant for ONNX.',
    PatternNames.SOFTMAX_RESHAPE_TRANSPOSE_MATMUL: 'Not relevant for ONNX.',
    PatternNames.SOFTMAX_RESHAPE_TRANSPOSE_GATHER_MATMUL: 'Not relevant for ONNX.',
    PatternNames.EQUAL_LOGICALNOT: 'Not relevant for ONNX.',
    PatternNames.FC_BN_HSWISH_ACTIVATION: 'Not relevant for ONNX.',
    PatternNames.HSWISH_ACTIVATION: 'Not relevant for ONNX.',
    PatternNames.HSWISH_ACTIVATION_V2: 'Not relevant for ONNX.',
    PatternNames.HSWISH_ACTIVATION_WITHOUT_DENOMINATOR: 'Not relevant for ONNX.',
    PatternNames.SOFTMAX: 'Not relevant for ONNX.',
    PatternNames.INPUT_CONVERT_TRANSPOSE_PROCESSING: 'Not relevant for ONNX.',
    PatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_ADD: 'Not relevant for ONNX.',
    PatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_SCALE_SHIFT: 'Not relevant for ONNX.',
    PatternNames.INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT: 'Not relevant for ONNX.',
    PatternNames.INPUT_REVERSE_ADD: 'Not relevant for ONNX.',
    PatternNames.INPUT_REVERSE_SCALE_SHIFT: 'Not relevant for ONNX.',
    PatternNames.INPUT_TRANSPOSE_PROCESSING: 'Not relevant for ONNX.',
    PatternNames.INPUT_TRANSPOSE_REVERSE_ADD: 'Not relevant for ONNX.',
    PatternNames.INPUT_TRANSPOSE_SCALE_SHIFT: 'Not relevant for ONNX.',
    PatternNames.LINEAR_ARITHMETIC_ACTIVATIONS: 'Not relevant for ONNX.',
    PatternNames.HSWISH_ACTIVATION_CLAMP_MULTIPLY: 'Not relevant for ONNX.',
    PatternNames.LINEAR_BIASED_SCALE_SHIFT: 'Not relevant for ONNX.',
    PatternNames.LINEAR_ACTIVATION_SCALE_SHIFT: 'Not relevant for ONNX.',
    PatternNames.LINEAR_BIASED_ACTIVATION_SCALE_SHIFT: 'Not relevant for ONNX.',
    PatternNames.LINEAR_ELEMENTWISE: 'Not relevant for ONNX.',
    PatternNames.LINEAR_BIASED_ELEMENTWISE: 'Not relevant for ONNX.',
    PatternNames.LINEAR_ACTIVATION_ELEMENTWISE: 'Not relevant for ONNX.',
    PatternNames.LINEAR_BIASED_ACTIVATION_ELEMENTWISE: 'Not relevant for ONNX.',
    PatternNames.MVN_SCALE_SHIFT_ACTIVATIONS: 'Not relevant for ONNX.',
}


def test_pattern_manager():
    check_patterns(BackendType.ONNX, IGNORED_PATTERN_REASONS)
