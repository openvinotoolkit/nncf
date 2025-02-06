# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nncf.common.graph.patterns import HWFusedPatternNames
from nncf.common.graph.patterns import IgnoredPatternNames
from nncf.common.utils.backend import BackendType
from tests.cross_fw.shared.patterns import check_hw_patterns
from tests.cross_fw.shared.patterns import check_ignored_patterns

IGNORING_HW_PATTERN_REASONS = {
    HWFusedPatternNames.L2_NORM: "Not relevant for ONNX.",
    HWFusedPatternNames.GROUP_NORM_RELU: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_CONST_MULTIPLY: "Not relevant for ONNX.",
    HWFusedPatternNames.ADD_SCALE_SHIFT_OUTPUT: "Not relevant for ONNX.",
    HWFusedPatternNames.BATCH_INDEX: "Not relevant for ONNX.",
    HWFusedPatternNames.NORMALIZE_L2_MULTIPLY: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_WITH_BIAS: "Linear layers contains biases in ONNX.",
    HWFusedPatternNames.SOFTMAX_DIV: "Not relevant for ONNX.",
    HWFusedPatternNames.HSWISH_ACTIVATION_V2: "Is already covered by HSWISH_ACTIVATION for ONNX.",
    HWFusedPatternNames.SOFTMAX: "Not relevant for ONNX.",
    HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_PROCESSING: "Not relevant for ONNX.",
    HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_ADD: "Not relevant for ONNX.",
    HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_SCALE_SHIFT: "Not relevant for ONNX.",
    HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT: "Not relevant for ONNX.",
    HWFusedPatternNames.INPUT_REVERSE_ADD: "Not relevant for ONNX.",
    HWFusedPatternNames.INPUT_REVERSE_SCALE_SHIFT: "Not relevant for ONNX.",
    HWFusedPatternNames.INPUT_TRANSPOSE_PROCESSING: "Not relevant for ONNX.",
    HWFusedPatternNames.INPUT_TRANSPOSE_REVERSE_ADD: "Not relevant for ONNX.",
    HWFusedPatternNames.INPUT_TRANSPOSE_SCALE_SHIFT: "Not relevant for ONNX.",
    HWFusedPatternNames.HSWISH_ACTIVATION_CLAMP_MULTIPLY: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_BIASED_SCALE_SHIFT: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_ACTIVATION_SCALE_SHIFT: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_BIASED_ACTIVATION_SCALE_SHIFT: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_ELEMENTWISE: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_BIASED_ELEMENTWISE: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_ACTIVATION_ELEMENTWISE: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_BIASED_ACTIVATION_ELEMENTWISE: "Not relevant for ONNX.",
    HWFusedPatternNames.MVN_SCALE_SHIFT_ACTIVATIONS: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_ACTIVATIONS_UNSQUEEZE_BN_SQUEEZE: "Not relevant for ONNX.",
    HWFusedPatternNames.ARITHMETIC_ACTIVATIONS_ARITHMETIC: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_BATCH_TO_SPACE_ARITHMETIC_ACTIVATIONS: "Not relevant for ONNX.",
    HWFusedPatternNames.LINEAR_BATCH_TO_SPACE_SCALE_SHIFT_ACTIVATIONS: "Not relevant for ONNX.",
}

IGNORING_IGNORED_PATTERN_REASONS = {
    IgnoredPatternNames.FC_BN_HSWISH_ACTIVATION: "Not relevant for ONNX.",
    IgnoredPatternNames.EQUAL_LOGICALNOT: "Not relevant for ONNX.",
}


def test_pattern_manager():
    check_hw_patterns(BackendType.ONNX, IGNORING_HW_PATTERN_REASONS)
    check_ignored_patterns(BackendType.ONNX, IGNORING_IGNORED_PATTERN_REASONS)
