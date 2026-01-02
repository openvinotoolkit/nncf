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
    HWFusedPatternNames.ADD_SCALE_SHIFT_OUTPUT: "Not relevant for Torch.",
    HWFusedPatternNames.BATCH_INDEX: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_WITH_BIAS: "Not relevant for Torch.",
    HWFusedPatternNames.MVN_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.NORMALIZE_L2_MULTIPLY: "Not relevant for Torch.",
    HWFusedPatternNames.SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.SOFTMAX_DIV: "Not relevant for Torch.",
    HWFusedPatternNames.HSWISH_ACTIVATION: "Not relevant for Torch.",
    HWFusedPatternNames.HSWISH_ACTIVATION_V2: "Not relevant for Torch.",
    HWFusedPatternNames.HSWISH_ACTIVATION_WITHOUT_DENOMINATOR: "Not relevant for Torch.",
    HWFusedPatternNames.SOFTMAX: "Not relevant for Torch.",
    HWFusedPatternNames.SWISH_WITH_HARD_SIGMOID: "Not relevant for Torch.",
    HWFusedPatternNames.SWISH_WITH_SIGMOID: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_PROCESSING: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_ADD: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_REVERSE_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_CONVERT_TRANSPOSE_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_PROCESSING: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_REVERSE_ADD: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_REVERSE_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_TRANSPOSE_PROCESSING: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_TRANSPOSE_REVERSE_ADD: "Not relevant for Torch.",
    HWFusedPatternNames.INPUT_TRANSPOSE_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.ACTIVATIONS_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.ARITHMETIC_ACTIVATIONS_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.ARITHMETIC_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.ARITHMETIC_SCALE_SHIFT_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.BATCH_NORM_SCALE_SHIFT_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_ACTIVATIONS_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_ARITHMETIC_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_ARITHMETIC_ACTIVATIONS_ARITHMETIC: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_BATCH_NORM_SCALE_SHIFT_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_SCALE_SHIFT_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.SCALE_SHIFT_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.HSWISH_ACTIVATION_CLAMP_MULTIPLY: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_ACTIVATION_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_BIASED_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_BIASED_ACTIVATION_SCALE_SHIFT: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_ELEMENTWISE: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_BIASED_ELEMENTWISE: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_ACTIVATION_ELEMENTWISE: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_BIASED_ACTIVATION_ELEMENTWISE: "Not relevant for Torch.",
    HWFusedPatternNames.MVN_SCALE_SHIFT_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_SQUEEZE_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_SQUEEZE_ARITHMETIC_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_ACTIVATIONS_UNSQUEEZE_BN_SQUEEZE: "Not relevant for Torch.",
    HWFusedPatternNames.MVN: "Not relevant for Torch.",
    HWFusedPatternNames.GELU: "Not relevant for Torch.",
    HWFusedPatternNames.ARITHMETIC_ACTIVATIONS_ARITHMETIC: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_BATCH_TO_SPACE_ARITHMETIC_ACTIVATIONS: "Not relevant for Torch.",
    HWFusedPatternNames.LINEAR_BATCH_TO_SPACE_SCALE_SHIFT_ACTIVATIONS: "Not relevant for Torch.",
}

IGNORING_IGNORED_PATTERN_REASONS = {
    IgnoredPatternNames.FC_BN_HSWISH_ACTIVATION: "Not relevant for Torch.",
    IgnoredPatternNames.EQUAL_LOGICALNOT: "Not relevant for Torch.",
}


def test_pattern_manager():
    check_hw_patterns(BackendType.TORCH, IGNORING_HW_PATTERN_REASONS)
    check_ignored_patterns(BackendType.TORCH, IGNORING_IGNORED_PATTERN_REASONS)
