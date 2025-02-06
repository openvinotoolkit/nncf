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
from nncf.common.utils.backend import BackendType
from tests.cross_fw.shared.patterns import check_hw_patterns
from tests.cross_fw.shared.patterns import check_ignored_patterns

IGNORING_HW_PATTERN_REASONS = {
    HWFusedPatternNames.L2_NORM: "Not relevant for OpenVINO.",
    HWFusedPatternNames.GROUP_NORM_RELU: "Not relevant for OpenVINO.",
    HWFusedPatternNames.LINEAR_CONST_MULTIPLY: "Not relevant for OpenVINO.",
    HWFusedPatternNames.SWISH_WITH_SIGMOID: "Swish exists in the OpenVINO as layer.",
    HWFusedPatternNames.ACTIVATIONS_SCALE_SHIFT: "Not relevant for OpenVINO.",
    HWFusedPatternNames.ARITHMETIC_ACTIVATIONS_BATCH_NORM: "Not relevant for OpenVINO.",
    HWFusedPatternNames.ARITHMETIC_ACTIVATIONS_SCALE_SHIFT: "Not relevant for OpenVINO.",
    HWFusedPatternNames.ARITHMETIC_BATCH_NORM: "Not relevant for OpenVINO.",
    HWFusedPatternNames.ARITHMETIC_BATCH_NORM_ACTIVATIONS: "Not relevant for OpenVINO.",
    HWFusedPatternNames.ARITHMETIC_SCALE_SHIFT: "Not relevant for OpenVINO.",
    HWFusedPatternNames.ARITHMETIC_SCALE_SHIFT_ACTIVATIONS: "Not relevant for OpenVINO.",
    HWFusedPatternNames.BATCH_NORM_SCALE_SHIFT_ACTIVATIONS: "Not relevant for OpenVINO.",
    HWFusedPatternNames.LINEAR_ACTIVATIONS_BATCH_NORM: "Not relevant for OpenVINO.",
    HWFusedPatternNames.LINEAR_ACTIVATIONS_SCALE_SHIFT: "Not relevant for OpenVINO.",
    HWFusedPatternNames.LINEAR_BATCH_NORM: "Not relevant for OpenVINO.",
    HWFusedPatternNames.LINEAR_BATCH_NORM_ACTIVATIONS: "Not relevant for OpenVINO.",
    HWFusedPatternNames.LINEAR_BATCH_NORM_SCALE_SHIFT_ACTIVATIONS: "Not relevant for OpenVINO.",
    HWFusedPatternNames.LINEAR_SCALE_SHIFT_ACTIVATIONS: "Not relevant for OpenVINO.",
    HWFusedPatternNames.MVN: "Not relevant for OpenVINO.",
    HWFusedPatternNames.GELU: "Not relevant for OpenVINO.",
}

IGNORING_IGNORED_PATTERN_REASONS = {}


def test_pattern_manager():
    check_hw_patterns(BackendType.OPENVINO, IGNORING_HW_PATTERN_REASONS)
    check_ignored_patterns(BackendType.OPENVINO, IGNORING_IGNORED_PATTERN_REASONS)
