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
from nncf.parameters import TargetDevice
from nncf.common.graph.patterns.patterns import HWFusedPatterns
from nncf.experimental.openvino_native.hardware.fused_patterns import OPENVINO_HW_FUSED_PATTERNS
from nncf.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS


class PatternsManager:

    BACKEND_TO_PATTERNS_MAP = {
        BackendType.ONNX: ONNX_HW_FUSED_PATTERNS,
        BackendType.OPENVINO: OPENVINO_HW_FUSED_PATTERNS
    }

    def get_backend_patterns(self, backend: BackendType):
        return self.BACKEND_TO_PATTERNS_MAP[backend].registry_dict

    def get_patterns(self, backend: BackendType, device: TargetDevice) -> HWFusedPatterns:
        backend_registry = self.BACKEND_TO_PATTERNS_MAP[backend]
        hw_fused_patterns = HWFusedPatterns()

        for pattern_desc, pattern in backend_registry.registry_dict.items():
            pattern_desc_devices = pattern_desc.value.devices
            if pattern() is None:
                continue
            if pattern_desc_devices is None or device in pattern_desc_devices:
                hw_fused_patterns.register(pattern(), pattern_desc.value.name)
        return hw_fused_patterns
