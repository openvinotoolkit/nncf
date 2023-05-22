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
from typing import Callable, Dict, Optional

from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.patterns.patterns import PatternNames
from nncf.common.graph.patterns.patterns import Patterns
from nncf.common.utils.backend import BackendType
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice


class PatternsManager:
    """
    The main purpose of the class is to return the backend- & device-specific patterns.
    """

    @staticmethod
    def get_backend_hw_patterns_map(backend: BackendType) -> Dict[PatternNames, Callable]:
        """
        Returns the backend-specific map from the Registry.

        :param backend: BackendType instance.
        :return: Dictionary with the PatternNames instance as keys and callable as value.
        """
        if backend == BackendType.ONNX:
            from nncf.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS

            return ONNX_HW_FUSED_PATTERNS.registry_dict
        if backend == BackendType.OPENVINO:
            from nncf.openvino.hardware.fused_patterns import OPENVINO_HW_FUSED_PATTERNS

            return OPENVINO_HW_FUSED_PATTERNS.registry_dict
        if backend == BackendType.TORCH:
            from nncf.torch.hardware.fused_patterns import PT_HW_FUSED_PATTERNS

            return PT_HW_FUSED_PATTERNS.registry_dict
        raise ValueError(f"Hardware-fused patterns not implemented for {backend} backend.")

    @staticmethod
    def get_backend_ignored_patterns_map(backend: BackendType) -> Dict[PatternNames, Callable]:
        if backend == BackendType.OPENVINO:
            from nncf.openvino.quantization.ignored_patterns import OPENVINO_IGNORED_PATTERNS

            return OPENVINO_IGNORED_PATTERNS.registry_dict

    @staticmethod
    def _get_patterns(pattern_desc_to_graph, device, model_type):
        patterns = Patterns()
        for pattern_desc, pattern in pattern_desc_to_graph.items():
            pattern_desc_devices = pattern_desc.value.devices
            pattern_desc_model_types = pattern_desc.value.model_types
            devices_condition = pattern_desc_devices is None or device in pattern_desc_devices
            model_types_condition = pattern_desc_model_types is None or model_type in pattern_desc_model_types
            if devices_condition and model_types_condition:
                patterns.register(pattern(), pattern_desc.value.name)
        return patterns

    @staticmethod
    def get_full_hw_pattern_graph(
        backend: BackendType, device: TargetDevice, model_type: Optional[ModelType] = None
    ) -> GraphPattern:
        """
        Returns the backend-, device- & model_type-specific Pattern instance.

        :param backend: BackendType instance.
        :param device: TargetDevice instance.
        :param model_type: ModelType instance.
        :return: Completed GraphPattern value based on the backend, device & model_type.
        """
        backend_registry_map = PatternsManager.get_backend_hw_patterns_map(backend)
        hw_fused_patterns = PatternsManager._get_patterns(backend_registry_map, device, model_type)
        return hw_fused_patterns.get_full_pattern_graph()

    @staticmethod
    def get_full_ignored_pattern_graph(
        backend: BackendType, device: TargetDevice, model_type: Optional[ModelType] = None
    ) -> GraphPattern:
        backend_registry_map = PatternsManager.get_backend_ignored_patterns_map(backend)
        ignored_patterns = PatternsManager._get_patterns(backend_registry_map, device, model_type)
        return ignored_patterns.get_full_pattern_graph()
