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
from typing import Callable, Dict, Optional, Union, cast

from nncf.common.graph.patterns.patterns import GraphPattern
from nncf.common.graph.patterns.patterns import HWFusedPatternNames
from nncf.common.graph.patterns.patterns import IgnoredPatternNames
from nncf.common.graph.patterns.patterns import Patterns
from nncf.common.utils.backend import BackendType
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice

PatternNames = Union[IgnoredPatternNames, HWFusedPatternNames]


class PatternsManager:
    """
    Class provides interface to get hardware or ignored full patterns graph.
    """

    @staticmethod
    def _get_backend_hw_patterns_map(backend: BackendType) -> Dict[HWFusedPatternNames, Callable[[], GraphPattern]]:
        """
        Returns the backend-specific hardware-fused patterns map from the Registry.

        :param backend: BackendType instance.
        :return: Dictionary with the HWFusedPatternNames instance as keys and creator function as a value.
        """
        registry: Dict[HWFusedPatternNames, Callable[[], GraphPattern]] = {}
        if backend == BackendType.ONNX:
            from nncf.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS

            registry = cast(Dict[HWFusedPatternNames, Callable[[], GraphPattern]], ONNX_HW_FUSED_PATTERNS.registry_dict)
            return registry
        if backend == BackendType.OPENVINO:
            from nncf.openvino.hardware.fused_patterns import OPENVINO_HW_FUSED_PATTERNS

            registry = cast(
                Dict[HWFusedPatternNames, Callable[[], GraphPattern]], OPENVINO_HW_FUSED_PATTERNS.registry_dict
            )
            return registry
        if backend in (BackendType.TORCH, BackendType.TORCH_FX):
            from nncf.torch.hardware.fused_patterns import PT_HW_FUSED_PATTERNS

            registry = cast(Dict[HWFusedPatternNames, Callable[[], GraphPattern]], PT_HW_FUSED_PATTERNS.registry_dict)
            return registry
        raise ValueError(f"Hardware-fused patterns not implemented for {backend} backend.")

    @staticmethod
    def _get_backend_ignored_patterns_map(
        backend: BackendType,
    ) -> Dict[IgnoredPatternNames, Callable[[], GraphPattern]]:
        """
        Returns the backend-specific ignored patterns map from the Registry.

        :param backend: BackendType instance.
        :return: Dictionary with the HWFusedPatternNames instance as keys and creator function as a value.
        """
        registry: Dict[IgnoredPatternNames, Callable[[], GraphPattern]] = {}
        if backend == BackendType.ONNX:
            from nncf.onnx.quantization.ignored_patterns import ONNX_IGNORED_PATTERNS

            registry = cast(Dict[IgnoredPatternNames, Callable[[], GraphPattern]], ONNX_IGNORED_PATTERNS.registry_dict)
            return registry
        if backend == BackendType.OPENVINO:
            from nncf.openvino.quantization.ignored_patterns import OPENVINO_IGNORED_PATTERNS

            registry = cast(
                Dict[IgnoredPatternNames, Callable[[], GraphPattern]], OPENVINO_IGNORED_PATTERNS.registry_dict
            )
            return registry
        if backend in (BackendType.TORCH, BackendType.TORCH_FX):
            from nncf.torch.quantization.ignored_patterns import PT_IGNORED_PATTERNS

            registry = cast(Dict[IgnoredPatternNames, Callable[[], GraphPattern]], PT_IGNORED_PATTERNS.registry_dict)
            return registry
        raise ValueError(f"Ignored patterns not implemented for {backend} backend.")

    @staticmethod
    def _filter_patterns(
        patterns_to_filter: Dict[PatternNames, Callable[[], GraphPattern]],
        device: TargetDevice,
        model_type: Optional[ModelType] = None,
    ) -> Dict[PatternNames, Callable[[], GraphPattern]]:
        """
        Returns all patterns from patterns_to_filter that are satisfied device and model_type parameters.

        :param patterns_to_filter: Dictionary with the PatternNames instance as keys and creator function as a value.
        :param device: TargetDevice instance.
        :param model_type: ModelType instance.
        :return: Filtered patterns_to_filter.
        """
        filtered_patterns = {}
        for pattern_desc, pattern_creator in patterns_to_filter.items():
            pattern_desc_devices = pattern_desc.value.devices
            pattern_desc_model_types = pattern_desc.value.model_types
            devices_condition = pattern_desc_devices is None or device in pattern_desc_devices
            model_types_condition = pattern_desc_model_types is None or model_type in pattern_desc_model_types
            if devices_condition and model_types_condition:
                filtered_patterns[pattern_desc] = pattern_creator
        return filtered_patterns

    @staticmethod
    def _get_full_pattern_graph(
        backend_patterns_map: Dict[PatternNames, Callable[[], GraphPattern]],
        device: TargetDevice,
        model_type: Optional[ModelType] = None,
    ) -> GraphPattern:
        """
        Filters patterns and returns GraphPattern with registered filtered patterns.

        :param backend_patterns_map: Dictionary with the PatternNames instance as keys and creator function as a value.
        :param device: TargetDevice instance.
        :param model_type: ModelType instance.
        :return: Completed GraphPattern based on the backend, device & model_type.
        """
        filtered_patterns = PatternsManager._filter_patterns(backend_patterns_map, device, model_type)
        patterns = Patterns()
        for pattern_desc, pattern_creator in filtered_patterns.items():
            patterns.register(pattern_creator(), pattern_desc.value.name)
        return patterns.get_full_pattern_graph()

    @staticmethod
    def get_full_hw_pattern_graph(
        backend: BackendType, device: TargetDevice, model_type: Optional[ModelType] = None
    ) -> GraphPattern:
        """
        Returns a GraphPattern containing all registered hardware patterns specifically
        for backend, device, and model_type parameters.

        :param backend: BackendType instance.
        :param device: TargetDevice instance.
        :param model_type: ModelType instance.
        :return: Completed GraphPattern based on the backend, device & model_type.
        """
        backend_patterns_map = cast(
            Dict[PatternNames, Callable[[], GraphPattern]], PatternsManager._get_backend_hw_patterns_map(backend)
        )
        return PatternsManager._get_full_pattern_graph(backend_patterns_map, device, model_type)

    @staticmethod
    def get_full_ignored_pattern_graph(
        backend: BackendType, device: TargetDevice, model_type: Optional[ModelType] = None
    ) -> GraphPattern:
        """
        Returns a GraphPattern containing all registered ignored patterns specifically
        for backend, device, and model_type parameters.

        :param backend: BackendType instance.
        :param device: TargetDevice instance.
        :param model_type: ModelType instance.
        :return: Completed GraphPattern with registered value based on the backend, device & model_type.
        """
        backend_patterns_map = cast(
            Dict[PatternNames, Callable[[], GraphPattern]], PatternsManager._get_backend_ignored_patterns_map(backend)
        )
        return PatternsManager._get_full_pattern_graph(backend_patterns_map, device, model_type)
