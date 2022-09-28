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

from typing import List
import numpy as np
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.utils.registry import Registry

from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import LAYERS_WITH_BIAS_METATYPES
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.experimental.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.onnx.statistics.collectors import ONNXMeanStatisticCollector
from nncf.experimental.onnx.tensor import ONNXNNCFTensor
from nncf.experimental.post_training.algorithms.fast_bias_correction.algo_backend import ALGO_BACKENDS
from nncf.experimental.post_training.algorithms.fast_bias_correction.algo_backend import FBCAlgoBackend


@ALGO_BACKENDS.register()
class ONNXFBCAlgoBackend(FBCAlgoBackend):

    BACKEND_TYPE = BackendType.ONNX

    @property
    def operation_metatypes(self) -> Registry:
        return ONNX_OPERATION_METATYPES

    @property
    def layers_with_bias_metatypes(self) -> Registry:
        return LAYERS_WITH_BIAS_METATYPES

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str = None, edge_name: str = None) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, edge_name)

    @staticmethod
    def bias_correction_command(target_point: ONNXTargetPoint,
                                bias_value: np.ndarray,
                                threshold: float) -> ONNXBiasCorrectionCommand:
        return ONNXBiasCorrectionCommand(target_point, bias_value, threshold)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> ONNXModelExtractionCommand:
        return ONNXModelExtractionCommand(inputs, outputs)

    @staticmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: int = None,
                                 window_size: int = None) -> ONNXMeanStatisticCollector:
        return ONNXMeanStatisticCollector(reduction_shape,  num_samples, window_size)

    @staticmethod
    def nncf_tensor(tensor: np.ndarray) -> ONNXNNCFTensor:
        return ONNXNNCFTensor(tensor)
