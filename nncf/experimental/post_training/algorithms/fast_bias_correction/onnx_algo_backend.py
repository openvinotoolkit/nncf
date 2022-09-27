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

from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import LAYERS_WITH_BIAS_METATYPES
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.experimental.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.onnx.statistics.collectors import ONNXMeanStatisticCollector
from nncf.experimental.onnx.tensor import ONNXNNCFTensor


class ONNXAlgoBackend:

    @property
    def operation_metatypes(self):
        return ONNX_OPERATION_METATYPES

    @property
    def layers_with_bias_metatypes(self):
        return LAYERS_WITH_BIAS_METATYPES

    @staticmethod
    def target_point(target_type, target_node_name, edge_name=None):
        return ONNXTargetPoint(target_type, target_node_name, edge_name)

    @staticmethod
    def bias_correction_command(target_point, bias_value, threshold):
        return ONNXBiasCorrectionCommand(target_point, bias_value, threshold)

    @staticmethod
    def model_extraction_command(inputs, outputs):
        return ONNXModelExtractionCommand(inputs, outputs)

    @staticmethod
    def mean_statistic_collector(reduction_shape, num_samples=None, window_size=None):
        return ONNXMeanStatisticCollector(reduction_shape,  num_samples, window_size)

    @staticmethod
    def nncf_tensor(tensor):
        return ONNXNNCFTensor(tensor)
