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
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.collectors import MeanStatisticCollector
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import infer_backend_from_model
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.onnx.graph.transformations.layout import ONNXTransformationLayout
from nncf.experimental.onnx.statistics.collectors import ONNXMeanStatisticCollector


class BackendGraphFactory:
    @staticmethod
    def create(model):
        model_backend = infer_backend_from_model(model)
        if model_backend == BackendType.ONNX:
            return ONNXGraph(model)
        raise RuntimeError('Cannot create backend-specific graph'
                           'because {0} is not supported!'.format(model_backend))


class NNCFGraphFactory:
    @staticmethod
    def create(model):
        model_backend = infer_backend_from_model(model)
        if model_backend == BackendType.ONNX:
            return GraphConverter.create_nncf_graph(model)
        raise RuntimeError('Cannot create backend-specific graph'
                           'because {0} is not supported!'.format(model_backend))


class PTQTransformationLayoutFactory:
    @staticmethod
    def create(backend) -> TransformationLayout:
        if backend == BackendType.ONNX:
            transformation_layout = ONNXTransformationLayout()
        return transformation_layout


class PTQTargetPointFactory:
    @staticmethod
    def create(backend, target_type, target_node_name, edge_name=None) -> TargetPoint:
        if backend == BackendType.ONNX:
            target_point = ONNXTargetPoint(
                target_type, target_node_name, edge_name)
        return target_point


class PTQBiasCorrectionCommandFactory:
    @staticmethod
    def create(backend, target_point, bias_value) -> TargetPoint:
        if backend == BackendType.ONNX:
            bias_correction_command = ONNXBiasCorrectionCommand(
                target_point, bias_value)
        return bias_correction_command


class PTQMeanStatisticCollectorFactory:
    @staticmethod
    def create(backend, reduction_shape, num_samples) -> MeanStatisticCollector:
        if backend == BackendType.ONNX:
            mean_stats_collector = ONNXMeanStatisticCollector(
                reduction_shape=reduction_shape, num_samples=num_samples)
        return mean_stats_collector
