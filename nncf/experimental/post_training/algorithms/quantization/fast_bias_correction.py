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

from typing import Dict
from typing import List
from typing import TypeVar
from typing import Union

import numpy as np
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.backend import infer_backend_from_model
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import LAYERS_WITH_BIAS_METATYPES
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.onnx.tensor import ONNXNNCFTensor
from nncf.experimental.post_training.algorithms import AlgorithmParameters
from nncf.experimental.post_training.algorithms.algorithm import Algorithm
from nncf.experimental.post_training.algorithms.algorithm import PostTrainingAlgorithms
from nncf.experimental.post_training.graph.factories import NNCFGraphFactory, PTQModelExtractionCommandFactory
from nncf.experimental.post_training.graph.factories import PTQBiasCorrectionCommandFactory
from nncf.experimental.post_training.graph.factories import PTQTargetPointFactory
from nncf.experimental.post_training.graph.factories import PTQMeanStatisticCollectorFactory
from nncf.experimental.post_training.statistics.statistic_point import StatisticPoint
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer

ModelType = TypeVar('ModelType')

# pylint: disable=protected-access


class FastBiasCorrectionParameters(AlgorithmParameters):
    """
    Base class of FastBiasCorrection parameters.
    """

    def __init__(self, number_samples: int = 100):
        self.number_samples = number_samples

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        """
        Serialize all FastBiasCorrection parameters to JSON.
        """


class FastBiasCorrection(Algorithm):

    def __init__(self, parameters: FastBiasCorrectionParameters):
        super().__init__()
        self.number_samples = parameters.number_samples
        self.nncf_graph = None
        self._target_points_to_correct = []  # type: List[ONNXTargetPoint]
        self._channel_axis_by_types = {
            'Conv': 1,
            'Gemm': -1,
        }

    def _generate_stat_collector(self, model_backend, axes) -> TensorStatisticCollectorBase:
        return PTQMeanStatisticCollectorFactory.create(
            model_backend,
            reduction_shape=axes,
            num_samples=self.number_samples)

    @staticmethod
    def _input_filter_func(point):
        return PostTrainingAlgorithms.FastBiasCorrection in point.algorithm_to_tensor_collectors and \
            point.target_point.type == TargetType.PRE_LAYER_OPERATION

    @staticmethod
    def _output_filter_func(point):
        return PostTrainingAlgorithms.FastBiasCorrection in point.algorithm_to_tensor_collectors and \
            point.target_point.type == TargetType.POST_LAYER_OPERATION

    def _apply(self, model: ModelType, engine: ONNXEngine,
               statistic_points: StatisticPointsContainer) -> ModelType:

        model_backend = infer_backend_from_model(model)

        transformation_layout = TransformationLayout()
        nncf_graph = NNCFGraphFactory.create(model)

        biased_nodes = nncf_graph.get_nodes_by_metatypes(
            LAYERS_WITH_BIAS_METATYPES)

        for node in biased_nodes:
            node_name = node.node_name
            input_fp = []
            input_shape = []
            output_fp = []

            for tensor_collector in statistic_points.iter_through_algorithm_tensor_collectors_in_target_node(
                    node_name,
                    self._input_filter_func,
                    PostTrainingAlgorithms.FastBiasCorrection):
                input_fp.extend(tensor_collector.get_statistics().mean_values)
                input_shape.extend(tensor_collector.get_statistics().shape)
            for tensor_collector in statistic_points.iter_through_algorithm_tensor_collectors_in_target_node(
                    node_name,
                    self._output_filter_func,
                    PostTrainingAlgorithms.FastBiasCorrection):
                output_fp.extend(tensor_collector.get_statistics().mean_values)

            input_names = []
            for dequantize_node in nncf_graph.get_previous_nodes(node):
                for quantize_node in nncf_graph.get_previous_nodes(dequantize_node):
                    input_position = 0  # input tensor position
                    input_names.append(
                        quantize_node.layer_attributes.input_tensor_names[input_position])
            output_names = node.layer_attributes.output_tensor_names

            model_extraction_command = PTQModelExtractionCommandFactory.create(
                model_backend, input_names, output_names)
            me_transformation_layout = TransformationLayout()
            me_transformation_layout.register(model_extraction_command)
            extracted_model = self._model_transformer.transform(
                me_transformation_layout)

            # TODO: Optimize _calculate_bias_shift method signature
            channel_axis = self._channel_axis_by_types[node.node_type]
            stat_collector = PTQMeanStatisticCollectorFactory.create(
                model_backend,
                reduction_shape=channel_axis,
                num_samples=self.number_samples)
            bias_shift = self._calculate_bias_shift(
                engine,
                extracted_model,
                stat_collector,
                channel_axis,
                input_shape,
                input_fp,
                output_fp,
                input_names,
                output_names)

            target_point = PTQTargetPointFactory.create(
                model_backend, TargetType.LAYER, node.node_name)
            bias_correction_command = PTQBiasCorrectionCommandFactory.create(
                model_backend, target_point, bias_shift)
            transformation_layout.register(bias_correction_command)

        quantized_model = self._model_transformer.transform(
            transformation_layout)
        return quantized_model

    def get_statistic_points(self, model: ModelType) -> StatisticPointsContainer:
        model_backend = infer_backend_from_model(model)
        nncf_graph = NNCFGraphFactory.create(
            model) if self.nncf_graph is None else self.nncf_graph
        biased_nodes = nncf_graph.get_nodes_by_metatypes(
            LAYERS_WITH_BIAS_METATYPES)

        statistic_container = StatisticPointsContainer()

        for node in biased_nodes:
            edge_name = node.layer_attributes.input_tensor_names[0]
            if edge_name == '463':
                print('s')
            pre_layer_statistic_point = PTQTargetPointFactory.create(
                model_backend, TargetType.PRE_LAYER_OPERATION, node.node_name, edge_name)
            post_layer_statistic_point = PTQTargetPointFactory.create(
                model_backend, TargetType.POST_LAYER_OPERATION, node.node_name)
            channel_axis = self._channel_axis_by_types[node.node_type]

            self._add_statistic_point(
                model_backend, statistic_container, pre_layer_statistic_point, channel_axis)
            self._add_statistic_point(
                model_backend, statistic_container, post_layer_statistic_point, channel_axis)

        return statistic_container

    def _add_statistic_point(self, model_backend, container, point, axes):
        stat_collector = PTQMeanStatisticCollectorFactory.create(
            model_backend,
            reduction_shape=axes,
            num_samples=self.number_samples)
        container.add_statistic_point(StatisticPoint(target_point=point,
                                                     tensor_collector=stat_collector,
                                                     algorithm=PostTrainingAlgorithms.FastBiasCorrection))

    def _create_input_blob(self, input_shape, input_fp, input_names):
        input_blob = np.zeros(input_shape)
        for i, value in enumerate(input_fp):
            input_blob[:, i] = value
        input_blob = input_blob.astype(np.float32)

        input_data = ONNXNNCFTensor(input_blob)
        input_data = {n: input_data for n in input_names}
        return input_data

    def _calculate_bias_shift(self,
                              engine,
                              model,
                              stat_collector,
                              channel_axis,
                              input_shape,
                              input_fp,
                              output_fp,
                              input_names,
                              output_names):
        input_blob = self._create_input_blob(
            input_shape, input_fp, input_names)

        engine.rt_session_options['providers'] = ['OpenVINOExecutionProvider']
        engine.set_model(model)
        q_outputs = engine.infer(input_blob)
        engine.rt_session_options['providers'] = ['CPUExecutionProvider']
        q_outputs = q_outputs[output_names[0]]
        q_outputs = stat_collector._get_processor().mean_per_channel(
            q_outputs, channel_axis).tensor
        bias_shift = np.array(output_fp) - q_outputs
        return bias_shift
