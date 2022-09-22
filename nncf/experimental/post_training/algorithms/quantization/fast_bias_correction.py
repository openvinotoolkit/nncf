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
import onnx
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.backend import infer_backend_from_model
from nncf.experimental.onnx.engine import ONNXEngine
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import LAYERS_WITH_BIAS_METATYPES
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.onnx.tensor import ONNXNNCFTensor
from nncf.experimental.post_training.algorithms import AlgorithmParameters
from nncf.experimental.post_training.algorithms.algorithm import Algorithm
from nncf.experimental.post_training.algorithms.algorithm import PostTrainingAlgorithms
from nncf.experimental.post_training.graph.factories import NNCFGraphFactory
from nncf.experimental.post_training.graph.factories import PTQBiasCorrectionCommandFactory
from nncf.experimental.post_training.graph.factories import PTQTargetPointFactory
from nncf.experimental.post_training.graph.factories import PTQTransformationLayoutFactory
from nncf.experimental.post_training.graph.factories import PTQMeanStatisticCollectorFactory
from nncf.experimental.post_training.statistics.statistic_point import StatisticPoint
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer

ModelType = TypeVar('ModelType')


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
        self._axes = (0, 2, 3)

    def generate_stat_collector(self, model_backend) -> TensorStatisticCollectorBase:
        return PTQMeanStatisticCollectorFactory.create(model_backend, reduction_shape=self._axes, num_samples=self.number_samples)

    @staticmethod
    def _input_filter_func(point):
        return PostTrainingAlgorithms.FastBiasCorrection in point.algorithm_to_tensor_collectors and \
            point.target_point.type == TargetType.PRE_LAYER_OPERATION

    @staticmethod
    def _output_filter_func(point):
        return PostTrainingAlgorithms.FastBiasCorrection in point.algorithm_to_tensor_collectors and \
            point.target_point.type == TargetType.POST_LAYER_OPERATION

    def _apply(self, model: onnx.ModelProto, engine: ONNXEngine,
               statistic_points: StatisticPointsContainer) -> onnx.ModelProto:

        model_backend = infer_backend_from_model(model)

        transformation_layout = PTQTransformationLayoutFactory.create(model_backend)
        transformation_commands = []
        nncf_graph = NNCFGraphFactory.create(model)

        biased_nodes = self.get_nodes_to_correct(model)

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
            output_names = []

            for dequantize_node in nncf_graph.get_previous_nodes(node):
                for quantize_node in nncf_graph.get_previous_nodes(dequantize_node):
                    input_position = 0 # input tensor position
                    input_names.append(quantize_node.layer_attributes.input_tensor_names[input_position])
            output_names = node.layer_attributes.output_tensor_names

            extracted_model = self._model_transformer.extract_model_by_inputs_outputs(model, input_names, output_names)
            # TODO: Optimize calculate_bias_shift method signature
            bias_shift = self.calculate_bias_shift(engine, extracted_model, input_shape, input_fp, output_fp, input_names, output_names)

            target_point = PTQTargetPointFactory.create(model_backend, TargetType.LAYER, node.node_name)
            command = PTQBiasCorrectionCommandFactory.create(model_backend, target_point, bias_shift)
            transformation_commands.append(command)

        for transformation_command in transformation_commands:
            transformation_layout.register(transformation_command)

        quantized_model = self._model_transformer.transform(transformation_layout)
        return quantized_model

    def get_statistic_points(self, model: onnx.ModelProto) -> StatisticPointsContainer:
        model_backend = infer_backend_from_model(model)
        biased_nodes = self.get_nodes_to_correct(model)

        statistic_points = []

        for node in biased_nodes:
            edge_name = node.layer_attributes.input_tensor_names[0]
            pre_layer_statistic_point = PTQTargetPointFactory.create(
                model_backend, TargetType.PRE_LAYER_OPERATION, node.node_name, edge_name)
            post_layer_statistic_point = PTQTargetPointFactory.create(
                model_backend, TargetType.POST_LAYER_OPERATION, node.node_name)
            statistic_points.extend(
                [pre_layer_statistic_point, post_layer_statistic_point])

        output = StatisticPointsContainer()
        for target_point in statistic_points:
            stat_collector = self.generate_stat_collector(model_backend)
            output.add_statistic_point(StatisticPoint(target_point=target_point,
                                                      tensor_collector=stat_collector,
                                                      algorithm=PostTrainingAlgorithms.FastBiasCorrection))
        return output

    def get_nodes_to_correct(self, model: onnx.ModelProto) -> StatisticPointsContainer:
        nncf_graph = NNCFGraphFactory.create(
            model) if self.nncf_graph is None else self.nncf_graph
        return nncf_graph.get_nodes_by_metatypes(LAYERS_WITH_BIAS_METATYPES)

    def create_input_blob(self, input_shape, input_fp, input_names):
        input_blob = np.zeros(input_shape)
        # axis = 1
        # input_blob = np.moveaxis(input_blob, axis, 1)
        for i, value in enumerate(input_fp):
            input_blob[:, i] = value
        # input_blob = np.moveaxis(input_blob, 1, axis)
        input_blob = input_blob.astype(np.float32)

        input_data = ONNXNNCFTensor(input_blob)
        input_data = {n: input_data for n in input_names}
        return input_data
    
    def calculate_bias_shift(self,
                             engine,
                             model,
                             input_shape,
                             input_fp,
                             output_fp,
                             input_names,
                             output_names):
        input_blob = self.create_input_blob(input_shape, input_fp, input_names)

        engine.rt_session_options['providers'] = ['OpenVINOExecutionProvider']
        engine.set_model(model)
        q_outputs = engine.infer(input_blob)
        engine.rt_session_options['providers'] = ['CPUExecutionProvider']
        q_outputs = np.mean(q_outputs[output_names[0]].tensor, axis=self._axes)
        bias_shift = np.array(output_fp) - q_outputs
        return bias_shift
