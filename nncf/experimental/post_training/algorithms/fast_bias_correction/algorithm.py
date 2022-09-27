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

from audioop import bias
from typing import Dict
from typing import List
from typing import TypeVar
from typing import Union

import numpy as np
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.backend import BackendType, get_backend
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.experimental.post_training.algorithms import AlgorithmParameters
from nncf.experimental.post_training.algorithms.algorithm import Algorithm
from nncf.experimental.post_training.algorithms.algorithm import PostTrainingAlgorithms
from nncf.experimental.post_training.algorithms.fast_bias_correction.onnx_algo_backend import ONNXAlgoBackend
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.factories import NNCFGraphFactory
from nncf.experimental.post_training.statistics.statistic_point import StatisticPoint
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer

ModelType = TypeVar('ModelType')

# pylint: disable=protected-access


class FastBiasCorrectionParameters(AlgorithmParameters):
    """
    Base class of FastBiasCorrection parameters.
    """

    def __init__(self, number_samples: int = 100, threshold: float = 2.0):
        self.number_samples = number_samples
        self.threshold = threshold

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        """
        Serialize all FastBiasCorrection parameters to JSON.
        """


# pylint: disable = too-many-function-args
class FastBiasCorrection(Algorithm):

    def __init__(self, parameters: FastBiasCorrectionParameters):
        super().__init__()
        self.number_samples = parameters.number_samples
        self.threshold = parameters.threshold
        self.nncf_graph = None
        self._target_points_to_correct = []  # type: List[TargetPoint]
        self._channel_axis_by_types = {
            'Conv': 1,
            'Gemm': -1,
        }
        self._backend_entity = None
    
    def _set_backend_entity(self, model):
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            self._backend_entity = ONNXAlgoBackend()
        else:
            raise RuntimeError('Cannot return backend-specific entity'
                               'because {} is not supported!'.format(model_backend))

    def _apply(self, model: ModelType, engine: Engine,
               statistic_points: StatisticPointsContainer) -> ModelType:

        self._set_backend_entity(model)

        transformation_layout = TransformationLayout()
        nncf_graph = NNCFGraphFactory.create(model)

        layers_with_bias_types = self._backend_entity.layers_with_bias_metatypes
        biased_nodes = nncf_graph.get_nodes_by_metatypes(layers_with_bias_types)

        for node in biased_nodes:
            node_name = node.node_name
            input_fp, input_shape = self._get_fp_inputs(statistic_points, node_name)
            output_fp = self._get_fp_outputs(statistic_points, node_name)

            input_names = []
            for dequantize_node in nncf_graph.get_previous_nodes(node):
                for quantize_node in nncf_graph.get_previous_nodes(dequantize_node):
                    # we uses the first (0) tensor position as data tensor
                    tensor_name = quantize_node.layer_attributes.input_tensor_names[0]
                    input_names.append(tensor_name)
            output_names = node.layer_attributes.output_tensor_names

            extracted_model = self._extract_submodel(input_names, output_names)

            channel_axis = self._channel_axis_by_types[node.node_type]
            stat_collector = self._backend_entity.mean_statistic_collector(reduction_shape=channel_axis,
                                                                           num_samples=self.number_samples)
            input_blob = self._create_input_blob(input_shape,
                                                 input_fp,
                                                 input_names)
            bias_shift = self._get_bias_shift(
                engine=engine,
                model=extracted_model,
                stat_collector=stat_collector,
                input_blob=input_blob,
                channel_axis=channel_axis,
                output_fp=output_fp,
                output_names=output_names)

            target_point = self._backend_entity.target_point(TargetType.LAYER,
                                                             node.node_name)
            bias_correction_command = self._backend_entity.bias_correction_command(target_point,
                                                                                   bias_shift,
                                                                                   self.threshold)
            transformation_layout.register(bias_correction_command)

        quantized_model = self._model_transformer.transform(transformation_layout)
        return quantized_model

    def _get_fp_inputs(self, statistic_points: StatisticPointsContainer, node_name: str):
        def input_filter_func(point):
            return PostTrainingAlgorithms.FastBiasCorrection in point.algorithm_to_tensor_collectors and \
                point.target_point.type == TargetType.PRE_LAYER_OPERATION

        input_fp = []
        input_shape = []
        for tensor_collector in statistic_points.iter_through_algorithm_tensor_collectors_in_target_node(
                node_name,
                input_filter_func,
                PostTrainingAlgorithms.FastBiasCorrection):
            input_fp.extend(tensor_collector.get_statistics().mean_values)
            input_shape.extend(tensor_collector.get_statistics().shape)
        return input_fp, input_shape

    def _get_fp_outputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> List:
        def output_filter_func(point):
            return PostTrainingAlgorithms.FastBiasCorrection in point.algorithm_to_tensor_collectors and \
                point.target_point.type == TargetType.POST_LAYER_OPERATION

        output_fp = []
        for tensor_collector in statistic_points.iter_through_algorithm_tensor_collectors_in_target_node(
                node_name,
                output_filter_func,
                PostTrainingAlgorithms.FastBiasCorrection):
            output_fp.extend(tensor_collector.get_statistics().mean_values)
        return output_fp

    def _extract_submodel(self, input_names, output_names):
        model_extraction_command = self._backend_entity.model_extraction_command(input_names,
                                                                                 output_names)
        me_transformation_layout = TransformationLayout()
        me_transformation_layout.register(model_extraction_command)
        extracted_model = self._model_transformer.transform(me_transformation_layout)
        return extracted_model

    def _add_statistic_point(self, container, point, axes):
        stat_collector = self._backend_entity.mean_statistic_collector(reduction_shape=axes,
                                                                       num_samples=self.number_samples)
        container.add_statistic_point(StatisticPoint(target_point=point,
                                                     tensor_collector=stat_collector,
                                                     algorithm=PostTrainingAlgorithms.FastBiasCorrection))

    def _create_input_blob(self, input_shape, input_fp, input_names):
        input_blob = np.zeros(input_shape)
        for i, value in enumerate(input_fp):
            input_blob[:, i] = value
        input_blob = input_blob.astype(np.float32)

        input_data = self._backend_entity.nncf_tensor(input_blob)
        input_data = {n: input_data for n in input_names}
        return input_data

    def _get_bias_shift(self,
                        engine,
                        model,
                        stat_collector,
                        input_blob,
                        channel_axis,
                        output_fp,
                        output_names):

        engine.rt_session_options['providers'] = ['OpenVINOExecutionProvider']
        engine.set_model(model)
        q_outputs = engine.infer(input_blob)
        engine.rt_session_options['providers'] = ['CPUExecutionProvider']
        q_outputs = q_outputs[output_names[0]]
        q_outputs = stat_collector._get_processor().mean_per_channel(q_outputs, channel_axis).tensor
        bias_shift = np.array(output_fp) - q_outputs
        return bias_shift

    def get_statistic_points(self, model: ModelType) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph
        layers_with_bias_types = self._backend_entity.layers_with_bias_metatypes
        biased_nodes = nncf_graph.get_nodes_by_metatypes(layers_with_bias_types)

        statistic_container = StatisticPointsContainer()

        for node in biased_nodes:
            edge_name = node.layer_attributes.input_tensor_names[0]
            pre_layer_statistic_point = self._backend_entity.target_point(TargetType.PRE_LAYER_OPERATION,
                                                                          node.node_name,
                                                                          edge_name)
            post_layer_statistic_point = self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION,
                                                                           node.node_name)
            channel_axis = self._channel_axis_by_types[node.node_type]

            self._add_statistic_point(statistic_container,
                                      pre_layer_statistic_point,
                                      channel_axis)
            self._add_statistic_point(statistic_container,
                                      post_layer_statistic_point,
                                      channel_axis)

        return statistic_container
