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

from typing import Dict, Tuple, List, TypeVar, Optional

import numpy as np
from nncf import Dataset
from nncf.common.tensor import NNCFTensor
from nncf.common.logging import nncf_logger
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.quantization.algorithms.algorithm import AlgorithmParameters
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS
from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import EngineFactory
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer

TModel = TypeVar('TModel')


class FastBiasCorrectionParameters(AlgorithmParameters):
    """
    Parameters of FastBiasCorrection algorithm

    :param number_samples: The number of the samples for the statistics collection.
    :param threshold: The magnitude threshold that regulates the application of the shift.
    """

    def __init__(self, number_samples: int = 100, threshold: float = 2.0) -> None:
        """
        :param number_samples: The number of the samples for the statistics collection.
            This statistics uses for the further calculation of the bias shift.
        :param threshold: The magnitude threshold that regulates the application of the shift.
            Magnitude calculates as the maximum of the absolute ratio of the shift to the original bias value.
            If the calculated value less than threshold, shift will apply to the bias.
        """
        self.number_samples = number_samples
        self.threshold = threshold


class FastBiasCorrection(Algorithm):
    """
    Post-training FastBiasCorrection algorithm implementation

    The main purpose of this algorithm to reduce quantization error
    via correction the bias of the Convolutions, FullyConnected, etc. layers.
    The algorithm pipeline is very simple:
        - we collects floating-point statistics from the corresponding model for the layers with bias;
        - then we gets the quantized model and try to reduce it's error by correction of the bias;
        - the shift calculates using the sub-graph that consists of the correction layer and
        weight quantizer-dequantizer pair or fake quantize node;
        - the floating-point statistics uses as input for
        the sub-graph and further quantization output calculation;
        - in the end we corrects the original bias by the difference (shift)
        between floating-point and quantized outputs.

    :param number_samples: The number of the samples for the statistics collection.
    :param threshold: The magnitude threshold that regulates the application of the shift.
    :param nncf_graph: NNCFGraph class for the algorithm.
    """

    def __init__(self, parameters: FastBiasCorrectionParameters) -> None:
        """
        :param parameters: The instance of the FastBiasCorrectionParameters.
        """
        super().__init__()
        self.number_samples = parameters.number_samples
        self.threshold = parameters.threshold
        self.nncf_graph = None
        self._backend_entity = None

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return ALGO_BACKENDS.registry_dict

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm

        :param model: backend-specific input model
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.fast_bias_correction.onnx_backend import \
                ONNXFBCAlgoBackend
            self._backend_entity = ONNXFBCAlgoBackend()
        else:
            raise RuntimeError('Cannot return backend-specific entity'
                               'because {} is not supported!'.format(model_backend))

    def _apply(self,
               model: TModel,
               statistic_points: Optional[StatisticPointsContainer] = None,
               dataset: Optional[Dataset] = None) -> TModel:
        self._set_backend_entity(model)

        model_transformer = self._backend_entity.model_transformer(model)
        transformation_layout = TransformationLayout()
        nncf_graph = NNCFGraphFactory.create(model)

        layers_with_bias_types = self._backend_entity.layers_with_bias_metatypes
        biased_nodes = nncf_graph.get_nodes_by_metatypes(layers_with_bias_types)

        for node in biased_nodes:
            node_name = node.node_name

            if not self._backend_entity.is_node_with_bias(node):
                nncf_logger.debug(f'Skipping node {node_name} because there is no bias')
                continue
            if not self._backend_entity.is_quantized_weights(node, model):
                nncf_logger.debug(f'Skipping node {node_name} because weights were not quantized')
                continue

            input_fp, input_shape = self._get_fp_inputs(statistic_points, node_name)
            output_fp = self._get_fp_outputs(statistic_points, node_name)

            input_tensor_names, output_tensor_names = self._backend_entity.get_tensor_names(node)
            input_name = input_tensor_names[0]
            output_name = output_tensor_names[0]

            extracted_model = self._extract_submodel(model, [input_name], [output_name])

            channel_axis = self._backend_entity.channel_axis_by_types[node.node_type]
            input_blob = self._create_input_data(input_shape,
                                                 input_fp,
                                                 input_name)
            bias_shift = self._get_bias_shift(
                model=extracted_model,
                input_blob=input_blob,
                channel_axis=channel_axis,
                output_fp=output_fp,
                output_name=output_name)

            current_bias = self._backend_entity.get_bias_value(model, node)
            updated_bias = current_bias + bias_shift
            magnitude = self._get_bias_shift_magnitude(current_bias, updated_bias)

            if magnitude < self.threshold:
                nncf_logger.debug(f'{node_name} bias would be changed')
                bias_port_id = self._backend_entity.get_bias_port_id(model, node)
                target_point = self._backend_entity.target_point(TargetType.LAYER,
                                                                 node.node_name,
                                                                 bias_port_id)
                bias_correction_command = self._backend_entity.bias_correction_command(target_point,
                                                                                       updated_bias)
                transformation_layout.register(bias_correction_command)
            else:
                nncf_logger.debug(f'{node_name} bias skipped by threshold')

        quantized_model = model_transformer.transform(transformation_layout)
        return quantized_model

    def _get_fp_inputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> Tuple[List, List]:
        """
        Makes out per-layer needed data from the floating-point collected statistics

        :param statistic_points: filled StatisticPointsContainer
        :param node_name: name of the current layer
        :return: collected mean tensor data and shape for the further bias calculation
        """

        def input_filter_func(point):
            return FastBiasCorrection in point.algorithm_to_tensor_collectors and \
                   point.target_point.type == TargetType.PRE_LAYER_OPERATION

        input_fp = []
        input_shape = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
                node_name,
                input_filter_func,
                FastBiasCorrection):
            input_fp.extend(tensor_collector.get_statistics().mean_values)
            input_shape.extend(tensor_collector.get_statistics().shape)
        return input_fp, input_shape

    def _get_fp_outputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> List[np.ndarray]:
        """
        Makes out per-layer needed data from the floating-point collected statistics

        :param statistic_points: filled StatisticPointsContainer
        :param node_name: name of the current layer
        :return: collected mean tensor data for the further bias calculation
        """

        def output_filter_func(point):
            return FastBiasCorrection in point.algorithm_to_tensor_collectors and \
                   point.target_point.type == TargetType.POST_LAYER_OPERATION

        output_fp = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
                node_name,
                output_filter_func,
                FastBiasCorrection):
            output_fp.extend(tensor_collector.get_statistics().mean_values)
        return output_fp

    def _extract_submodel(self,
                          model: TModel,
                          input_names: List[str],
                          output_names: List[str]) -> TModel:
        """
        Extracts sub-model from the original based on the input & output tensor names

        :param model: backend-specific model
        :param input_names: list of the input names
        :param output_names: list of the output names
        :return: backend-specific sub-model
        """
        model_transformer = self._backend_entity.model_transformer(model)
        model_extraction_command = self._backend_entity.model_extraction_command(input_names,
                                                                                 output_names)
        me_transformation_layout = TransformationLayout()
        me_transformation_layout.register(model_extraction_command)
        extracted_model = model_transformer.transform(me_transformation_layout)
        return extracted_model

    def _add_statistic_point(self, container: StatisticPointsContainer, point: TargetPoint, axis: int) -> None:
        """
        Adds specific statistic point

        :param container: StatisticPointsContainer
        :param point: TargetPoint for statistic collection
        :param axis: channel axis for the statistics calculation
        """
        stat_collector = self._backend_entity.mean_statistic_collector(reduction_shape=axis,
                                                                       num_samples=self.number_samples)
        container.add_statistic_point(StatisticPoint(target_point=point,
                                                     tensor_collector=stat_collector,
                                                     algorithm=FastBiasCorrection))

    def _create_input_data(self,
                           input_shape: Tuple[int],
                           input_fp: List[np.ndarray],
                           input_name: str) -> Dict[str, NNCFTensor]:
        """
        Creates input blob for the bias shift calculation

        :param input_shape: input shape for the blob
        :param input_fp: input data for the blob
        :param input_name: name for the output dict
        :return: dictionary of the blob by input name
        """
        input_blob = self._backend_entity.create_blob(input_shape, input_fp)
        input_data = {input_name: input_blob}
        return input_data

    def _get_bias_shift(self,
                        model: TModel,
                        input_blob: Dict[str, NNCFTensor],
                        channel_axis: Tuple[int],
                        output_fp: List[np.ndarray],
                        output_name: str) -> np.ndarray:
        """
        Calculates bias shift for the further corretion

        :param engine: backend-specific engine instance for the model execution
        :param model: backend-specific sub-model for the execution
        :param input_blob: input data for the execution
        :param channel_axis: channel axes for the raw data aggregation
        :param output_fp: output data for the shift calculation
        :param output_name: name of the output tensor for the data collection
        :return: calculated bias shift
        """
        engine = EngineFactory.create(model)
        raw_output = engine.infer(input_blob)
        q_outputs = self._backend_entity.process_model_output(raw_output, output_name)
        q_outputs = self._backend_entity.tensor_processor.mean_per_channel(q_outputs, channel_axis).tensor
        bias_shift = np.array(output_fp) - q_outputs
        return bias_shift

    @staticmethod
    def _get_bias_shift_magnitude(current_bias_value: np.ndarray, updated_bias_value: np.ndarray) -> float:
        """
        Calculates bias shift magnitude based on the current and updated values

        :param current_bias_value: the original bias value
        :param updated_bias_value: the updated bias value
        :return: magnitude between original and updated bias values
        """
        bias_shift_magnitude = np.inf
        if np.count_nonzero(current_bias_value == 0) == 0:
            bias_shift_magnitude = np.max(np.abs((updated_bias_value - current_bias_value) / current_bias_value))
        return bias_shift_magnitude

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph
        layers_with_bias_types = self._backend_entity.layers_with_bias_metatypes
        biased_nodes = nncf_graph.get_nodes_by_metatypes(layers_with_bias_types)

        statistic_container = StatisticPointsContainer()

        for node in biased_nodes:
            if not self._backend_entity.is_node_with_bias(node):
                continue
            input_port_id, output_port_id = self._backend_entity.get_activation_port_ids_for_bias_node(model, node)
            pre_layer_statistic_point = self._backend_entity.target_point(TargetType.PRE_LAYER_OPERATION,
                                                                          node.node_name,
                                                                          input_port_id)
            post_layer_statistic_point = self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION,
                                                                           node.node_name,
                                                                           output_port_id)
            channel_axis = self._backend_entity.channel_axis_by_types[node.node_type]

            self._add_statistic_point(statistic_container,
                                      pre_layer_statistic_point,
                                      channel_axis)
            self._add_statistic_point(statistic_container,
                                      post_layer_statistic_point,
                                      channel_axis)

        return statistic_container
