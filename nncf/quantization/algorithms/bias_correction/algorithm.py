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

from typing import Dict, List, TypeVar, Union, Optional, Tuple

import numpy as np
from copy import deepcopy
from collections import deque

from functools import partial
from nncf import Dataset
from nncf import nncf_logger
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.algorithm import AlgorithmParameters
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.bias_correction.backend import ALGO_BACKENDS
from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import EngineFactory
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer

TModel = TypeVar('TModel')


class BiasCorrectionParameters(AlgorithmParameters):
    """
    Parameters of BiasCorrection algorithm

    :param number_samples: Number of samples for statistics collection.
    :param threshold: Magnitude threshold that regulates application of shift.
    """

    def __init__(self, number_samples: int = 100, threshold: float = 1000) -> None:
        """
        :param number_samples: The number of samples for the statistics collection.
            This statistic uses for the further calculation of the bias shift.
        :param threshold: The magnitude threshold regulates the application of the shift.
            Magnitude calculates as the maximum of the absolute ratio of the shift to the original bias value.
            If the calculated value is less than the threshold, the shift will apply to the bias.
        """
        self.number_samples = number_samples
        self.threshold = threshold

    def to_json(self) -> Dict[str, Union[str, float, int]]:
        """
        Serialize all BiasCorrection parameters to JSON.
        """


class BiasCorrection(Algorithm):

    """
    Post-training BiasCorrection algorithm implementation

    The main purpose of this algorithm to reduce quantization error
    via correction the bias of the Convolutions, FullyConnected, etc. layers.
    The algorithm's pipeline looks like this:
        - we collect floating-point statistics from the first layers with the bias in the model;
        - then we get the quantized model and collect the necessary information to create a sub-graph;
        - after the information collection, we drop the quantizer-dequantizer pair or fake quantize node on activations;
        - using collected information, we try to create the model sub-graph dynamically:
        this sub-graph contains the layer in which bias would be corrected and the other layers that need to
        collect the new statistics for the next layer with the bias;
        - the shift calculates using the sub-graph that consists of the correction layer and
        weight quantizer-dequantizer pair or fake quantize node, and some other layers;
        - then we correct the original bias by the difference (shift) between floating-point and quantized outputs in
        the sub-graph and the model without quantizer-dequantizer pair or fake quantize node.
        - at the next step, we collect the new statistics for the next layer (that would be corrected) from
        the sub-graph with the updated bias value on the current step;
        - after the new statistics were collected, we drops the unnecessary statistics to reduce memory consumption;
        - in the end, we correct all needed biases in the original model.

    :param number_samples: The number of the samples for the statistics collection.
    :param threshold: The magnitude threshold that regulates the application of the shift.
    :param nncf_graph: NNCFGraph class for the algorithm.
    """

    def __init__(self, parameters: BiasCorrectionParameters) -> None:
        """
        :param parameters: The instance of the BiasCorrectionParameters.
        """
        super().__init__()
        self.number_samples = max(np.int(parameters.number_samples * 0.2), 1)
        self.threshold = parameters.threshold
        self.nncf_graph = None
        self._backend_entity = None
        self._collected_stat_inputs = set()
        self._fp_inputs = {}

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return ALGO_BACKENDS.registry_dict

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.ONNX:
            from nncf.quantization.algorithms.bias_correction.onnx_backend import \
                ONNXBiasCorrectionAlgoBackend
            self._backend_entity = ONNXBiasCorrectionAlgoBackend()
        else:
            raise RuntimeError('Cannot return backend-specific entity'
                               'because {} is not supported!'.format(model_backend))

    def _apply(self,
               model: TModel,
               statistic_points: Optional[StatisticPointsContainer] = None,
               dataset: Optional[Dataset] = None) -> TModel:

        self._set_backend_entity(model)
        main_transformations_layout = TransformationLayout()
        main_model_transformer = self._backend_entity.model_transformer(model)

        model_copy = deepcopy(model)
        model_copy = self._remove_fq_from_inputs(model_copy)
        nncf_graph = NNCFGraphFactory.create(model_copy)
        subgraphs_data = self._fill_subgraphs_data(nncf_graph)

        for node_name, subgraph_data in subgraphs_data.items():
            node = nncf_graph.get_node_by_name(node_name)

            if not self._backend_entity.is_node_with_bias(node):
                nncf_logger.debug(f'Skipping node {node_name} because there is no bias')
                continue
            if not self._backend_entity.is_quantized_weights(node, model):
                nncf_logger.debug(f'Skipping node {node_name} because weights was not quantized')
                continue

            # We do not make an additional copy of the model because
            # the model transformer (that uses during sub-graph extraction) already does this internally when creating.
            model_copy_subgraph = self._prepare_subgraph(node, model_copy, subgraph_data)

            feed_dicts = self._create_feed_dicts(nncf_graph, subgraph_data, statistic_points)

            bias_shift = self._compute_bias_shift(node, model_copy_subgraph, feed_dicts, statistic_points)

            current_bias = self._backend_entity.get_bias_value(model, node)
            updated_bias = current_bias + bias_shift
            magnitude = self._get_bias_shift_magnitude(current_bias, updated_bias)

            if magnitude < self.threshold:
                nncf_logger.debug(f'{node_name} bias would be changed. Magnitude: {magnitude}')
                bias_port_id = self._backend_entity.get_bias_port_id(model, node)
                target_point = self._backend_entity.target_point(TargetType.LAYER,
                                                                 node.node_name,
                                                                 bias_port_id)
                bias_correction_command = self._backend_entity.bias_correction_command(target_point,
                                                                                       updated_bias)
                model_copy_subgraph = self._correct_bias(model_copy_subgraph, bias_correction_command)
                model_copy = self._correct_bias(model_copy, bias_correction_command)
                main_transformations_layout.register(bias_correction_command)
            else:
                nncf_logger.debug(f'{node_name} bias skipped by threshold. Magnitude: {magnitude}')

            self._collect_new_stats(nncf_graph, model_copy_subgraph, feed_dicts, subgraph_data)
            self._remove_unnecessary_stats(node_name, subgraphs_data)
        return main_model_transformer.transform(main_transformations_layout)

    def _remove_fq_from_inputs(self, model: TModel) -> TModel:
        """
        This model removes the activation Fake Quantize nodes (or Quantize-Dequantize pairs) from the model.
        It's needed for the further bias shift calculation that relates on quantized weights.

        :param model: Backend-specific model.
        :return: Backend-specific model without activation Fake Quantize nodes (or Quantize-Dequantize pairs).
        """
        transformation_layout = TransformationLayout()
        skip_types = []
        nncf_graph = NNCFGraphFactory.create(model)

        model_transformer = self._backend_entity.model_transformer(model)
        for skip_type in self._backend_entity.quantizer_types:
            skip_types.extend(skip_type.op_names)

        seen_nodes = []
        nodes_queue = deque(nncf_graph.get_input_nodes())
        while nodes_queue:
            current_node = nodes_queue.popleft()
            current_node_name = current_node.node_name

            if current_node_name in seen_nodes:
                continue

            seen_nodes.append(current_node_name)
            if current_node.node_type in skip_types:
                target_point = self._backend_entity.target_point(
                    TargetType.LAYER, current_node_name, 0)
                command = self._backend_entity.node_removing_command(target_point)
                transformation_layout.register(command)
            nodes_queue.extend(nncf_graph.get_next_nodes(current_node))

        return model_transformer.transform(transformation_layout)

    def _fill_subgraphs_data(self, nncf_graph: NNCFGraph) -> Dict[str, Dict]:
        """
        This method collects necessary data for the further optimized subgraph inference
        that reduces algorithm execution time.

        :param nncf_graph: NNCFGraph instance.
        :return: A dictionary with the node name as key and needed data for the subgraph building.
        """
        subgraphs_data = {}

        for node in nncf_graph.topological_sort():
            if node.node_type not in self._backend_entity.layers_with_bias_metatypes:
                continue
            input_node_names, stat_node_names = self._get_subgraph_data_for_node(node, nncf_graph)
            subgraphs_data[node.node_name] = {'input_node_names': input_node_names,
                                              'stat_node_names': stat_node_names}
        return subgraphs_data

    def _get_subgraph_data_for_node(self, node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[set, set]:
        """
        This method collects necessary data for the specified node and its subgraph.
        This data contains the nodes (NNCFNode) for the subgraph building
        and statistics collection (for the next step).

        :param node: NNCFNode instance. This is the main node that with bias that would be corrected (or not).
        :param nncf_graph: NNCFGraph instance for graph analysis.
        :return: A tuple with the set of the nodes for the subgraph input and statistics collection.
        """
        stats_nodes = set()
        input_nodes = set()

        def traverse_to_layers_with_bias(node, output):
            if node.node_type in self._backend_entity.layers_with_bias_metatypes:
                output.append(node)
                self._collected_stat_inputs.add(node.node_name)
                return True, output
            return False, output

        def traverse_to_input_layers(node, output):
            if node.node_name in self._collected_stat_inputs and node not in stats_nodes:
                output.append(node)
                return True, output
            return False, output

        for next_node in nncf_graph.get_next_nodes(node):
            stats_nodes.update(nncf_graph.traverse_graph(next_node, traverse_to_layers_with_bias))

        stats_nodes = stats_nodes if stats_nodes else nncf_graph.get_next_nodes(node)
        for stat_node in stats_nodes:
            input_nodes.update(nncf_graph.traverse_graph(stat_node, traverse_to_input_layers, traverse_forward=False))

        return [input_node.node_name for input_node in input_nodes], [stat_node.node_name for stat_node in stats_nodes]

    def _prepare_subgraph(self, node: NNCFNode, model: TModel, subgraph_data: Dict) -> TModel:
        """
        This method prepares the subgraph from the model for the further inference.

        :param node: NNCFNode instance for the current layer.
        :param model: Backend-specifig model instance.
        :param subgraph_data: A dictionary with the layers for the graph building.
        :return: Backend-specific subgraph extracted from the model.
        """
        input_node_names, statistic_node_names = subgraph_data['input_node_names'], subgraph_data['stat_node_names']
        extracted_model = self._backend_entity.extract_model(model, input_node_names, statistic_node_names)

        transformation_layout = TransformationLayout()
        model_transformer = self._backend_entity.model_transformer(extracted_model)
        _, output_port_id = self._backend_entity.get_activation_port_ids_for_bias_node(model, node)
        statistic_point = self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION,
                                                            node.node_name,
                                                            output_port_id)
        output_insertion_command = self._backend_entity.output_insertion_command(statistic_point)
        transformation_layout.register(output_insertion_command)
        return model_transformer.transform(transformation_layout)

    def _create_feed_dicts(self,
                           nncf_subgraph: NNCFGraph,
                           subgraph_data: Dict,
                           statistic_points: StatisticPointsContainer) -> List[Dict]:
        """
        Creates the list of the dictionaries that contains the input data for the model exection.

        :param nncf_subgraph: NNCFGraph instance.
        :param subgraph_data: A dictionary with the necessary data for current node.
        :param statistic_points: StatisticPointsContainer instance.
        :return: List of the dictionaries with the input data.
        """
        feed_dicts = []
        for stat_id in range(self.number_samples):
            feed_dict = {}
            for input_node_name in subgraph_data['input_node_names']:
                input_node = nncf_subgraph.get_node_by_name(input_node_name)
                input_tensor_names, _ = self._backend_entity.get_tensor_names(input_node)
                input_fp = self._get_fp_inputs(statistic_points, input_node_name)
                feed_dict[input_tensor_names[0]] = np.mean(input_fp[stat_id], axis=0, keepdims=True)
            feed_dicts.append(feed_dict)
        return feed_dicts

    def _compute_bias_shift(self,
                            node: NNCFNode,
                            model: TModel,
                            feed_dicts: List,
                            statistic_points: StatisticPointsContainer) -> np.ndarray:
        """
        Computes bias shift that will be used for the futher bias correction.

        :param node: NNCFNode instance, current layer.
        :param model: Backend-specific model.
        :param feed_dicts: List of dictionaries with the input data for model execition.
        :param statistic_points: StatisticPointsContainer instance.
        :return: Calculated bias shift value.
        """
        output_fp = self._get_fp_outputs(statistic_points, node.node_name)
        output_tensor_names = self._backend_entity.get_output_names(model, node.node_name)
        engine = EngineFactory.create(model)
        channel_axis = self._backend_entity.channel_axis_by_types[node.node_type]
        q_outputs = []
        for feed_dict in feed_dicts:
            q_output = engine.infer(feed_dict)
            q_output = self._backend_entity.process_model_output(q_output, output_tensor_names[0])
            q_outputs.append(self._backend_entity.tensor_processor.mean_per_channel(q_output, channel_axis).tensor)
        q_output = np.mean(q_outputs, axis=0)
        return output_fp - q_output

    @staticmethod
    def _get_bias_shift_magnitude(current_bias_value: np.ndarray, updated_bias_value: np.ndarray) -> float:
        """
        Calculates bias shift magnitude based on the current and updated values.

        :param current_bias_value: Initial bias value.
        :param updated_bias_value: Updated bias value.
        :return: Magnitude between original and updated bias values.
        """
        bias_shift_magnitude = np.inf
        if np.count_nonzero(current_bias_value == 0) == 0:
            bias_shift_magnitude = np.max(np.abs((updated_bias_value - current_bias_value) / current_bias_value))
        return bias_shift_magnitude

    def _correct_bias(self, model: TModel, bias_correction_command: TransformationCommand) -> TModel:
        """
        Returns the model (which can be represended as subgraph) with the updated bias value for the current layer.

        :param model: Backend-specific model.
        :param bias_correction_command: TransformationCommand instance for the bias correction.
        :return: Backend-specific model, but with the updated bias value.
        """
        model_transformer = self._backend_entity.model_transformer(model)
        transformation_layout = TransformationLayout()
        transformation_layout.register(bias_correction_command)
        return model_transformer.transform(transformation_layout)

    def _collect_new_stats(self, nncf_graph: NNCFGraph, model: TModel, feed_dicts: List, subgraph_data: Dict) -> None:
        """
        Updates the self._fp_inputs with the new statistics for the next layers
        after the correction of the bias for the current.

        :param nncf_graph: NNCFGraph instance.
        :param model: Backend-specific subgraph.
        :param feed_dicts: List of dictionaries with the input data for the subgraph.
        :param subgraph_data: A dictionary with the needed list of the statistic nodes that will be updated.
        """
        engine = EngineFactory.create(model)
        for feed_dict in feed_dicts:
            new_q_output = engine.infer(feed_dict)
            for stat_node_name in subgraph_data['stat_node_names']:
                stat_node = nncf_graph.get_node_by_name(stat_node_name)
                input_tensor_names, _ = self._backend_entity.get_tensor_names(stat_node)
                if stat_node_name not in self._fp_inputs:
                    self._fp_inputs[stat_node_name] = []
                self._fp_inputs[stat_node_name].append(new_q_output[input_tensor_names[0]])

    def _remove_unnecessary_stats(self, node_name: str, subgraphs_data: Dict[str, Dict]) -> None:
        """
        Removes unnecessary statistics that were collected before to reduce the memory usage.

        :param node_name: Current name of the node that was corrected.
        :param subgraphs_data: A dictionary of the data (input & statistic node names) that
            uses for the sub-graphs creation.
        """
        needed_stats_list = self._get_current_stats_list(node_name, subgraphs_data)
        node_inputs_name = subgraphs_data[node_name]['input_node_names']
        for node_input_name in node_inputs_name:
            if node_input_name not in needed_stats_list and node_input_name in self._fp_inputs:
                nncf_logger.debug(f'Dropped {node_input_name}')
                self._fp_inputs[node_input_name] = []

    def _get_current_stats_list(self, current_node_name: str, subgraphs_data: Dict[str, Dict]) -> List[str]:
        """
        Collects list of the statistics that needed for the future layers.

        :param node_name: Current name of the node that was corrected.
        :param subgraphs_data: A dictionary of the data (input & statistic node names) that
            uses for the sub-graphs creation.
        :return: The list of the layer names.
        """
        stat_nodes_list = []
        trigger = False
        for node_name in subgraphs_data:
            if node_name == current_node_name:
                trigger = True
                continue
            if trigger:
                for stat_node_name in subgraphs_data[node_name]['input_node_names']:
                    stat_nodes_list.append(stat_node_name)
        return stat_nodes_list

    def _get_fp_inputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> np.ndarray:
        """
        Makes out pre-layer needed data from the floating-point collected statistics.

        :param statistic_points: Filled StatisticPointsContainer.
        :param node_name: Name of the current layer.
        :return: Collected mean tensor data and shape for the further bias calculation.
        """
        def input_filter_func(point):
            return BiasCorrection in point.algorithm_to_tensor_collectors and \
                point.target_point.type == TargetType.PRE_LAYER_OPERATION

        if node_name in self._fp_inputs:
            return self._fp_inputs[node_name]

        input_fp = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(node_name,
                                                                              input_filter_func,
                                                                              BiasCorrection):
            input_fp.extend(tensor_collector.get_statistics().values)
        self._fp_inputs[node_name] = input_fp
        return self._fp_inputs[node_name]

    def _get_fp_outputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> np.ndarray:
        """
        Makes out post-layer needed data from the floating-point collected statistics.

        :param statistic_points: Filled StatisticPointsContainer.
        :param node_name: Name of the current layer.
        :return: Collected mean tensor data for the further bias calculation.
        """

        def output_filter_func(point):
            return BiasCorrection in point.algorithm_to_tensor_collectors and \
                point.target_point.type == TargetType.POST_LAYER_OPERATION

        output_fp = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(node_name,
                                                                              output_filter_func,
                                                                              BiasCorrection):
            output_fp.extend(tensor_collector.get_statistics().mean_values)
        return np.array(output_fp)

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        nncf_graph = NNCFGraphFactory.create(model) if self.nncf_graph is None else self.nncf_graph
        statistic_container = StatisticPointsContainer()

        biased_nodes = []
        for node in nncf_graph.topological_sort():
            if node.node_type in self._backend_entity.layers_with_bias_metatypes:
                biased_nodes.append(node)

        model_inputs = nncf_graph.get_input_nodes()
        biased_after_input_nodes = self._get_biased_after_input_nodes(nncf_graph, model_inputs)

        for biased_node in biased_nodes:
            biased_node_name = biased_node.node_name
            channel_axis = self._backend_entity.channel_axis_by_types[biased_node.node_type]
            input_port_id, output_port_id = self._backend_entity.get_activation_port_ids_for_bias_node(model,
                                                                                                       biased_node)
            if biased_node_name in biased_after_input_nodes:
                self._collected_stat_inputs.add(biased_node_name)
                statistic_point = self._backend_entity.target_point(TargetType.PRE_LAYER_OPERATION,
                                                                    biased_node_name,
                                                                    input_port_id)
                stat_collector = self._backend_entity.batch_statistic_collector(num_samples=self.number_samples)
                statistic_container.add_statistic_point(StatisticPoint(target_point=statistic_point,
                                                                       tensor_collector=stat_collector,
                                                                       algorithm=BiasCorrection))
            statistic_point = self._backend_entity.target_point(TargetType.POST_LAYER_OPERATION,
                                                                biased_node_name,
                                                                output_port_id)
            stat_collector = self._backend_entity.mean_statistic_collector(reduction_shape=channel_axis,
                                                                           num_samples=self.number_samples)
            statistic_container.add_statistic_point(StatisticPoint(target_point=statistic_point,
                                                                   tensor_collector=stat_collector,
                                                                   algorithm=BiasCorrection))

        return statistic_container

    def _get_biased_after_input_nodes(self, nncf_graph: NNCFGraph, model_inputs: List[NNCFNode]) -> Dict[str, str]:
        """
        This method finds and returns the first nodes with the bias in the model that follows after the input nodes.

        :param nncf_graph: NNCFGraph instance.
        :param model_inputs: List of the model inputs as NNCFNodes.
        :return: A dictionary with the names of the nodes with bias as keys and their input node names as values.
        """
        def traverse_to_biased(node, output, biased_op_types):
            if node.node_type in biased_op_types:
                output.append(node)
                return True, output
            return False, output

        biased_after_param_nodes = {}

        traverse_fn = partial(traverse_to_biased, biased_op_types=self._backend_entity.layers_with_bias_metatypes)
        for model_input in model_inputs:
            biased_nodes = nncf_graph.traverse_graph(model_input, traverse_fn)
            for biased_node in biased_nodes:
                activation_input = self._backend_entity.get_node_through_quantizer(biased_node, nncf_graph)
                biased_after_param_nodes[biased_node.node_name] = activation_input.node_name
        return biased_after_param_nodes
