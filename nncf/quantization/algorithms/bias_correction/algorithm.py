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

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import numpy as np
from tqdm import tqdm

from nncf import Dataset
from nncf import nncf_logger
from nncf.common.factory import EngineFactory
from nncf.common.factory import ModelTransformerFactory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import copy_model
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.bias_correction.backend import ALGO_BACKENDS

TModel = TypeVar("TModel")

BIAS_CORRECTION_THRESHOLD = 1000
OUTPUT_PORT_OF_NODE = 0


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
    """

    def __init__(
        self,
        subset_size: int = 100,
        threshold: float = BIAS_CORRECTION_THRESHOLD,
        apply_for_all_nodes: bool = False,
        inplace_statistics: bool = True,
        backend_params: Optional[Dict[str, Any]] = None,
    ):
        """
        :param subset_size: Size of a subset for the statistics collection,
            defaults to 100.
        :param threshold: The magnitude threshold that regulates the application of the
            shift. Magnitude calculates as the maximum of the absolute ratio of the
            shift to the original bias value. If the calculated value is less than the
            threshold, the shift will apply to the bias, defaults to 1000.
        :param apply_for_all_nodes: If True, then the bias correction be applied to all
            quantized nodes, if the node has no bias then a bias node will be inserted,
            and if False, then the bias correction will only be applied to quantized
            nodes that have a bias.
        :param inplace_statistics: Defines wheather to calculate quantizers statistics
            by backend graph operations or by default Python implementation, defaults
            to True.
        :param backend_params: Backend specific parameters.
        """
        super().__init__()
        self.subset_size = subset_size
        self.threshold = threshold
        self.apply_for_all_nodes = apply_for_all_nodes
        self.inplace_statistics = inplace_statistics
        self.backend_params = backend_params
        self.nncf_graph = None
        self._backend_entity = None
        self._collected_stat_inputs_map = {}
        self._fp_inputs = defaultdict(list)
        self._algorithm_key = f"BC_{hash(self)}"

        if self.apply_for_all_nodes:
            raise RuntimeError("BiasCorrection algorithm does not support apply_for_all_nodes=True yet")

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
            from nncf.quantization.algorithms.bias_correction.onnx_backend import ONNXBiasCorrectionAlgoBackend

            self._backend_entity = ONNXBiasCorrectionAlgoBackend()
        elif model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.bias_correction.openvino_backend import OVBiasCorrectionAlgoBackend

            self._backend_entity = OVBiasCorrectionAlgoBackend()
        else:
            raise RuntimeError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend)
            )

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._set_backend_entity(model)
        model = self._backend_entity.insert_null_biases(model, graph)
        main_transformations_layout = TransformationLayout()
        main_model_transformer = ModelTransformerFactory.create(model)

        model_copy = copy_model(model)
        graph_copy = NNCFGraphFactory.create(model_copy)
        model_copy = self._backend_entity.remove_fq_from_inputs(model_copy, graph_copy)
        nncf_graph = NNCFGraphFactory.create(model_copy)

        nodes_with_bias = []
        for node in nncf_graph.topological_sort():
            if self._backend_entity.is_node_with_bias(node, nncf_graph) and self._backend_entity.is_quantized_weights(
                node, nncf_graph
            ):
                nodes_with_bias.append(node)

        # We pre-collect information about the subgraph we need in order
        # to collect statistics for the change in the bias of each layer.
        # Also here we collect a list of layers that depend on the current one.

        # The collected information contains lists of input and output layers,
        # for which we will create a subgraph for inference and collection of statistics.
        subgraphs_data = [self._get_subgraph_data_for_node(node, nncf_graph) for node in nodes_with_bias]

        for position, (node, subgraph_data) in tqdm(
            list(enumerate(zip(nodes_with_bias, subgraphs_data))), desc="Applying Bias correction"
        ):
            node_name = node.node_name

            # We do not make an additional copy of the model because
            # the model transformer (that uses during sub-graph extraction) already does this internally when creating.
            model_copy_subgraph = self._prepare_subgraph(node, model_copy, nncf_graph, subgraph_data)

            # Then we create the necessary data lists from the previously collected statistics,
            # for the subgraph inference.
            feed_dicts = self._create_feed_dicts(model_copy_subgraph, subgraph_data, statistic_points)

            bias_shift = self._compute_bias_shift(node, model_copy_subgraph, feed_dicts, statistic_points)

            current_bias = self._backend_entity.get_bias_value(node, model_copy, nncf_graph)

            channel_axis = node.metatype.output_channel_axis
            if current_bias.ndim > 1:
                channel_axis = range(current_bias.ndim)[channel_axis]
                axes = [i for i in range(current_bias.ndim) if i != channel_axis]
                bias_shift = np.expand_dims(bias_shift, axes)

            updated_bias = current_bias + bias_shift
            magnitude = self._get_bias_shift_magnitude(current_bias, updated_bias)

            if magnitude < self.threshold:
                nncf_logger.debug(f"{node_name} bias would be changed. Magnitude: {magnitude}")
                bias_correction_command = self._backend_entity.create_bias_correction_command(
                    node, updated_bias, nncf_graph
                )
                model_copy_subgraph = self._correct_bias(model_copy_subgraph, bias_correction_command)
                model_copy = self._correct_bias(model_copy, bias_correction_command)
                main_transformations_layout.register(bias_correction_command)
            else:
                nncf_logger.debug(f"{node_name} bias skipped by threshold. Magnitude: {magnitude}")

            # After collecting data to change the bias value, we need to collect statistics for subsequent nodes,
            # but already take into account the bias update made earlier.
            self._collect_new_stats(model_copy_subgraph, feed_dicts, subgraph_data)

            # Also, we need to remove unnecessary statistics that we don't need anymore,
            # to reduce memory usage during the algorithm's pipeline.
            self._remove_unnecessary_stats(position, subgraphs_data)

        return main_model_transformer.transform(main_transformations_layout)

    def _get_subgraph_data_for_node(self, node: NNCFNode, nncf_graph: NNCFGraph) -> Dict[str, List[str]]:
        """
        This method collects necessary data for the specified node and its subgraph.
        This data contains the nodes (NNCFNode) for the subgraph building
        and statistics collection (for the next step).

        :param node: NNCFNode instance. This is the main node with bias that would be corrected (or not).
        :param nncf_graph: NNCFGraph instance for graph analysis.
        :return: A dict with the list of the nodes for the subgraph input and statistics collection.
        """
        statistic_nodes, subgraph_input_nodes, subgraph_output_nodes, subgraph_output_ids = [], [], [], []

        def fill_statistic_nodes(node):
            # A small hack to speed up graph traversal.
            if node in statistic_nodes or node in visited_nodes:
                return
            visited_nodes.append(node)

            # If we found a node with bias, we have to collect it as a statistic node,
            # and its input for _collected_stat_inputs_map,
            # which will be used during the collection of statistics for the next node.
            if self._backend_entity.is_node_with_bias(node, nncf_graph) and self._backend_entity.is_quantized_weights(
                node, nncf_graph
            ):
                statistic_nodes.append(node)
                activation_node, output_port_id = self._get_activation_node_and_port(node, nncf_graph)
                subgraph_output_nodes.append(activation_node)

                output_id = (activation_node.node_name, output_port_id)
                subgraph_output_ids.append(output_id)
                self._collected_stat_inputs_map[node.node_name] = output_id
                return

            for next_node in nncf_graph.get_next_nodes(node):
                fill_statistic_nodes(next_node)

        def fill_subgraph_input_nodes(node):
            # A small hack to speed up graph traversal.
            if node in subgraph_input_nodes or node in visited_nodes:
                return
            visited_nodes.append(node)

            # Since we need to find the inputs for the subgraph,
            # we can take only those layers for which we have already collected statistics.
            if node.node_name in self._collected_stat_inputs_map and node not in statistic_nodes:
                subgraph_input_nodes.append(node)
                return

            for previous_node in nncf_graph.get_previous_nodes(node):
                fill_subgraph_input_nodes(previous_node)

        # First, we need to find out the nodes with bias that follow by main node.
        # To collect statistics for next nodes.
        visited_nodes = []
        for next_node in nncf_graph.get_next_nodes(node):
            fill_statistic_nodes(next_node)

        # We then need to find nodes for which statistics have already been collected,
        # to use them as inputs for the subgraph.
        statistic_nodes = statistic_nodes if statistic_nodes else nncf_graph.get_next_nodes(node)
        visited_nodes = []
        for stat_node in statistic_nodes:
            fill_subgraph_input_nodes(stat_node)

        # In case the outputs were not found during the collection of statistics nodes,
        # we use the latter as the outputs of the subgraph.
        subgraph_output_nodes = subgraph_output_nodes if subgraph_output_nodes else statistic_nodes
        subgraph_output_names = [
            n.node_name for n in subgraph_output_nodes if NNCFGraphNodeType.OUTPUT_NODE not in n.node_name
        ]
        subgraph_data = {
            "subgraph_input_names": set(n.node_name for n in subgraph_input_nodes),
            "subgraph_output_names": set(subgraph_output_names),
            "subgraph_output_ids": set(subgraph_output_ids),
        }

        return subgraph_data

    def _prepare_subgraph(self, node: NNCFNode, model: TModel, nncf_graph: NNCFGraph, subgraph_data: Dict) -> TModel:
        """
        This method prepares the subgraph from the model for the further inference.

        :param node: NNCFNode instance for the current layer.
        :param model: Backend-specific model instance.
        :param nncf_graph: Instance of NNCFGraph.
        :param subgraph_data: A dictionary with the layers for the graph building.
        :return: Backend-specific subgraph extracted from the model.
        """
        extracted_model = self.extract_model(
            model, subgraph_data["subgraph_input_names"], subgraph_data["subgraph_output_names"]
        )

        transformation_layout = TransformationLayout()
        model_transformer = ModelTransformerFactory.create(extracted_model)

        # For layers with weights, there is only one output port - 0.
        statistic_point = self._backend_entity.target_point(
            TargetType.POST_LAYER_OPERATION, node.node_name, port_id=OUTPUT_PORT_OF_NODE
        )
        output_insertion_command = self._backend_entity.output_insertion_command(nncf_graph, statistic_point)
        transformation_layout.register(output_insertion_command)
        return model_transformer.transform(transformation_layout)

    def _create_feed_dicts(
        self, model: TModel, subgraph_data: Dict, statistic_points: StatisticPointsContainer
    ) -> List[Dict]:
        """
        Creates the list of the dictionaries that contains the input data for the model execution.

        :param model: TModel instance.
        :param subgraph_data: A dictionary with the necessary data for current node.
        :param statistic_points: StatisticPointsContainer instance.
        :return: List of the dictionaries with the input data.
        """
        feed_dicts = []
        statistics_size = self.subset_size
        statistics_per_input = {}

        for input_node_name in subgraph_data["subgraph_input_names"]:
            input_tensor_name = self._backend_entity.get_input_name(model, input_node_name)
            activation_name, port_id = self._collected_stat_inputs_map[input_node_name]
            input_fp = self._get_fp_inputs(statistic_points, node_name=activation_name, port_id=port_id)
            statistics_per_input[input_tensor_name] = input_fp
            statistics_size = min(statistics_size, len(input_fp))

        for stat_id in range(statistics_size):
            feed_dict = {}
            for input_node_name in subgraph_data["subgraph_input_names"]:
                input_tensor_name = self._backend_entity.get_input_name(model, input_node_name)
                # Since we do not use as inputs the layers from which the statistics are gathered,
                # but those that follow them, we need to take this into account when creating feed dicts.
                activation_name, port_id = self._collected_stat_inputs_map[input_node_name]
                feed_dict[input_tensor_name] = statistics_per_input[input_tensor_name][stat_id]
            feed_dicts.append(feed_dict)
        return feed_dicts

    def _compute_bias_shift(
        self, node: NNCFNode, model: TModel, feed_dicts: List, statistic_points: StatisticPointsContainer
    ) -> np.ndarray:
        """
        Computes bias shift that will be used for the further bias correction.

        :param node: NNCFNode instance, current layer.
        :param model: Backend-specific model.
        :param feed_dicts: List of dictionaries with the input data for model execution.
        :param statistic_points: StatisticPointsContainer instance.
        :return: Calculated bias shift value.
        """
        output_fp = self._get_fp_outputs(statistic_points, node.node_name)
        output_tensor_name = self._backend_entity.get_output_name(model, node.node_name, OUTPUT_PORT_OF_NODE)
        engine = EngineFactory.create(model)
        channel_axis = node.metatype.output_channel_axis
        q_outputs = []
        for feed_dict in feed_dicts:
            q_output = engine.infer(feed_dict)
            q_output = self._backend_entity.process_model_output(q_output, output_tensor_name)
            q_outputs.append(self._backend_entity.tensor_processor.mean_per_channel(q_output, channel_axis).tensor)
        # Here we get the per-sample average, so the axis is 0.
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
        Returns the model (which can be represented as subgraph) with the updated bias value for the current layer.

        :param model: Backend-specific model.
        :param bias_correction_command: TransformationCommand instance for the bias correction.
        :return: Backend-specific model, but with the updated bias value.
        """
        model_transformer = ModelTransformerFactory.create(model)
        transformation_layout = TransformationLayout()
        transformation_layout.register(bias_correction_command)
        return model_transformer.transform(transformation_layout)

    def _collect_new_stats(self, model: TModel, feed_dicts: List, subgraph_data: Dict) -> None:
        """
        Updates the self._fp_inputs with the new statistics for the next layers
        after the correction of the bias for the current.

        :param model: Backend-specific subgraph.
        :param feed_dicts: List of dictionaries with the input data for the subgraph.
        :param subgraph_data: A dictionary with the needed list of the statistic nodes that will be updated.
        """
        engine = EngineFactory.create(model)
        for feed_dict in feed_dicts:
            new_q_output = engine.infer(feed_dict)
            for output_node_name, output_id in subgraph_data["subgraph_output_ids"]:
                output_tensor_name = self._backend_entity.get_output_name(model, output_node_name, output_id)
                self._fp_inputs[(output_node_name, output_id)].append(new_q_output[output_tensor_name])

    def _remove_unnecessary_stats(self, position: int, subgraphs_data: Dict[str, Dict]) -> None:
        """
        Removes unnecessary statistics that were collected before to reduce the memory usage.

        :param position: Zero-based position of the current node that was corrected.
        :param subgraphs_data: A dictionary of the data (input & statistic node names) that
            uses for the sub-graphs creation.
        """
        # Collects list of the statistics that needed for the future layers.
        needed_stats_list = []
        for i in range(position + 1, len(subgraphs_data)):
            input_names = subgraphs_data[i]["subgraph_input_names"]
            needed_stats_list.extend([self._collected_stat_inputs_map[name][0] for name in input_names])

        node_inputs_name = subgraphs_data[position]["subgraph_input_names"]
        for node_input_name in node_inputs_name:
            activation_name, port_id = self._collected_stat_inputs_map[node_input_name]
            input_id = (activation_name, port_id)
            if activation_name not in needed_stats_list and input_id in self._fp_inputs:
                nncf_logger.debug(f"Dropped {activation_name} output statistics.")
                self._fp_inputs[input_id] = []

    def _get_fp_inputs(self, statistic_points: StatisticPointsContainer, node_name: str, port_id: int) -> np.ndarray:
        """
        Makes out pre-layer needed data from the floating-point collected statistics.

        :param statistic_points: Filled StatisticPointsContainer.
        :param node_name: Name of the current layer.
        :param port_id: Port id for statistics collection.
        :return: Collected mean tensor data and shape for the further bias calculation.
        """

        def input_filter_func(point):
            # For the floating-point statistics collected in POST_LAYER style,
            # we also need to determine the output port id.
            # For the cases when the layer has more than one (0) output port.
            return (
                self._algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.POST_LAYER_OPERATION
                and point.target_point.port_id == port_id
            )

        input_id = (node_name, port_id)
        if input_id in self._fp_inputs:
            return self._fp_inputs[input_id]

        input_fp = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            node_name, input_filter_func, self._algorithm_key
        ):
            input_fp.extend(tensor_collector.get_statistics().values)
        self._fp_inputs[input_id] = input_fp
        return self._fp_inputs[input_id]

    def _get_fp_outputs(self, statistic_points: StatisticPointsContainer, node_name: str) -> np.ndarray:
        """
        Makes out post-layer needed data from the floating-point collected statistics.

        :param statistic_points: Filled StatisticPointsContainer.
        :param node_name: Name of the current layer.
        :return: Collected mean tensor data for the further bias calculation.
        """

        def output_filter_func(point):
            return (
                self._algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.POST_LAYER_OPERATION
            )

        output_fp = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            node_name, output_filter_func, self._algorithm_key
        ):
            output_fp.extend(tensor_collector.get_statistics().mean_values)
        return np.array(output_fp)

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        model_copy = self._backend_entity.remove_fq_from_inputs(copy_model(model), graph)
        graph_copy = NNCFGraphFactory.create(model_copy)
        model_copy = self._backend_entity.insert_null_biases(model_copy, graph_copy)
        nncf_graph = NNCFGraphFactory.create(model_copy)
        statistic_container = StatisticPointsContainer()

        nodes_with_bias = [
            node for node in nncf_graph.topological_sort() if self._backend_entity.is_node_with_bias(node, nncf_graph)
        ]
        model_inputs = nncf_graph.get_input_nodes()

        # Collection of statistics after layers where biases will be corrected.
        for node in nodes_with_bias:
            node_name = node.node_name
            channel_axis = node.metatype.output_channel_axis

            # For layers with weights, there is only one output port - 0.
            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, node_name, port_id=OUTPUT_PORT_OF_NODE
            )
            stat_collector = self._backend_entity.mean_statistic_collector(
                reduction_shape=channel_axis, num_samples=self.subset_size, inplace=self.inplace_statistics
            )
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )

        # We must collect the nodes with biases following the model inputs.
        biased_after_input_nodes = self._get_biased_after_nodes(nncf_graph, model_inputs, model_copy)

        for biased_after_input_node in biased_after_input_nodes:
            # We need to collect activation input to register it for the biased layer as the layer with statistics.
            activation_node, output_port_id = self._get_activation_node_and_port(biased_after_input_node, nncf_graph)
            activation_node_name = activation_node.node_name

            self._collected_stat_inputs_map[biased_after_input_node.node_name] = (activation_node_name, output_port_id)
            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, activation_node_name, port_id=output_port_id
            )
            stat_collector = self._backend_entity.raw_statistic_collector(
                num_samples=self.subset_size, inplace=self.inplace_statistics
            )
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )

        # Then we need also to collect model input statistics to prevent cases when nodes with bias have no input data.
        for input_node in model_inputs:
            # We assume that input node has only one output port
            input_name = input_node.node_name
            if input_name in statistic_container:
                continue
            for next_layer in nncf_graph.get_next_nodes(input_node):
                self._collected_stat_inputs_map[next_layer.node_name] = (input_node.node_name, OUTPUT_PORT_OF_NODE)
            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, input_node.node_name, port_id=OUTPUT_PORT_OF_NODE
            )
            stat_collector = self._backend_entity.raw_statistic_collector(
                num_samples=self.subset_size, inplace=self.inplace_statistics
            )
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )

        return statistic_container

    def _get_activation_node_and_port(self, node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[NNCFNode, int]:
        """
        This method returns the activation layer and corresponding port id for the node.

        :param node: NNCFGraph node for which the activation is sought.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Tuple with the activation node and port id.
        """
        activation_port = self._backend_entity.get_activation_port_id(node, nncf_graph)
        activation_node = nncf_graph.get_input_edges(node)[activation_port].from_node
        port_id = nncf_graph.get_edge(activation_node, node).output_port_id
        return activation_node, port_id

    def _get_biased_after_nodes(self, nncf_graph: NNCFGraph, nodes: List[NNCFNode], model: TModel) -> List[NNCFNode]:
        """
        This method finds and returns nodes with the bias in the model that follows after the input nodes.

        :param nncf_graph: NNCFGraph instance.
        :param nodes: List of the model inputs as NNCFNodes.
        :param model: TModel instance.
        :return: List of the nodes with bias.
        """

        def traverse_to_biased(node, condition_container):
            # A small hack to speed up graph traversal.
            if node in visited_nodes:
                return
            visited_nodes.append(node)

            # We need to collect nodes for the next recursion step.
            node_children = nncf_graph.get_next_nodes(node)

            # Check that node is with bias.
            if self._backend_entity.is_node_with_bias(node, nncf_graph):
                condition_container.add(node)
                return

            for node_child in node_children:
                traverse_to_biased(node_child, condition_container)

        biased_nodes = set()
        visited_nodes = []
        for node in nodes:
            nncf_logger.debug(f"Looking for biased nodes after {node.node_name} layer.")
            traverse_to_biased(node, condition_container=biased_nodes)

        dependant_nodes = set()
        # After finding the nodes following the provided layers, we need to make sure
        # that the found nodes really only depend on the main layers, and not on each other.
        for biased_node in biased_nodes:
            visited_nodes = []
            nncf_logger.debug(f"Filtering biased nodes after {biased_node.node_name} layer.")
            for next_node in nncf_graph.get_next_nodes(biased_node):
                traverse_to_biased(next_node, condition_container=dependant_nodes)

        return list(biased_nodes - dependant_nodes)

    def extract_model(self, model: TModel, input_node_names: List[str], output_node_names: List[str]) -> TModel:
        """
        Returns the backend-specific model that bounded by the specified input & output layers.

        :param model: Backend-specific model.
        :param input_node_names: List with the input node names.
        :param output_node_names: List with the output node names.
        :return: Extracted backend-specific model.
        """
        transformation_layout = TransformationLayout()
        model_transformer = ModelTransformerFactory.create(model)
        model_extraction_command = self._backend_entity.model_extraction_command(input_node_names, output_node_names)
        transformation_layout.register(model_extraction_command)
        return model_transformer.transform(transformation_layout)
