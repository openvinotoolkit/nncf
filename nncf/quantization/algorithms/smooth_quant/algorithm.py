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

from collections import Counter
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, TypeVar

from nncf import Dataset
from nncf.common.factory import ModelTransformerFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.logging.track_progress import track
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.algorithm import Algorithm

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")
STATISTIC_BRANCH_KEY = "abs_max"
ALPHA_MAP = {"convolution": 0.05, "matmul": 0.95}


class SmoothQuant(Algorithm):
    """
    Post-training SmoothQuant algorithm implementation.

    The main purpose of this algorithm is to reduce activation quantization error
    via the insertion of nodes with smoothing scales for weighted layers.
    """

    def __init__(
        self, subset_size: int = 300, inplace_statistics: bool = True, alpha_map: Dict[str, float] = ALPHA_MAP
    ):
        """
        :param subset_size: Size of a subset for the statistics collection,
            default is 300.
        :param inplace_statistics: Defines whether to calculate quantizers statistics
            by backend graph operations or by default Python implementation, defaults
            to True.
        :param alpha_map: The parameter that regulates the calculation of the scale for different layers.
            The default value for each layer in the ALPHA_MAP.
            Negative value switches off the algorithm for correspondent nodes.
        """

        super().__init__()
        self._subset_size = subset_size
        self._inplace_statistics = inplace_statistics
        self._backend_entity = None
        self._algorithm_key = f"SQ_{hash(self)}"
        self._cached_multiply_names = Counter()
        self._alpha_map = alpha_map

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.OPENVINO]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.smooth_quant.openvino_backend import OVSmoothQuantAlgoBackend

            self._backend_entity = OVSmoothQuantAlgoBackend()
        else:
            raise RuntimeError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._set_backend_entity(model)
        alpha_map = self._get_alpha_map()

        nodes_to_smooth_data = self._get_nodes_to_smooth_data(graph, alpha_map.keys())
        model_transformer = ModelTransformerFactory.create(model)
        transformation_layout = TransformationLayout()

        node_groups = self._group_nodes_by_source(nodes_to_smooth_data, graph)

        best_scale = None
        for group_id, nodes in track(node_groups.items(), description="Applying Smooth Quant"):
            best_ratio = 0.0
            empty_statistic = False
            for node_to_smooth in nodes:
                source_node, input_port_id, source_output_port_id, _ = group_id
                activations_value = self._get_statistics_for_node(
                    statistic_points, node_to_smooth.node_name, input_port_id
                )
                if any(val is None for val in activations_value):
                    empty_statistic = True
                    break
                activations_value = self._backend_entity.clip_statistics(activations_value)

                weight_port = self._backend_entity.get_weight_tensor_port_id(node_to_smooth)
                weight_value = self._backend_entity.get_weight_value(node_to_smooth, model, weight_port)
                weight_statistics = self._process_weight_statistics(node_to_smooth, weight_value, weight_port)
                weight_statistics = self._backend_entity.clip_statistics(weight_statistics)

                alpha = alpha_map[node_to_smooth.metatype]

                scales, ratio = self._backend_entity.calculate_scale_and_ratio(
                    activations_value, weight_statistics, alpha
                )

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_scale = deepcopy(scales)

            if empty_statistic:
                nncf_logger.debug(
                    f"Skipped SmoothQuant for nodes after {source_node.node_name} because of the empty statistics."
                )
                continue

            if best_scale is None:
                nncf_logger.debug(
                    f"Skipped SmoothQuant for nodes after {source_node.node_name} because of the empty scale."
                )
                continue

            for node_to_smooth in nodes:
                weights_scale = self._calculate_weight_scale(best_scale, node_to_smooth)
                weight_port = self._backend_entity.get_weight_tensor_port_id(node_to_smooth)
                weight_value = self._backend_entity.get_weight_value(node_to_smooth, model, weight_port)
                scaled_weight = weight_value * weights_scale
                weight_update_command = self._backend_entity.weight_update_command(
                    node_to_smooth, scaled_weight, weight_port
                )
                transformation_layout.register(weight_update_command)

            activations_shape = graph.get_output_edges(source_node)[source_output_port_id].tensor_shape
            activation_scale = self._calculate_activation_scale(best_scale, activations_shape, nodes, graph)

            scale_node_name = self._create_scale_node_name(source_node.node_name, source_output_port_id)
            scale_insertion_command = self._backend_entity.scale_insertion_command(
                source_node, activation_scale, source_output_port_id, nodes, scale_node_name
            )
            transformation_layout.register(scale_insertion_command)

        transformed_model = model_transformer.transform(transformation_layout)
        return transformed_model

    def _group_nodes_by_source(self, nodes_to_smooth: List[Dict], nncf_graph: NNCFGraph) -> Dict[tuple, List]:
        """
        Groups nodes that will be smoothed by source (parent node).

        :param nodes_to_smooth: List of the nodes that will be smoothed.
        :param nncf_graph: NNCFGraph instance.
        :return: Dictionary with the source info as key and grouped nodes as value.
        """
        groups = defaultdict(list)
        for node_data in nodes_to_smooth:
            node_to_smooth = node_data["node_to_smooth"]
            input_act_port = node_data["input_act_port"]

            source_node = nncf_graph.get_input_edges(node_to_smooth)[input_act_port].from_node
            edge = nncf_graph.get_edge(source_node, node_to_smooth)
            # Such group_id (with node, ports, and shape as a hash) allows us to be confident
            # that all sensitive parameters are equal for successor nodes are equal.
            group_id = (source_node, input_act_port, edge.output_port_id, hash(str(edge.tensor_shape)))
            groups[group_id].append(node_to_smooth)

        return groups

    def _get_statistics_for_node(
        self, statistic_points: StatisticPointsContainer, node_name: str, act_port: int
    ) -> List[TTensor]:
        """
        Collects statistics for node.

        :param statistic_points: StatisticPointsContainer instance.
        :param node_name: Name of the node for collection.
        :param act_port: Activation port id.
        :return: List of the TTensor instances.
        """

        def filter_func(point: StatisticPoint) -> bool:
            return (
                self._algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.PRE_LAYER_OPERATION
                and point.target_point.port_id == act_port
            )

        statistics_for_node = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            node_name, filter_func, self._algorithm_key
        ):
            statistics_for_node.append(tensor_collector.get_statistics()[STATISTIC_BRANCH_KEY])
        return statistics_for_node

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        statistic_container = StatisticPointsContainer()

        self._set_backend_entity(model)
        alpha_map = self._get_alpha_map()

        nodes_to_smooth_data = self._get_nodes_to_smooth_data(graph, alpha_map.keys())

        for node_data in nodes_to_smooth_data:
            node_to_smooth = node_data["node_to_smooth"]
            target_point = self._backend_entity.target_point(
                TargetType.PRE_LAYER_OPERATION,
                target_node_name=node_to_smooth.node_name,
                port_id=node_data["input_act_port"],
            )
            input_reduction_axes = self._calculate_input_reduction_axes(
                graph, node_to_smooth, node_data["input_act_port"]
            )
            stat_collector = self._backend_entity.get_abs_max_channel_collector(
                self._subset_size, input_reduction_axes, self._inplace_statistics, STATISTIC_BRANCH_KEY
            )
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=target_point,
                    tensor_collector=stat_collector,
                    algorithm=self._algorithm_key,
                )
            )
        return statistic_container

    def _get_nodes_to_smooth_data(self, nncf_graph: NNCFGraph, node_metatypes: List[OperatorMetatype]) -> List[Dict]:
        """
        Collects layers whose activations will be smoothed.

        :param nncf_graph: NNCFGraph instance.
        :param node_metatypes: Metatypes for nodes to search for.
        :return: List with the data for each layer.
        """
        nodes_with_weights = nncf_graph.get_nodes_by_metatypes(node_metatypes)
        nodes_to_smooth_data = []

        for node_with_weight in nodes_with_weights:
            if not self._backend_entity.is_node_with_weights(node_with_weight):
                continue

            ports_map = self._backend_entity.get_input_ports_map(node_with_weight, nncf_graph)
            input_edges = nncf_graph.get_input_edges(node_with_weight)
            weight_node = input_edges[ports_map["weight"]].from_node
            activation_node = input_edges[ports_map["activation"]].from_node

            # Skipping agnostic layers as inputs to propagate quantizer
            # Only for Convolution layers
            if (
                node_with_weight.metatype == self._backend_entity.convolution_metatype
                and activation_node.metatype in self._backend_entity.quantize_agnostic_metatypes
            ):
                continue

            # Skipping shared weights
            if len(nncf_graph.get_next_nodes(weight_node)) > 1:
                continue

            nodes_to_smooth_data.append(
                {
                    "node_to_smooth": node_with_weight,
                    "input_act_port": ports_map["activation"],
                }
            )
        return nodes_to_smooth_data

    def _calculate_activation_scale(
        self, scale_value: TTensor, activations_shape: List[int], nodes: List[NNCFNode], nncf_graph: NNCFGraph
    ) -> TTensor:
        """
        Calculates activation scales for Smooth node.

        :param scale_value: Base scale value.
        :param activations_shape: activation tensor shape.
        :param nodes: List of consumers for Smooth node.
        :return: Calculated per-channel activation scale.
        """
        activation_ports_map = {
            node: self._backend_entity.get_input_ports_map(node, nncf_graph)["activation"] for node in nodes
        }
        channel_axes = [
            self._backend_entity.get_activation_channel_axis(node, port) for node, port in activation_ports_map.items()
        ]
        channel_axis = channel_axes[0]

        if not all(axis == channel_axis for axis in channel_axes):
            raise RuntimeError(f"Channel axes for nodes {[n.node_name for n in nodes]} are not identical")

        activations_size = len(activations_shape)
        return self._backend_entity.calculate_activation_scale(scale_value, activations_size, channel_axis)

    def _calculate_weight_scale(self, scale_value: TTensor, node: NNCFNode) -> TTensor:
        """
        Calculates scale for weight tensor.

        :param scale_value: Base scale value.
        :param node: Consumer for Smooth node.
        :return: Calculated scale for weights.
        """
        port_id = self._backend_entity.get_weight_tensor_port_id(node)
        weights_size = len(node.layer_attributes.constant_attributes[port_id]["shape"])
        if weights_size > 1:
            channel_axis = self._backend_entity.get_weight_channel_axis(node, port_id)
            return self._backend_entity.calculate_weight_scale(scale_value, weights_size, channel_axis)
        return scale_value

    def _calculate_input_reduction_axes(self, nncf_graph: NNCFGraph, node: NNCFNode, input_port: int) -> Tuple[int]:
        """
        Returns reduction axes for specified input.

        :param nncf_graph: NNCFGraph instance.
        :param node: NNCFNode to check.
        :param input_port: Specified input port id.
        :return: Calculated reduction axes.
        """
        shape = nncf_graph.get_input_edges(node)[input_port].tensor_shape
        reduction_axes = tuple([])
        if len(shape) > 1:
            channel_axis = self._backend_entity.get_activation_channel_axis(node, input_port)
            reduction_axes = self._backend_entity.get_channel_agnostic_reduction_axes(channel_axis, shape)
        return reduction_axes

    def _process_weight_statistics(self, node: NNCFNode, weights: TTensor, port_id: int) -> TTensor:
        """
        Returns processed weight statistics for node.

        :param node: NNCFNode to check.
        :param weights: Backend-specific weights.
        :param port_id: Weight port id.
        :return: Weight statistic for node.
        """
        channel_axis = 0
        if len(weights.shape) > 1:
            channel_axis = self._backend_entity.get_weight_channel_axis(node, port_id)
        reduction_shape = [i for i, _ in enumerate(weights.shape)]
        reduction_shape.pop(channel_axis)
        return self._backend_entity.process_weight_statistics(weights, tuple(reduction_shape))

    def _create_scale_node_name(self, source_name: str, source_port_id: int) -> str:
        """
        Returns uniqie scale node name for new layer.

        :param source_name: Source layer name.
        :param source_port_id: Source port id.
        :return: Generated uniqie name.
        """
        scale_node_name = f"{source_name}_{source_port_id}"
        unique_index = self._cached_multiply_names[scale_node_name]
        self._cached_multiply_names[scale_node_name] += 1
        return f"{scale_node_name}_{unique_index}/nncf_smooth_quant"

    def _get_alpha_map(self) -> Dict[OperatorMetatype, float]:
        """
        Returns alpha map by metatypes.

        :return: Alpha map by metatypes.
        """
        alpha_by_metatype_map = {}
        name_to_metatype = {
            "convolution": self._backend_entity.convolution_metatype,
            "matmul": self._backend_entity.matmul_metatype,
        }
        for type_name, alpha_value in self._alpha_map.items():
            if alpha_value < 0:
                nncf_logger.debug(
                    f"Smooth Quant algorithm does not support negative parameter for {type_name}! "
                    "Skipping these layers."
                )
                continue
            metatype = name_to_metatype[type_name]
            alpha_by_metatype_map[metatype] = alpha_value
        return alpha_by_metatype_map
