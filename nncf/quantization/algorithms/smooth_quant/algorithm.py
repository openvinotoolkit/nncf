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

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, TypeVar

from nncf import Dataset
from nncf.common.factory import ModelTransformerFactory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.logging import nncf_logger
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.smooth_quant.backend import ALGO_BACKENDS

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")
STATISTIC_BRANCH_KEY = "abs_max"


class SmoothQuant(Algorithm):
    """
    Post-training SmoothQuant algorithm implementation.

    The main purpose of this algorithm is to reduce activation quantization error
    via the insertion of nodes with smoothing scales for weighted layers.
    """

    def __init__(self, subset_size: int = 300, inplace_statistics: bool = True, alpha: Optional[int] = 0.95):
        """
        :param subset_size: Size of a subset for the statistics collection,
            default is 300.
        :param inplace_statistics: Defines wheather to calculate quantizers statistics
            by backend graph operations or by default Python implementation, defaults
            to True.
        :param alpha: The parameter that regulates the calculation of the scale.
            The default value is 0.95. Negative value switches off the algorithm.
        """
        super().__init__()
        self._subset_size = subset_size
        self._inplace_statistics = inplace_statistics
        self._backend_entity = None
        self._alpha = alpha

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        return ALGO_BACKENDS.registry_dict

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
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend)
            )

    def _apply(
        self,
        model: TModel,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        if self._alpha < 0:
            nncf_logger.info("Skipping SmoothQuant algorithm because alfa parameter is negative.")
            return model

        nncf_graph = NNCFGraphFactory.create(model)
        nodes_to_smooth_data = self._get_nodes_to_smooth_data(nncf_graph)
        model_transformer = ModelTransformerFactory.create(model)
        transformation_layout = TransformationLayout()

        node_groups = self._group_nodes_by_source(nodes_to_smooth_data, nncf_graph)

        for group_id, nodes in node_groups.items():
            best_ratio = 0.0
            for node_to_smooth in nodes:
                source_node, port_id = group_id
                activations_value = self._get_statistics_for_node(statistic_points, node_to_smooth.node_name, port_id)
                activations_value = self._backend_entity.clip_statistics(activations_value)

                weights_port = self._backend_entity.get_weight_tensor_port_id(node_to_smooth)
                weights_value = self._backend_entity.get_weight_statistics(node_to_smooth, model, weights_port)
                weights_value = self._backend_entity.clip_statistics(weights_value)

                scales, ratio = self._backend_entity.calculate_scale_and_ratio(
                    activations_value, weights_value, self._alpha
                )

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_scale = deepcopy(scales)

            activation_scales = self._backend_entity.calculate_activation_scale(best_scale, nodes)
            weight_scales = self._backend_entity.calculate_weight_scale(best_scale, nodes)

            for node_to_smooth in nodes:
                weights_port = self._backend_entity.get_weight_tensor_port_id(node_to_smooth)
                weights_value = self._backend_entity.get_weight_value(node_to_smooth, model, weights_port)

                scaled_weights = weights_value * weight_scales
                weight_update_command = self._backend_entity.weight_update_command(
                    node_to_smooth, scaled_weights, weights_port
                )
                transformation_layout.register(weight_update_command)

            scale_insertion_command = self._backend_entity.scale_insertion_command(
                source_node, activation_scales, port_id, nodes
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
            group_id = (source_node, input_act_port)
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
                SmoothQuant in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.PRE_LAYER_OPERATION
                and point.target_point.port_id == act_port
            )

        statistics_for_node = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(node_name, filter_func, SmoothQuant):
            statistics_for_node.append(tensor_collector.get_statistics()[STATISTIC_BRANCH_KEY])
        return statistics_for_node

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        statistic_container = StatisticPointsContainer()

        if self._alpha < 0:
            nncf_logger.debug(
                "Skipping statistics collection for SmoothQuant algorithm because alfa parameter is negative."
            )
            return statistic_container

        self._set_backend_entity(model)
        nncf_graph = NNCFGraphFactory.create(model)

        nodes_to_smooth_data = self._get_nodes_to_smooth_data(nncf_graph)

        for node_data in nodes_to_smooth_data:
            node_to_smooth = node_data["node_to_smooth"]
            target_point = self._backend_entity.target_point(
                TargetType.PRE_LAYER_OPERATION,
                target_node_name=node_to_smooth.node_name,
                port_id=node_data["input_act_port"],
            )
            input_reduction_shape = self._backend_entity.calculate_input_reduction_shape(
                nncf_graph, node_to_smooth, node_data["input_act_port"]
            )
            stat_collector = self._backend_entity.get_abs_max_channel_collector(
                self._subset_size, input_reduction_shape, self._inplace_statistics, STATISTIC_BRANCH_KEY
            )
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=target_point,
                    tensor_collector=stat_collector,
                    algorithm=SmoothQuant,
                )
            )
        return statistic_container

    def _get_nodes_to_smooth_data(self, nncf_graph: NNCFGraph) -> List[Dict]:
        """
        Collects layers whose activations will be smoothed.

        :param nncf_graph: NNCFGraph instance.
        :return: List with the data for each layer.
        """
        nodes_with_weights = nncf_graph.get_nodes_by_metatypes(self._backend_entity.weighted_metatypes)
        nodes_to_smooth_data = []

        for node_with_weight in nodes_with_weights:
            if not self._backend_entity.is_node_with_weights(node_with_weight):
                continue

            ports_map = self._backend_entity.get_input_ports_map(node_with_weight, nncf_graph)
            weight_node = nncf_graph.get_input_edges(node_with_weight)[ports_map["weight"]].from_node

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
