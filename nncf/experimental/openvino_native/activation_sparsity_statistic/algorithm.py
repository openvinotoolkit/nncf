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

from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar

from nncf import Dataset
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.logging import nncf_logger
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import copy_model
from nncf.common.utils.backend import get_backend
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.quantization.algorithms.algorithm import AlgorithmParameters
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS

TModel = TypeVar("TModel")


class ActivationSparsityStatisticParameters(AlgorithmParameters):
    """
    Parameters of ActivationSparsityStatistic algorithm
    """

    def __init__(self, number_samples: int = 100, target_node_types: Optional[list] = None) -> None:
        """
        :param number_samples: The number of the samples for the statistics collection.
        :param target_node_types: List of node types for which statistics will be collected.
            If None or empty, statistics will be collected for all nodes.
        """
        self.number_samples = number_samples
        self.target_node_types = target_node_types


class ActivationSparsityStatistic(Algorithm):
    """
    Collect activation sparsity statistic algorithm implementation.

    The main purpose of this algorithm to collect of percentage of zero in activation tensors.

    :param number_samples: The number of the samples for the statistics collection.
    :param nncf_graph: NNCFGraph class for the algorithm.
    """

    def __init__(self, parameters: ActivationSparsityStatisticParameters) -> None:
        """
        :param parameters: The instance of the FastBiasCorrectionParameters.
        """
        super().__init__()
        self.number_samples = parameters.number_samples
        self.target_node_types = parameters.target_node_types
        self.nncf_graph = None
        self._backend_entity = None

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
            from nncf.experimental.openvino_native.activation_sparsity_statistic.ov_backend import \
                OVActivationSparsityStatisticAlgoBackend

            self._backend_entity = OVActivationSparsityStatisticAlgoBackend()
        else:
            raise RuntimeError(
                "Cannot return backend-specific entity " "because {} is not supported!".format(model_backend)
            )

    def _create_statistics_aggregator(self, dataset: Dataset, backend: BackendType) -> StatisticsAggregator:
        """
        Creates backend-specific StatisticsAggregator.

        :param engine: engine for the model execution
        :param dataset: dataset for the statistics collection and validation
        :param model_transformer: backend-specific ModelTransformerBase instance
        :param backend: model backend type for the further differentiations
        :return: backend-specific StatisticsAggregator
        """
        if backend == BackendType.OPENVINO:
            from nncf.experimental.openvino_native.statistics.aggregator import OVStatisticsAggregator

            return OVStatisticsAggregator(dataset)
        return None

    def _get_target_nodes(self, nncf_graph: NNCFGraph) -> List[NNCFNode]:
        """
        Return list of target nodes

        :param nncf_graph: Graph of the model.

        :return List[NNCFNode]: list of target nodes.
        """
        if self.target_node_types is not None and not self.target_node_types:
            return nncf_graph.get_nodes_by_types(self.target_node_types)
        return nncf_graph.get_nodes_by_types(self._backend_entity.default_target_node_types())

    def _apply(
        self,
        model: TModel,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        modified_model = copy_model(model)
        if statistic_points is None:  # TODO: need ???
            backend = get_backend(modified_model)
            # TODO: Remove after OpenVINO Native is removed from experimental
            if backend == BackendType.OPENVINO:
                nncf_logger.warning("You are using experimental OpenVINO backend.")

            statistics_aggregator = self._create_statistics_aggregator(dataset, backend)

            statistic_points = self.get_statistic_points(modified_model)
            statistics_aggregator.register_stastistic_points(statistic_points)

            statistics_aggregator.collect_statistics(modified_model)
            statistic_points = statistics_aggregator.statistic_points

        for node_op in modified_model.get_ops():
            node_name = node_op.get_friendly_name()
            node_static_points = statistic_points.get(node_name, [])

            for node_static_point in node_static_points:
                # TODO: how to correctly get statistics
                tensor_collector = node_static_point.algorithm_to_tensor_collectors[ActivationSparsityStatistic][0]

                statistic = tensor_collector.get_statistics()
                percentage_of_zeros = statistic.percentage_of_zeros
                port_id = node_static_point.target_point.port_id
                # TODO: how to save data, it do nothing
                print(f"{node_name=} {port_id=} {percentage_of_zeros=}")
                node_op.set_attribute("percentage_of_zeros_statistic", percentage_of_zeros)

        return modified_model

    def _add_statistic_point(self, container: StatisticPointsContainer, point: TargetPoint) -> None:
        """
        Adds specific statistic point.

        :param container: StatisticPointsContainer instance.
        :param point: TargetPoint for statistic collection.
        """
        stat_collector = self._backend_entity.percentage_of_zeros_statistic_collector(num_samples=self.number_samples)
        container.add_statistic_point(
            StatisticPoint(target_point=point, tensor_collector=stat_collector, algorithm=ActivationSparsityStatistic)
        )

    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        self._set_backend_entity(model)
        nncf_graph = NNCFGraphFactory.create(model)

        nodes = self._get_target_nodes(nncf_graph)
        statistic_container = StatisticPointsContainer()
        for node in nodes:
            for edge in nncf_graph.get_input_edges(node):
                if edge.from_node.node_type not in self._backend_entity.ignored_input_node_types():
                    statistic_point = self._backend_entity.target_point(node.node_name, edge.input_port_id)
                    self._add_statistic_point(statistic_container, statistic_point)

        return statistic_container
