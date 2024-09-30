# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC
from collections import defaultdict
from typing import TypeVar, List, Tuple, Optional, Iterable

from nncf.common.graph import NNCFGraph, NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType, TargetPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer, StatisticPoint
from nncf.common.utils.backend import BackendType, get_backend
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.tensor import Tensor
from nncf.tensor import functions as fns

TModel = TypeVar("TModel")


class TokenAveragedStatisticsAlgorithm(Algorithm, ABC):
    def __init__(self):
        super().__init__()

        self._backend_entity: TokenAveragedStatisticsBackend = None
        self._algorithm_key = None

    def set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """

    def get_statistic_points(self, model: TModel, graph: NNCFGraph, nodes: Iterable[NNCFNode], subset_size: Optional[int] = None) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """

        statistic_container = StatisticPointsContainer()
        for node in nodes:
            act_node, output_port_id = self._get_activation_node_and_port(node, graph)
            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, act_node.node_name, port_id=output_port_id
            )
            stat_collector = self._backend_entity.statistic_collector(reduction_axes=(0, 1), subset_size=subset_size)
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )

        return statistic_container

    def _get_statistics(self, statistic_points: StatisticPointsContainer, node_name: str, port_id: int) -> Tuple[List[Tensor], List[Tuple]]:
        """
        Collects statistic values for the given node and port id.

        :param statistic_points: Filled StatisticPointsContainer.
        :param node_name: Name of the current layer.
        :param port_id: Port id for statistics collection.
        :return: Collected list of tensor data.
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

        mean_values = []
        shapes = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            node_name, input_filter_func, self._algorithm_key
        ):
            for value in tensor_collector.get_statistics()[self._backend_entity.MEAN_STAT]:
                mean_values.append(value[0, 0])
            for value in tensor_collector.get_statistics()[self._backend_entity.SHAPE_STAT]:
                shapes.append(value.data)

        return mean_values, shapes

    def _get_activation_node_and_port(self, node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[NNCFNode, int]:
        """
        This method returns the activation layer and corresponding port id for the node.

        :param node: NNCFGraph node for which the activation is sought.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Tuple with the activation node and port id.
        """
        activation_port = self._backend_entity.get_activation_port_id(node, nncf_graph)
        activation_edge = nncf_graph.get_input_edge_by_port_id(node, activation_port)
        activation_node = activation_edge.from_node
        port_id = activation_edge.output_port_id
        return activation_node, port_id

    @staticmethod
    def _process_stats(means: List[Tensor], shapes: List[Tuple], subset_size: int) -> Tuple[Tensor, Tensor]:
        """
        It's a processing of activations shared between AWQ, Scale Estimation and LoRA Correction algorithms.

        :param stats: list of activation statistics for a layer that contains N tensors with shape [SeqLen, HiddenDim]
        :type stats: List[TTensor]
        :param subset_size: The number of samples for AWQ.
        :type subset_size: int
        :return: tuple of the following tensors:
            s - maximum channel magnitude across samples [HiddenDim]
            X - average channel magnitude across tokens in the sequence [HiddenDim, SampleSize]
        :rtype: Tuple[TTensor, TTensor]
        """
        X = fns.stack(means)  # [Batch, HiddenDim]
        X_full = fns.transpose(X)  # [HiddenDim, Batch]

        # prevent high memory and time consumption
        if X_full.shape[1] > subset_size:
            lens = [shape[0] * shape[1] for shape in shapes]
            step = X_full.shape[1] // subset_size
            idxs = [i[0] for i in sorted(enumerate(lens), key=lambda x: -x[1])][::step]
            X = X_full[:, idxs]  # [HiddenDim, SampleSize]
        else:
            X = X_full
        s = fns.max(fns.abs(X_full), axis=1)  # [HiddenDim]
        return s, X
