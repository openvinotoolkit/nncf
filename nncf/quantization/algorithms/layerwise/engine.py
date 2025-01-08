# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data.dataset import Dataset
from nncf.errors import UnsupportedBackendError
from nncf.quantization.algorithms.layerwise.iterator import LayerwiseIterator
from nncf.quantization.algorithms.layerwise.scheduler import LayerwiseScheduler
from nncf.quantization.algorithms.layerwise.scheduler import LayerwiseStep
from nncf.quantization.algorithms.layerwise.scheduler import NodeOutputPort
from nncf.tensor import Tensor

TModel = TypeVar("TModel")


class LayerwiseEngine:
    """
    A class to perform layer-wise processing of a model using a specified strategy.

    This class allows for layer-wise analysis and manipulation of a model. It supports
    iteration through target nodes and collecting inputs or outputs of target nodes.
    """

    def __init__(
        self,
        subset_size: int = 100,
        collect_inputs: bool = True,
    ):
        """
        :param subset_size: Number of samples from the dataset to use, defaults to 100.
        :param collect_inputs: Whether to collect inputs for the layers, defaults to True.

        """
        self._scheduler = LayerwiseScheduler()
        self._subset_size = subset_size
        self._collect_inputs = collect_inputs

        self._backend_entity = None
        self._algorithm_key = f"LayerwiseEngine_{hash(self)}"

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic for the engine.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.layerwise.openvino_backend import OVLayerwiseEngineBackend

            self._backend_entity = OVLayerwiseEngineBackend()
        else:
            raise UnsupportedBackendError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def _get_statistics(self, statistic_points: StatisticPointsContainer, node_name: str, port_id: int) -> List[Tensor]:
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

        res = []
        for tensor_collector in statistic_points.get_algo_statistics_for_node(
            node_name, input_filter_func, self._algorithm_key
        ):
            for value in tensor_collector.get_statistics().values:
                res.append(Tensor(value))
        return res

    def _create_cache(
        self, outputs: List[NodeOutputPort], statistic_points: StatisticPointsContainer
    ) -> Dict[NodeOutputPort, List[Tensor]]:
        """
        Creates a cache for the outputs using the provided statistic points.

        :param outputs: List of node output ports.
        :param statistic_points: Container holding statistic points.
        :return: Dictionary mapping node output ports to lists of tensor data.
        """
        cache = {}
        for out in outputs:
            x = self._get_statistics(statistic_points, out.node_name, out.output_port)
            if not x:
                raise RuntimeError(f"Statistics for {out.node_name} node on {out.output_port} port is not provided.")
            cache[out] = x
        return cache

    def _get_outputs_following_model_inputs(
        self, graph: NNCFGraph, schedule: List[LayerwiseStep]
    ) -> List[NodeOutputPort]:
        """
        Identifies the outputs that follow the model inputs.

        :param graph: The model graph.
        :param schedule: List of layer-wise steps defining the schedule.
        :return: List of node output ports following model inputs.
        """
        input_node_names = [node.node_name for node in graph.get_input_nodes()]
        outputs = set()
        for step in schedule:
            graph_inputs = set()
            for sub_input in step.subgraph_inputs:
                if sub_input.node_name in input_node_names:
                    graph_inputs.add(sub_input)
            num_graph_inputs = len(graph_inputs)
            if num_graph_inputs == len(step.subgraph_inputs):
                outputs |= set(step.subgraph_outputs)
            else:
                for sub_input in graph_inputs:
                    outputs.add(sub_input)
        return list(outputs)

    def create_iterator_through_target_nodes(
        self,
        model: TModel,
        graph: NNCFGraph,
        target_nodes: List[NNCFNode],
        dataset: Dataset,
        statistic_points: Optional[StatisticPointsContainer] = None,
    ) -> LayerwiseIterator:
        """
        Create the iterator through the target nodes of the model.

        :param model: The model to be iterated over.
        :param graph: The model graph.
        :param target_nodes: List of target nodes to iterate through.
        :param dataset: The dataset to be used for obtaining inputs to the model.
        :param statistic_points: Optional container holding statistic points.
        :return: An iterator yielding inputs or outputs for each target node of the model.
        """
        self._set_backend_entity(model)

        schedule = self._scheduler.schedule(graph, target_nodes, self._collect_inputs)
        cache = None
        if statistic_points is not None:
            outputs = self._get_outputs_following_model_inputs(graph, schedule)
            cache = self._create_cache(outputs, statistic_points)
        return self._backend_entity.create_layerwise_iterator(model, graph, schedule, dataset, self._subset_size, cache)

    def get_statistic_points(
        self, model: TModel, graph: NNCFGraph, target_nodes: List[NNCFNode]
    ) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: The model for statistics collection.
        :param graph: The model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """
        self._set_backend_entity(model)

        schedule = self._scheduler.schedule(graph, target_nodes, self._collect_inputs)
        outputs = self._get_outputs_following_model_inputs(graph, schedule)

        statistic_container = StatisticPointsContainer()
        for node_name, output_port_id in outputs:
            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, node_name, port_id=output_port_id
            )
            stat_collector = self._backend_entity.raw_statistic_collector(num_samples=self._subset_size)
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )
        return statistic_container
