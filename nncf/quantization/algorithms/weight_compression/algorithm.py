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
from typing import Dict, List, Optional, Tuple, TypeVar

import numpy as np

from nncf import Dataset
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.scopes import should_consider_scope
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.parameters import CompressWeightsMode
from nncf.parameters import SensitivityMetric
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class WeightCompression(Algorithm):
    """
    Post-training Weight Compression algorithm implementation.

    Compresses weights of Linear and Embedding layers to 8-bit integer or
    to nf4 depending on mode, ratio and group size.
    """

    def __init__(
        self,
        mode: CompressWeightsMode,
        ratio: float,
        group_size: int,
        ignored_scope: IgnoredScope,
        all_layers: bool,
        sensitivity_metric: SensitivityMetric,
    ):
        """
        :param mode: Defines a mode for weight compression.
            INT8_SYM stands for 8-bit integer symmetric quantization of all weights.
                Weights are quantized symmetrically with a fixed zero point equals to 128.
            INT8_ASYM is the same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
                Weights are quantized to a primary precision symmetrically with a fixed zero point equals to 8.
                All embeddings and the last layer are always compressed to a backup precision, which is INT8_ASYM,
                by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
                criteria and the given ratio.
            INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
                with a typical non-fixed zero point.
            NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
        :param ratio: the ratio between primary and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
            and the rest to INT8_ASYM).
        :param group_size: number of weights (e.g. 128) in the channel dimension
            that share quantization parameters (scale). The value -1 means no grouping.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param all_layers: Indicates whether embeddings and last layers should be compressed to a primary
            precision. By default, the backup precision is assigned for the embeddings and last layers.
        :param sensitivity_metric: The sensitivity metric for assigning quantization precision to layers. In order to
            preserve the accuracy of the model, the more sensitive layers receives a higher precision.
        """
        super().__init__()
        self._mode = mode
        self._group_size = group_size
        self._ratio = ratio
        self._ignored_scope = ignored_scope
        self._backend_entity = None
        self._algorithm_key = f"CW_{hash(self)}"
        self._fp_inputs = defaultdict(list)
        self._all_layers = all_layers
        self._sensitivity_metric = sensitivity_metric

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.OPENVINO]

    def _get_fp_inputs(
        self, statistic_points: StatisticPointsContainer, node_name: str, port_id: int
    ) -> List[np.ndarray]:
        """
        Collects floating-point statistics for the given node and port id.

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

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.OPENVINO:
            from nncf.quantization.algorithms.weight_compression.openvino_backend import OVWeightCompressionAlgoBackend

            self._backend_entity = OVWeightCompressionAlgoBackend()
        else:
            raise RuntimeError(
                "Cannot return backend-specific entity because {} is not supported!".format(model_backend.value)
            )

    def _get_activation_node_and_port(self, node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[NNCFNode, int]:
        """
        This method returns the activation layer and corresponding port id for the node.

        :param node: NNCFGraph node for which the activation is sought.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Tuple with the activation node and port id.
        """
        activation_port = self._backend_entity.get_activation_port_id(node, nncf_graph)
        activation_edge = nncf_graph.get_input_edges(node)[activation_port]
        activation_node = activation_edge.from_node
        port_id = activation_edge.output_port_id
        return activation_node, port_id

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        self._set_backend_entity(model)
        self._backend_entity.validate_params(self._mode, self._ignored_scope)
        nodes_to_compress = self._get_nodes_to_compress(graph)

        activations = {}
        if dataset is not None and self._sensitivity_metric != SensitivityMetric.WEIGHT_QUANTIZATION_ERROR:
            activations = self._get_activations(dataset, nodes_to_compress, graph, model)

        transformed_model = self._backend_entity.do_compression(
            model,
            nodes_to_compress,
            self._mode,
            self._ratio,
            self._group_size,
            self._all_layers,
            activations,
            self._sensitivity_metric,
        )
        return transformed_model

    def _get_nodes_to_compress(self, nncf_graph: NNCFGraph) -> List[NNCFNode]:
        """
        Collects nodes in the model's graph corresponding to the layers for weight compression.

        :param nncf_graph: NNCFGraph instance.
        :return: List with the data for each layer.
        """
        weighted_metatypes = self._backend_entity.matmul_metatypes + self._backend_entity.embedding_metatypes
        ordered_nodes_to_compress = []
        ignored_names = list(
            get_ignored_node_names_from_ignored_scope(
                self._ignored_scope, nncf_graph, strict=self._ignored_scope.validate
            )
        )
        for node in nncf_graph.topological_sort():
            is_node_with_weights = self._backend_entity.is_node_with_weights(node)
            is_within_scope = should_consider_scope(node.node_name, ignored_names)
            if node.metatype in weighted_metatypes and is_node_with_weights and is_within_scope:
                ordered_nodes_to_compress.append(node)
        return ordered_nodes_to_compress

    def get_statistic_points(self, model: TModel, graph: NNCFGraph) -> StatisticPointsContainer:
        """
        Returns statistic points, for which StatisticsCollector should collect statistics.

        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: Statistic points, for which StatisticsCollector should collect statistics.
        """

    def _get_activations(
        self, dataset: Dataset, nodes_to_compress: List[NNCFNode], graph: NNCFGraph, model: TModel
    ) -> Dict[str, List[np.ndarray]]:
        """
        Collects input activations for the given nodes on the dataset.

        :param dataset: Dataset to collect values.
        :param nodes_to_compress: List of nodes, whose inputs are collected.
        :param model: Model for statistics collection.
        :param graph: Model graph.
        :return: statistics values itself per node name.
        """
        subset_size = 128

        activations = {}
        _collected_stat_inputs_map = {}
        statistic_container = StatisticPointsContainer()
        all_act_nodes = set()
        act_vs_shared_node_names_mapping = defaultdict(list)
        matmul_metatypes = self._backend_entity.matmul_metatypes
        filtered_nodes = filter(lambda node: node.metatype in matmul_metatypes, nodes_to_compress)
        for node in filtered_nodes:
            act_node, output_port_id = self._get_activation_node_and_port(node, graph)
            act_node_name = act_node.node_name
            if act_node_name in all_act_nodes:
                act_vs_shared_node_names_mapping[act_node_name].append(node.node_name)
                continue
            all_act_nodes.add(act_node_name)
            output_id = (act_node_name, output_port_id)
            _collected_stat_inputs_map[node.node_name] = output_id

            statistic_point = self._backend_entity.target_point(
                TargetType.POST_LAYER_OPERATION, act_node_name, port_id=output_port_id
            )
            inplace_statistics = False

            stat_collector = self._backend_entity.raw_statistic_collector(
                num_samples=subset_size, inplace=inplace_statistics
            )
            statistic_container.add_statistic_point(
                StatisticPoint(
                    target_point=statistic_point, tensor_collector=stat_collector, algorithm=self._algorithm_key
                )
            )

        statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        statistics_aggregator.register_statistic_points(statistic_container)
        statistics_aggregator.collect_statistics(model, graph)

        for node_name, output_id in _collected_stat_inputs_map.items():
            act_node_name, output_port_id = output_id
            x_fp = self._get_fp_inputs(statistic_container, node_name=act_node_name, port_id=output_port_id)
            x_fp = [i.squeeze() for i in x_fp]  # List[tensor(seq_length, hidden_dim)]
            activations[node_name] = x_fp

            for shared_node_name in act_vs_shared_node_names_mapping[act_node_name]:
                activations[shared_node_name] = x_fp

        return activations
