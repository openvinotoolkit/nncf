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
from abc import abstractmethod
from typing import Dict, List, Optional, Type, TypeVar, Union

import nncf
from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.scopes import should_consider_scope
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data import Dataset
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.torch.sparsify_activations.ema_aggregator import EMAAggregator
from nncf.experimental.torch.sparsify_activations.target_scope import TargetScope
from nncf.experimental.torch.sparsify_activations.target_scope import get_target_node_names_from_target_scope
from nncf.scopes import IgnoredScope
from nncf.scopes import get_ignored_node_names_from_ignored_scope
from nncf.torch.model_creation import is_wrapped_model
from nncf.torch.model_creation import wrap_model

TModel = TypeVar("TModel")
STATISTIC_BRANCH_KEY = "abs_quantile"
ALGORITHM_KEY = "AS"


class SparsifyActivationsAlgoBackend(ABC):
    """
    Abstract class for activation sparsification algorithm backend.
    """

    @property
    @abstractmethod
    def supported_metatypes(self) -> List[Type[OperatorMetatype]]:
        """
        Property for the backend-specific metatypes for supported layers.
        """

    @abstractmethod
    def abs_quantile_reducer(self, quantile: Optional[Union[float, List[float]]] = None) -> AbsQuantileReducer:
        """ """

    @abstractmethod
    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> TargetPoint:
        """ """

    @abstractmethod
    def insert_sparsifiers(
        self,
        model: TModel,
        graph: NNCFGraph,
        threshold_by_node: Dict[NNCFNode, float],
    ) -> TModel:
        """
        Inserts the activation sparsifiers to the model.

        :param model: The model to conduct activation sparsification.
        :param graph: The model's NNCF graph.
        :param target_sparsity_by_node: The target sparsity level for the input activation in each given node layer.
        :return: The model with inserted activation sparsifiers.
        """

    @staticmethod
    @abstractmethod
    def get_activation_port_id(node: NNCFNode, graph: NNCFGraph) -> int:
        """
        Finds the input activation port id for the node.

        :param node: The node to find its activation port id.
        :param graph: The NNCF graph containing the node.
        :return: The activation port id.
        """


class SparsifyActivationsAlgorithm:
    """
    Implementation of activation sparsification algorithm.
    """

    def __init__(
        self,
        target_sparsity_by_scope: Dict[TargetScope, float],
        ignored_scope: IgnoredScope,
    ):
        """
        :param target_sparsity_by_scope: A dictionary that defines the target sparsity level for specified layers.
        :param ignored_scope: An ignored scope that defines the list of model control flow
            graph nodes to be ignored during activation sparsification.
        """
        self._target_sparsity_by_scope = target_sparsity_by_scope
        self._ignored_scope = ignored_scope
        self._backend_entity: SparsifyActivationsAlgoBackend = None

    @property
    def available_backends(self) -> List[BackendType]:
        """
        Supported backends for this algorithm.
        """
        return [BackendType.TORCH, BackendType.OPENVINO]

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        dataset: Dataset,
    ) -> TModel:
        """
        Applies the algorithm to the given model.

        :param model: The model to be sparsified.
        :param graph: The model's NNCF graph.
        :param dataset: The dataset to calibrate the activation sparsifiers.
        :return: The sparsified model.
        """
        self._set_backend_entity(model)
        target_sparsity_by_node = self._get_target_sparsity_by_node(graph)
        threshold_by_node = self._get_threshold_by_node(model, graph, target_sparsity_by_node, dataset)
        sparse_model = self._backend_entity.insert_sparsifiers(model, graph, threshold_by_node)
        return sparse_model

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backend-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.TORCH:
            from nncf.experimental.torch.sparsify_activations.torch_backend import PTSparsifyActivationsAlgoBackend

            self._backend_entity = PTSparsifyActivationsAlgoBackend()
        elif model_backend == BackendType.OPENVINO:
            from nncf.experimental.torch.sparsify_activations.openvino_backend import OVSparsifyActivationsAlgoBackend

            self._backend_entity = OVSparsifyActivationsAlgoBackend()
        else:
            raise nncf.UnsupportedBackendError(
                f"{model_backend.value} backend is not supported for `sparsify_activations`."
            )

    def _get_target_sparsity_by_node(self, graph: NNCFGraph) -> Dict[NNCFNode, float]:
        """
        Collects nodes in the model's graph corresponding to the layers for sparsification.

        :param graph: NNCFGraph instance.
        :return: A dictionary with nodes and the corresponding target sparsity level.
        """
        supported_metatypes = self._backend_entity.supported_metatypes
        ignored_names = get_ignored_node_names_from_ignored_scope(
            self._ignored_scope, graph, strict=self._ignored_scope.validate
        )
        target_sparsity_by_node = {}
        for scope, target_sparsity in self._target_sparsity_by_scope.items():
            target_names = get_target_node_names_from_target_scope(scope, graph, strict=scope.validate)
            for node_name in target_names:
                node = graph.get_node_by_name(node_name)
                if node.metatype not in supported_metatypes or not should_consider_scope(
                    node.node_name, ignored_scopes=ignored_names
                ):
                    continue
                if node in target_sparsity_by_node:
                    raise nncf.ValidationError(
                        f'"{node.node_name}" is matched by multiple items in `target_sparsity_by_scope`.'
                    )
                target_sparsity_by_node[node] = target_sparsity
        if not target_sparsity_by_node:
            raise nncf.ValidationError("No layers to conduct activation sparsification.")
        return target_sparsity_by_node

    def _get_threshold_by_node(self, model, graph, target_sparsity_by_node, dataset: Dataset) -> Dict[NNCFNode, float]:
        statistic_points_container = StatisticPointsContainer()
        for node, sparsity in target_sparsity_by_node.items():
            stat_collector = TensorCollector()
            stat_collector.register_statistic_branch(
                container_key=STATISTIC_BRANCH_KEY,
                reducer=self._backend_entity.abs_quantile_reducer(
                    quantile=[
                        sparsity,
                    ]
                ),
                aggregator=EMAAggregator(alpha=0.2),
            )
            activation_port_id = self._backend_entity.get_activation_port_id(node, graph)
            statistic_point = StatisticPoint(
                target_point=self._backend_entity.target_point(
                    TargetType.PRE_LAYER_OPERATION, node.node_name, port_id=activation_port_id
                ),
                tensor_collector=stat_collector,
                algorithm=ALGORITHM_KEY,
            )
            statistic_points_container.add_statistic_point(statistic_point)

        statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        statistics_aggregator.register_statistic_points(statistic_points_container)
        statistics_aggregator.collect_statistics(model, graph)

        threshold_by_node = {}
        for nncf_node in target_sparsity_by_node:
            tensor_collector = next(
                iter(
                    statistic_points_container.get_algo_statistics_for_node(
                        nncf_node.node_name, lambda args: True, ALGORITHM_KEY
                    )
                )
            )
            threshold_by_node[nncf_node] = tensor_collector.get_statistics()[STATISTIC_BRANCH_KEY].data

        return threshold_by_node


def sparsify_activations(
    model: TModel,
    dataset: Dataset,
    target_sparsity_by_scope: Dict[TargetScope, float],
    ignored_scope: Optional[IgnoredScope] = None,
) -> TModel:
    """
    Post-training activation sparsification on the given model.

    This algorithm sparsifies the input activations in supported layers based on a calibration
    dataset. The goal is to zero out neurons with small activation values around 0, thereby
    roughly achieving the target sparsity at a statistical level.

    Note that currently only linear layers are supported.

    :param model: The model to be sparsified.
    :param dataset: The dataset to calibrate the activation sparsifiers.
    :param target_sparsity_by_scope: Defines the target activation sparsity level
        for specified layers. For each item, the key is an instance of `TargetScope` class
        representing the layers to match in the model's NNCF graph; the corresponding value
        is a float number in the range [0, 1] representing the target sparsity level.

        Example:
        ..  code-block:: python
            {
                # Target sparsity is 60% for node "Dummy/Linear[layer]/linear_0" in the model graph
                TargetScope(names=["Dummy/Linear[layer]/linear_0"]): 0.6,
                # Target sparsity is 30% for the layers whose name contains "up_proj" or "down_proj".
                TargetScope(patterns=[".*up_proj.*", ".*down_proj.*"]): 0.3,
            }

    :param ignored_scope: Optional. It defines the nodes in the model graph that should be
        ignored during activation sparsification. Note that unsupported layer types are already
        filtered out internally, so there is no need to mention them in `ignored_scope`.
    :return: The sparsified model.
    """

    for scope, target_sparsity in target_sparsity_by_scope.items():
        if target_sparsity < 0.0 or target_sparsity > 1.0:
            raise ValueError(f'Target sparsity for scope "{scope}" should be in range [0, 1].')

    if ignored_scope is None:
        ignored_scope = IgnoredScope()

    backend = get_backend(model)
    if backend == BackendType.TORCH and not is_wrapped_model(model):
        example_input = next(iter(dataset.get_inference_data()))
        model = wrap_model(
            model,
            example_input=example_input,
            trace_parameters=True,
        )

    algorithm = SparsifyActivationsAlgorithm(target_sparsity_by_scope, ignored_scope)

    graph = NNCFGraphFactory.create(model)
    sparse_model = algorithm.apply(model, graph, dataset)
    return sparse_model
