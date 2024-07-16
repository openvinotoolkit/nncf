from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple, TypeVar

import torch
import torch.nn as nn

import nncf
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.scopes import should_consider_scope
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data import Dataset
from nncf.experimental.tensor.tensor import Tensor
from nncf.quantization.algorithms.algorithm import Algorithm
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_creation import is_wrapped_model
from nncf.torch.model_creation import wrap_model
from nncf.torch.model_graph_manager import find_const_node_in_constant_subgraph
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.utils import is_tracing_state

TModel = TypeVar("TModel")


def node_name_matches_module_name(node_name, module_name):
    parts = module_name.split('.')
    return all(f'[{p}]' in node_name for p in parts)


class SparsifyActivationsAlgorithmBackend(ABC):
    @property
    @abstractmethod
    def supported_metatypes(self) -> List[OperatorMetatype]:
        pass

    @abstractmethod
    def do_sparsification(self, model, graph: NNCFGraph, nodes_to_sparsify: List[NNCFNode]):
        pass

    def transform_model(self,):
        pass


class SparsifyActivationsAlgorithm:

    def __init__(self, sparse_config: dict[str, float]):
        self.sparse_config = sparse_config

    @property
    def available_backends(self) -> List[BackendType]:
        return [BackendType.TORCH]

    def _set_backend_entity(self, model: TModel) -> None:
        """
        Creates a helper class with a backed-specific logic of the algorithm.

        :param model: Backend-specific input model.
        """
        model_backend = get_backend(model)
        if model_backend == BackendType.TORCH:
            from nncf.experimental.torch.sparsify_activations.torch_backend import PTPruneActivationAlgorithmBackend
            self._backend_entity = PTPruneActivationAlgorithmBackend()
        else:
            raise nncf.UnsupportedBackendError(
                f"{model_backend.value} backend is not supported for `sparsify_activations`."
            )

    def _get_nodes_to_sparsify(self, nncf_graph: NNCFGraph) -> List[NNCFNode]:
        """
        Collects nodes in the model's graph corresponding to the layers for weight compression.

        :param nncf_graph: NNCFGraph instance.
        :return: List with the data for each layer.
        """
        supported_metatypes = self._backend_entity.supported_metatypes
        ordered_nodes_to_sparsify = []
        for node in nncf_graph.topological_sort():
            if node.metatype in supported_metatypes:
                ordered_nodes_to_sparsify.append(node)
        return ordered_nodes_to_sparsify

    def _get_activation_node_and_port(self, node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[NNCFNode, int]:
        """
        This method returns the activation layer and corresponding port id for the node.

        :param node: NNCFGraph node for which the activation is sought.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Tuple with the activation node and port id.
        """
        activation_port = self._backend_entity.get_activation_port_id(
            node, nncf_graph)
        activation_edge = nncf_graph.get_input_edges(node)[activation_port]
        activation_node = activation_edge.from_node
        port_id = activation_edge.output_port_id
        return activation_node, port_id

    def apply(
        self,
        model: TModel,
        graph: NNCFGraph,
        dataset: Dataset,
    ) -> TModel:
        self._set_backend_entity(model)
        nodes_to_sparsify = self._get_nodes_to_sparsify(graph)
        transformed_model, activation_sparsifiers = self._backend_entity.do_sparsification(
            model, graph, nodes_to_sparsify
        )
        for sparsifier in activation_sparsifiers:
            sparsifier.reset_running_stats()
        self._backend_entity.inference(transformed_model, dataset)
        for sparsifier in activation_sparsifiers:
            sparsifier.freeze(True)
        transformed_model.nncf.rebuild_graph()
        return transformed_model


def sparsify_activations(
    model: TModel,
    dataset: Dataset,
    sparsity_config: dict[str, float],
    debug_folder=None,
) -> TModel:
    """
    Implementation of the `compress_weights()` method.
    """

    backend = get_backend(model)
    if backend == BackendType.TORCH and not is_wrapped_model(model):
        example_input = next(iter(dataset.get_inference_data()))
        model = wrap_model(
            model,
            example_input=example_input,
            trace_parameters=True,
        )

    algorithm = SparsifyActivationsAlgorithm(sparsity_config)
    graph = NNCFGraphFactory.create(model)
    if debug_folder:
        graph.dump_graph(
            Path(debug_folder, './before-sparsification.dot').as_posix())
    sparse_model = algorithm.apply(model, graph, dataset)
    graph = NNCFGraphFactory.create(sparse_model)
    if debug_folder:
        graph.dump_graph(
            Path(debug_folder, './after-sparsification.dot').as_posix())
    return sparse_model
