from pathlib import Path
from typing import List, Tuple, TypeVar

import torch
import torch.nn as nn

import nncf
import nncf.experimental.torch.sparsify_activations
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.common.utils.backend import get_backend
from nncf.data import Dataset
from nncf.experimental.tensor.tensor import Tensor
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import SparsifyActivationsAlgorithmBackend
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


class ActivationSparsifier(nn.Module):
    def __init__(self, target_sparsity: float, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.register_buffer('target_sparsity', torch.tensor(target_sparsity))
        self.register_buffer("running_threshold", torch.tensor(0.))
        self.register_buffer("num_batches_tracked", torch.tensor(0))
        self.running_threshold: torch.Tensor
        self.num_batches_tracked: torch.Tensor
        self._frozen = False

    def forward(self, x):
        threshold = None
        if not self._frozen:
            threshold = self._calculate_quantile(x.abs(), self.target_sparsity)
            self._update(threshold)
        assert self.num_batches_tracked > 0
        mask = torch.le(x.abs(), self.running_threshold)
        if '[1]' in self.node_name and 'up_proj' in self.node_name:
            print('sparsity', mask.float().mean(), 'cur_threshold',
                  threshold, 'threshold', self.running_threshold)
        x = torch.masked_fill(x, mask, 0.)
        return x

    def reset_running_stats(self):
        self.running_threshold.zero_()
        self.num_batches_tracked.zero_()

    def freeze(self, freeze: bool = True):
        self._frozen = freeze

    def extra_repr(self) -> str:
        return f"target_sparsity={self.target_sparsity.item()}"

    def _calculate_quantile(self, x: torch.Tensor, target_sparsity: float):
        return x.view(-1).quantile(q=target_sparsity, dim=-1)

    def _update(self, threshold: torch.Tensor):
        beta = 1.0 - self.alpha
        # Exponential Moving Average with decaying adjustment, similar to pandas.DataFrame.ewm(adjust=True).
        self.running_threshold = (
            threshold * self.alpha +
            self.running_threshold * beta *
            (1 - beta ** self.num_batches_tracked)
        ) / (1 - beta ** (self.num_batches_tracked + 1))
        self.num_batches_tracked += 1
        return self.running_threshold


def node_name_matches_module_name(node_name, module_name):
    parts = module_name.split('.')
    return all(f'[{p}]' in node_name for p in parts)


class PTPruneActivationAlgorithmBackend(SparsifyActivationsAlgorithmBackend):
    SUPPORTED_METATYPES = [om.PTLinearMetatype]

    def __init__(self) -> None:
        pass

    def do_sparsification(self):
        pass

    @property
    def supported_metatypes(self) -> List[OperatorMetatype]:
        return PTPruneActivationAlgorithmBackend.SUPPORTED_METATYPES

    @staticmethod
    def get_activation_port_id(node: NNCFNode, graph: NNCFGraph) -> NNCFNode:
        activation_ports = []
        for prev_node in graph.get_previous_nodes(node):
            if 'weight' in prev_node.node_name.lower() or 'bias' in prev_node.node_name:
                # TODO: find activation
                continue
            edge = graph.get_edge(prev_node, node)
            activation_ports.append(edge.input_port_id)
        assert len(activation_ports) == 1
        return activation_ports[0]

    @staticmethod
    def inference(model: nn.Module, dataset: Dataset) -> None:
        model = model.eval()
        with torch.no_grad():
            for batch in dataset.get_inference_data():
                model(**batch)

    def do_sparsification(self, model, graph: NNCFGraph, nodes_to_sparsify: List[NNCFNode]):
        transformation_layout = TransformationLayout()
        activation_sparsifiers = []

        for node in nodes_to_sparsify:
            activation_node, activation_port_id = self._get_activation_node_and_port(
                node, graph)
            target_sparsity = None
            for module_name in self.sparse_config:
                if node_name_matches_module_name(node.node_name, module_name):
                    target_sparsity = self.sparse_config[module_name]
                    break
            if target_sparsity is None:
                continue
            activation_sparsifier = ActivationSparsifier(
                target_sparsity=target_sparsity)
            activation_sparsifier.node_name = node.node_name
            activation_sparsifiers.append(activation_sparsifier)
            activation_sparsifier_name = f"activation_sparsifier_{node.node_name.replace('.', '_')}"
            transformation_layout.register(PTSharedFnInsertionCommand(
                [
                    PTTargetPoint(
                        TargetType.PRE_LAYER_OPERATION,
                        target_node_name=node.node_name,
                        input_port_id=activation_port_id)
                ],
                activation_sparsifier,
                activation_sparsifier_name,
            ))

        transformed_model = PTModelTransformer(
            model).transform(transformation_layout)
        return transformed_model, activation_sparsifiers
