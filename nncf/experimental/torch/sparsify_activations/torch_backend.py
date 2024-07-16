from typing import List, Tuple, TypeVar

import torch
import torch.nn as nn

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.data import Dataset
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import SparsifyActivationsAlgoBackend
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork

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
        self._frozen = True

    def forward(self, x):
        threshold = None
        if not self._frozen:
            threshold = self._calculate_quantile(x.abs(), self.target_sparsity)
            self._update(threshold)
        mask = torch.le(x.abs(), self.running_threshold)
        # if '[1]' in self.node_name and 'up_proj' in self.node_name:
        #     print('sparsity', mask.float().mean(), 'cur_threshold',
        #           threshold, 'threshold', self.running_threshold)
        x = torch.masked_fill(x, mask, 0.)
        return x

    def reset_running_stats(self):
        self.running_threshold.zero_()
        self.num_batches_tracked.zero_()

    def freeze(self, freeze: bool = True):
        self._frozen = freeze

    def extra_repr(self) -> str:
        return f"target_sparsity={self.target_sparsity.item()},{self.running_threshold:=},{self.num_batches_tracked}"

    def _calculate_quantile(self, x: torch.Tensor, target_sparsity: float):
        return x.float().view(-1).quantile(q=target_sparsity, dim=-1)

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


class PTSparsifyActivationsAlgoBackend(SparsifyActivationsAlgoBackend):
    SUPPORTED_METATYPES = [om.PTLinearMetatype]

    @property
    def supported_metatypes(self) -> List[OperatorMetatype]:
        return PTSparsifyActivationsAlgoBackend.SUPPORTED_METATYPES

    def insert_sparsifiers(self, model, graph: NNCFGraph, target_sparsity_by_node: dict[NNCFNode, float]) -> NNCFNetwork:
        transformation_layout = TransformationLayout()
        for node, target_sparsity in target_sparsity_by_node.items():
            act_node, act_port_id = self._get_activation_node_and_port(
                node, graph)
            sparsifier = ActivationSparsifier(target_sparsity=target_sparsity)
            sparsifier_name = f"activation_sparsifier_{node.node_name.replace('.', '_')}"
            transformation_layout.register(PTSharedFnInsertionCommand(
                [
                    PTTargetPoint(
                        TargetType.PRE_LAYER_OPERATION,
                        target_node_name=node.node_name,
                        input_port_id=act_port_id)
                ],
                sparsifier,
                sparsifier_name,
            ))

        transformed_model = PTModelTransformer(
            model).transform(transformation_layout)
        return transformed_model

    def get_sparsifiers(self, model: NNCFNetwork) -> List[ActivationSparsifier]:
        return [m for m in model.nncf.modules() if isinstance(m, ActivationSparsifier)]

    def calibrate_sparsifiers(self, model: NNCFNetwork, dataset: Dataset) -> NNCFNetwork:
        for sparsifier in self.get_sparsifiers(model):
            sparsifier.reset_running_stats()
            sparsifier.freeze(False)
        self.do_inference(model, dataset)
        return model

    def freeze_sparsifiers(self, model: NNCFNetwork) -> NNCFNetwork:
        for sparsifier in self.get_sparsifiers(model):
            sparsifier.freeze(True)
        model.nncf.rebuild_graph()
        return model

    def _get_activation_port_id(self, node: NNCFNode, graph: NNCFGraph) -> NNCFNode:
        activation_ports = []
        for prev_node in graph.get_previous_nodes(node):
            if 'weight' in prev_node.node_name.lower() or 'bias' in prev_node.node_name:
                # TODO: find activation
                continue
            edge = graph.get_edge(prev_node, node)
            activation_ports.append(edge.input_port_id)
        assert len(activation_ports) == 1
        return activation_ports[0]

    def _get_activation_node_and_port(self, node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[NNCFNode, int]:
        """
        This method returns the activation layer and corresponding port id for the node.

        :param node: NNCFGraph node for which the activation is sought.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Tuple with the activation node and port id.
        """
        activation_port = self._get_activation_port_id(node, nncf_graph)
        activation_edge = nncf_graph.get_input_edges(node)[activation_port]
        activation_node = activation_edge.from_node
        port_id = activation_edge.output_port_id
        return activation_node, port_id
