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

from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn as nn

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import SparsifyActivationsAlgoBackend
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork

ACTIVATIONS_SPARSIFIER_PREFIX = "activations_sparsifier"


class ActivationsSparsifier(nn.Module):
    """
    Sparsifies input activations by masking out values around zero.
    """

    def __init__(self, threshold: float):
        """ """
        super().__init__()
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))
        self.threshold: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.le(x.abs(), self.threshold)
        x = torch.masked_fill(x, mask, 0.0)
        return x

    def extra_repr(self) -> str:
        return f"target_sparsity={self.threshold}"


class PTSparsifyActivationsAlgoBackend(SparsifyActivationsAlgoBackend):
    """
    Torch backend for the activation sparsification algorithm.
    """

    @property
    def supported_metatypes(self) -> List[Type[OperatorMetatype]]:
        return [om.PTLinearMetatype]

    def abs_quantile_reducer(self, quantile: Optional[Union[float, List[float]]] = None) -> AbsQuantileReducer:
        return AbsQuantileReducer(quantile=quantile)

    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return PTTargetPoint(TargetType.PRE_LAYER_OPERATION, target_node_name, input_port_id=port_id)

    def insert_sparsifiers(
        self,
        model: NNCFNetwork,
        graph: NNCFGraph,
        threshold_by_node: Dict[NNCFNode, float],
    ) -> NNCFNetwork:
        transformation_layout = PTTransformationLayout()
        for node, threshold in threshold_by_node.items():
            activation_port_id = self.get_activation_port_id(node, graph)
            sparsifier = ActivationsSparsifier(threshold=threshold)
            sparsifier_name = f"{ACTIVATIONS_SPARSIFIER_PREFIX}_{node.node_name.replace('.', '_')}"
            transformation_layout.register(
                PTSharedFnInsertionCommand(
                    [
                        PTTargetPoint(
                            target_type=TargetType.PRE_LAYER_OPERATION,
                            target_node_name=node.node_name,
                            input_port_id=activation_port_id,
                        )
                    ],
                    sparsifier,
                    sparsifier_name,
                )
            )

        transformed_model = PTModelTransformer(model).transform(transformation_layout)
        return transformed_model

    @staticmethod
    def get_activation_port_id(node: NNCFNode, graph: NNCFGraph) -> int:
        """
        Finds the input activation port id for the node.

        :param node: The node to find its activation port id.
        :param graph: The NNCF graph containing the node.
        :return: The activation port id.
        """
        activation_ports = []
        for prev_node in graph.get_previous_nodes(node):
            edge = graph.get_edge(prev_node, node)
            if prev_node.metatype in CONST_NOOP_METATYPES or edge.input_port_id in node.metatype.weight_port_ids:
                continue
            activation_ports.append(edge.input_port_id)
        if len(activation_ports) != 1:
            raise nncf.InternalError(f'Cannot find activation port for node "{node}".')
        return activation_ports[0]
