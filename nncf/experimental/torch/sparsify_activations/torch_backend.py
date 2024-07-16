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

from typing import Dict, List, TypeVar

import numpy as np
import torch
import torch.nn as nn

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
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
from nncf.torch.utils import training_mode_switcher

TModel = TypeVar("TModel")


class ActivationSparsifier(nn.Module):
    """
    Sparsifies input activations by masking out values around zero.
    """

    def __init__(self, target_sparsity: float, alpha: float = 0.2):
        """
        :param target_sparsity: The target activation sparsity level.
        :param alpha: The exponential moving average decay factor in range (0, 1) for calibrating
            the threshold. A larger alpha will give more weight to the most recent batches.
        """
        super().__init__()
        self.target_sparsity = target_sparsity
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError("The decay factor `alpha` should be in range (0, 1).")
        self.alpha = alpha
        self.register_buffer("running_threshold", torch.tensor(0.0))
        self.register_buffer("num_batches_tracked", torch.tensor(0))
        self.running_threshold: torch.Tensor
        self.num_batches_tracked: torch.Tensor
        self._freeze = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._freeze:
            threshold = self._calculate_threshold(x, self.target_sparsity)
            self._update(threshold)
        mask = torch.le(x.abs(), self.running_threshold)
        x = torch.masked_fill(x, mask, 0.0)
        return x

    def reset_running_stats(self):
        """
        Resets the running threshold and the number of tracked batches to the initial stage.
        """
        self.running_threshold.zero_()
        self.num_batches_tracked.zero_()

    def freeze(self, freeze: bool = True):
        self._freeze = freeze

    def extra_repr(self) -> str:
        return f"target_sparsity={self.target_sparsity}"

    def _calculate_threshold(self, x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Calculates the threshold so that the target sparsity can be achieved.

        :param x: The input tensor.
        :param target_sparsity: The target sparsity level on the input tensor.
        :return: The threshold value.
        """
        # uses numpy's quantile implementation as torch's cannot handle large tensor
        value = np.quantile(
            x.detach().abs().cpu().numpy(),
            q=target_sparsity,
        )
        return torch.tensor(value, device=x.device, dtype=x.dtype)

    def _update(self, threshold: torch.Tensor) -> torch.Tensor:
        """
        Updates the running threshold by exponential moving average with decaying adjustment.
        The updating logic is similar to `pandas.DataFrame.ewm(adjust=True)`.

        :param threshold: The threshold value derived from this batch to update the running threshold.
        :return: The updated running threshold.
        """
        beta = 1.0 - self.alpha
        self.running_threshold = (
            threshold * self.alpha + self.running_threshold * beta * (1 - beta**self.num_batches_tracked)
        ) / (1 - beta ** (self.num_batches_tracked + 1))
        self.num_batches_tracked += 1
        return self.running_threshold


class PTSparsifyActivationsAlgoBackend(SparsifyActivationsAlgoBackend):
    """
    Torch backend for the activation sparsification algorithm.
    """

    SUPPORTED_METATYPES = [om.PTLinearMetatype]

    @property
    def supported_metatypes(self) -> List[type[OperatorMetatype]]:
        return PTSparsifyActivationsAlgoBackend.SUPPORTED_METATYPES

    def get_sparsifiers(self, model: NNCFNetwork) -> List[ActivationSparsifier]:
        """
        Finds all the activation sparsifiers in the model.

        :param model: The model with activation sparsifiers.
        :return: List of activation sparsifiers.
        """
        return [m for m in model.nncf.modules() if isinstance(m, ActivationSparsifier)]

    def insert_sparsifiers(
        self,
        model: NNCFNetwork,
        graph: NNCFGraph,
        target_sparsity_by_node: Dict[NNCFNode, float],
    ) -> NNCFNetwork:
        transformation_layout = TransformationLayout()
        for node, target_sparsity in target_sparsity_by_node.items():
            activation_port_id = self._get_activation_port_id(node, graph)
            sparsifier = ActivationSparsifier(target_sparsity=target_sparsity)
            # temporarily freeze it for model transformation
            sparsifier.freeze(True)
            sparsifier_name = f"activations_sparsifier_{node.node_name.replace('.', '_')}"
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

    def calibrate_sparsifiers(self, model: NNCFNetwork, graph: NNCFGraph, dataset: Dataset) -> NNCFNetwork:
        for sparsifier in self.get_sparsifiers(model):
            sparsifier.reset_running_stats()
            sparsifier.freeze(False)
        with training_mode_switcher(model, is_training=False):
            with torch.no_grad():
                self.do_inference(model, dataset)
        return model

    def freeze_sparsifiers(self, model: NNCFNetwork, graph: NNCFGraph) -> NNCFNetwork:
        for sparsifier in self.get_sparsifiers(model):
            sparsifier.freeze(True)
        model.nncf.rebuild_graph()
        return model

    def _get_activation_port_id(self, node: NNCFNode, graph: NNCFGraph) -> int:
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
