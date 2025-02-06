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

from typing import Dict, List, Type, TypeVar

import torch
import torch.nn as nn

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.data import Dataset
from nncf.experimental.torch.sparsify_activations.sparsify_activations_impl import SparsifyActivationsAlgoBackend
from nncf.tensor.functions.torch_numeric import quantile
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.utils import training_mode_switcher

ACTIVATIONS_SPARSIFIER_PREFIX = "activations_sparsifier"
TModel = TypeVar("TModel")


class ActivationsSparsifier(nn.Module):
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
        self.register_buffer("running_threshold", torch.tensor(float("-inf")))
        self.register_buffer("num_batches_tracked", torch.tensor(0))
        self.running_threshold: torch.Tensor
        self.num_batches_tracked: torch.Tensor
        self._freeze = True

    @staticmethod
    def calculate_threshold(x: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        """
        Calculates the threshold to sparsify the input tensor with target sparsity if locations of
        `x.abs() <= threshold` are zeroed out.

        :param x: The input tensor.
        :param target_sparsity: The target sparsity level on the input tensor.
        :return: The threshold value.
        """
        return quantile(x.detach().abs().view(-1), q=target_sparsity, axis=0)

    @property
    def freeze(self):
        return self._freeze

    @freeze.setter
    def freeze(self, value: bool):
        self._freeze = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.freeze:
            threshold = self.calculate_threshold(x, self.target_sparsity)
            self._update(threshold, dtype=x.dtype)
        mask = torch.le(x.abs(), self.running_threshold)
        x = torch.masked_fill(x, mask, 0.0)
        return x

    def reset_running_stats(self):
        """
        Resets the running threshold and the number of tracked batches to the initial stage.
        """
        self.running_threshold.fill_(float("-inf"))
        self.num_batches_tracked.zero_()

    def extra_repr(self) -> str:
        return f"target_sparsity={self.target_sparsity}"

    def _update(self, threshold: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Updates the running threshold by exponential moving average with decaying adjustment.
        The updating logic is similar to `pandas.DataFrame.ewm(adjust=True)`.

        :param threshold: The threshold value derived from this batch to update the running threshold.
        :param dtype: Data type of the updated running threshold.
        :return: The updated running threshold.
        """
        if self.num_batches_tracked == 0:
            running_threshold = threshold
        else:
            beta = 1.0 - self.alpha
            old_running_threshold = self.running_threshold.to(device=threshold.device, dtype=torch.float64)
            running_threshold = (
                threshold.to(torch.float64) * self.alpha
                + old_running_threshold * beta * (1 - beta**self.num_batches_tracked)
            ) / (1 - beta ** (self.num_batches_tracked + 1))
        self.running_threshold = running_threshold.type(dtype)
        self.num_batches_tracked += 1
        return self.running_threshold


class PTSparsifyActivationsAlgoBackend(SparsifyActivationsAlgoBackend):
    """
    Torch backend for the activation sparsification algorithm.
    """

    SUPPORTED_METATYPES = [om.PTLinearMetatype]

    @staticmethod
    def get_sparsifiers(model: NNCFNetwork) -> List[ActivationsSparsifier]:
        """
        Finds all the activation sparsifiers in the model.

        :param model: The model with activation sparsifiers.
        :return: List of activation sparsifiers.
        """
        return [m for m in model.nncf.modules() if isinstance(m, ActivationsSparsifier)]

    @property
    def supported_metatypes(self) -> List[Type[OperatorMetatype]]:
        return PTSparsifyActivationsAlgoBackend.SUPPORTED_METATYPES

    def insert_sparsifiers(
        self,
        model: NNCFNetwork,
        graph: NNCFGraph,
        target_sparsity_by_node: Dict[NNCFNode, float],
    ) -> NNCFNetwork:
        transformation_layout = PTTransformationLayout()
        for node, target_sparsity in target_sparsity_by_node.items():
            activation_port_id = self._get_activation_port_id(node, graph)
            sparsifier = ActivationsSparsifier(target_sparsity=target_sparsity)
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

    def calibrate_sparsifiers(self, model: NNCFNetwork, graph: NNCFGraph, dataset: Dataset) -> NNCFNetwork:
        sparsifiers = self.get_sparsifiers(model)
        for sparsifier in sparsifiers:
            sparsifier.reset_running_stats()
            sparsifier.freeze = False
        with training_mode_switcher(model, is_training=False):
            with torch.no_grad():
                self.do_inference(model, dataset)
        for sparsifier in sparsifiers:
            sparsifier.freeze = True
        return model

    @staticmethod
    def _get_activation_port_id(node: NNCFNode, graph: NNCFGraph) -> int:
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
