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
from typing import Optional, Type

import torch

from nncf.common.graph import NNCFGraph
from nncf.common.logging import nncf_logger
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.torch.nncf_network import NNCFNetwork


class FilterReorderingAlgorithm(MaskPropagationAlgorithm):
    """
    Reorders filters based on reordering indexes encoded in the `output_mask` attribute in the nodes of
    model graph.
    """

    def __init__(
        self,
        model: NNCFNetwork,
        graph: NNCFGraph,
        pruning_operator_metatypes: PruningOperationsMetatypeRegistry,
        tensor_processor: Optional[Type[NNCFPruningBaseTensorProcessor]] = None,
    ):
        super().__init__(graph, pruning_operator_metatypes, tensor_processor)
        self._model = model

    def apply_reordering_indexes(self) -> None:
        """
        Applying propagated masks (which encodes indexes to reorder filters) for all nodes in topological order:
        1. running input_reorder method for this node
        2. running output_reorder method for this node
        """
        pruned_node_modules = []
        with torch.no_grad():
            for node in self._graph.topological_sort():
                node_cls = self.get_meta_operation_by_type_name(node.node_type)
                node_module = self._model.nncf.get_containing_module(node.node_name)
                if node_module not in pruned_node_modules:
                    node_cls.input_reorder(self._model, node, self._graph)
                    node_cls.output_reorder(self._model, node, self._graph)
                    pruned_node_modules.append(node_module)
            nncf_logger.debug("Finished mask applying step")

    def reorder_filters(self) -> None:
        """
        Model pruner work in two stages:
        1. Mask propagation: propagate pruning masks through the graph.
        2. Applying calculated masks
        """
        nncf_logger.info("Start reordering filters")
        self.mask_propagation()
        self.apply_reordering_indexes()
        nncf_logger.info("Finished reordering filters")
