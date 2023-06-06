"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from typing import Callable, List, Optional

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.shape_pruning_processor import ShapePruningProcessor
from nncf.common.pruning.structs import PrunedLayerInfoBase
from nncf.common.pruning.utils import get_input_masks
from nncf.common.tensor import NNCFTensor


class NASShapePruningProcessor(ShapePruningProcessor):
    """
    BootstrapNAS shape pruning function.
    """

    def __init__(
        self,
        prunable_types: List[str],
        pruning_operations_metatype: List[str],
        get_input_masks_func: Callable[[NNCFNode, NNCFGraph], List[Optional[NNCFTensor]]] = get_input_masks,
    ):
        super().__init__(prunable_types, pruning_operations_metatype)
        self.get_input_masks_func = get_input_masks_func

    def _get_next_node_sparse_multiplier(
        self, graph: NNCFGraph, next_node: NNCFNode, cluster: Clusterization[PrunedLayerInfoBase]
    ) -> int:
        cluster_nodes_idxs = {node.nncf_node_id for node in cluster.elements}
        for input_mask in self.get_input_masks_func(next_node, graph):
            if not input_mask:
                continue
            for mask_producer in input_mask.mask_producers:
                if mask_producer.id in cluster_nodes_idxs:
                    return mask_producer.sparse_multiplier
        raise RuntimeError(f"Next node for cluster {cluster.elements} doesn't have closing mask")
