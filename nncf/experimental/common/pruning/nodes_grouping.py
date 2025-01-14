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

from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.experimental.common.graph.netron import save_for_netron
from nncf.experimental.common.pruning.block_hierarchy import BlockHierarchy
from nncf.experimental.common.pruning.propagation_data import ConsumerInfo
from nncf.experimental.common.pruning.propagation_data import ProducerInfo
from nncf.experimental.common.pruning.propagation_data import PropagationGroup
from nncf.experimental.common.pruning.propagation_data import PropagationMask
from nncf.experimental.common.pruning.propagation_data import PruningBlock


@dataclass
class PruningGroup:
    """
    Group of pruning blocks that is obtained after propagation.
    """

    block: PruningBlock
    producers: Set[ProducerInfo]
    consumers: Set[ConsumerInfo]

    def __eq__(self, other: "PruningGroup"):
        return self.block == other.block and self.producers == other.producers and self.consumers == other.consumers

    @classmethod
    def from_propagation_group(cls, group: PropagationGroup) -> "PruningGroup":
        """
        Creates an object by taking all necessary information from PropagationGroup.
        """
        return cls(copy(group.block), group.get_producers(), group.get_consumers())


def get_pruning_groups(
    graph: NNCFGraph,
    pruning_operations_metatypes: PruningOperationsMetatypeRegistry,
    prune_operations_types: List[str],
    dump_dir: Optional[Path] = None,
) -> List[PruningGroup]:
    """
    Determines how nodes of the given types should be pruned: which nodes should be pruned together, along which
    dimension, how many sequent channels with which offset. It's done by initializing PropagationMask's on the
    operations with prunable parameters (producers of pruning) and propagating them through the execution graph.

    :param graph: nncf graph to initialize and propagate masks.
    :param pruning_operations_metatypes: registry with operation metatypes pruning algorithm is aware of, i.e.
        metatypes describing operations with common pruning mask application and propagation properties, e.g.
        IdentityMaskForwardOps unifies operations that propagate pruning masks as is (relu, swish etc.), whereas
        Convolution unifies different convolution operations (conv1d, conv2d, conv3d) which accepts some input masks and
        provide some output masks.
    :param prune_operations_types: types of operations with prunable parameters.
    :param dump_dir: path to the directory for dumping debug files.
    :return: list of groups with parameters of pruning.
    """
    # 1. Initialize masks for producing nodes
    all_nodes_to_prune: List[NNCFNode] = graph.get_nodes_by_types(prune_operations_types)
    roots = {}
    for node in all_nodes_to_prune:
        assert isinstance(node.layer_attributes, (LinearLayerAttributes, ConvolutionLayerAttributes))
        pruning_dim = node.layer_attributes.get_target_dim_for_compression()
        output_tensors_shapes = [x.tensor_shape for x in graph.get_output_edges(node)]
        assert not len(set(output_tensors_shapes)) > 1, node.node_name
        output_tensors_shape = output_tensors_shapes[0]
        # TODO: (107663) generalize by introducing get_output_dim_affected_by_compression for layer_attributes
        target_output_dim_for_compression = len(output_tensors_shape) - 1
        root_group = PropagationGroup(block=PruningBlock(), producers={ProducerInfo(node.node_id, pruning_dim)})
        mask = PropagationMask(dim_groups_map={target_output_dim_for_compression: [root_group]})
        roots[node.node_id] = root_group
        node.attributes["output_mask"] = mask

    def get_attributes_fn(node: NNCFNode) -> Dict[str, Any]:
        result = {"metatype": str(node.metatype.name), "node_id": str(node.node_id)}
        if node.layer_attributes:
            result.update(map(lambda pair: (pair[0], str(pair[1])), node.layer_attributes.__dict__.items()))
        if "output_mask" in node.attributes:
            output_mask = node.attributes["output_mask"]
            if output_mask:
                result["output_mask"] = str(output_mask)
        return result

    try:
        # 2. Propagate masks
        MaskPropagationAlgorithm(graph, pruning_operations_metatypes).mask_propagation()
    finally:
        block_hierarchy = BlockHierarchy(list(roots.values()))
        if dump_dir is not None:
            block_hierarchy.visualize_graph(dump_dir / "latest_block_group_hierarchy.dot")
            save_for_netron(graph, str(dump_dir / "latest_propagated_graph.xml"), get_attributes_fn=get_attributes_fn)

    propagation_groups = block_hierarchy.get_groups_on_leaves()
    not_invalid_groups = filter(lambda group: not group.is_invalid, propagation_groups)
    return [PruningGroup.from_propagation_group(pg) for pg in not_invalid_groups]


def select_largest_groups(pruning_groups: List[PruningGroup]) -> List[PruningGroup]:
    """
    Selects largest pruning groups with larger number of pruning blocks.
    """
    selected_producers = set()
    sorted_groups = sorted(pruning_groups, key=lambda group: len(group.producers), reverse=True)
    result = []
    for group in sorted_groups:
        producers = {producer_info.node_id for producer_info in group.producers}
        if not producers.intersection(selected_producers):
            selected_producers.update(producers)
            result.append(group)
    return result
