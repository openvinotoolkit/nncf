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

import json
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Dict
from typing import List
from typing import Set

import networkx as nx
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.utils.debug import DEBUG_LOG_DIR
from nncf.torch.nested_objects_traversal import objwalk
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.experimental.common.graph.netron import save_for_netron


class MaskProducer:
    """
    Defines producer of the pruning.
    """

    def __init__(self, id_) -> None:
        """
        :param id_: identification of the producer node in the NNCFGraph.
        """
        self.id = id_


class PropagationBlock:
    """
    Defines a pruning block - how much and which particular channels are supposed to be pruned at the same time for
    the node when a single element of pruning mask is 0. We assume that the pruning mask is a vector with 1's and 0's.
    1 retains the corresponding set of channels in weights, 0 prunes it.
    The block is initialized on producers of pruning and is propagated until consumers within PropagationGroup and
    PropagationMask.
    Essentially, the propagation block is parametrized by size and offset. Size is the number of sequent channels to be
    included to the block. When offset is not equal to 0, the block is formed by taking the `size` number of
    sequent channels, then skipping the `offset` number of channels and repeating the procedure for the rest channels.

    Let's consider 3 possible examples.
    Notation:
    "-" is not a pruned channel
    "x" is a channel that is included to the block or pruned

    1) By default, a single element of structured pruning mask corresponds to a single channel in the producer.
    Size is equal to 1, offset is 0. It means that block consist of just 1 channel.

    size = 1
    offset = 0
    number of all channels = 18

    all channels                        ------------------
    block #1, mask: 011111111111111111  x-----------------

    2) But in some cases, pruning can be performed by set of channels, rather than individual one.
    For instance, head pruning in the Transformers is removal of N sequent channels, when N is the size of the
    head. In that case, the propagation block is encoded by size=N and offset=0.

    size = 6 (head size)
    offset = 0
    number of all channels = 18

    all channels           ------------------
    block #1, mask: 011    xxxxxx------------
    block #2, mask: 101    ------xxxxxx------
    block #3, mask: 110    ------------xxxxxx
    mask: 100              ------xxxxxxxxxxxx

    3) Or potentially, one can remove the same dimension in each Transformer's head. The propagation block would be as
    follows: size=1 and offset=N. It means that block is formed by taking the `size` number of sequent channels, then
    skipping the `offset` number of sequent channels and repeating the procedure for the rest channels. For that
    particular case, if total number of channels is 3N, the size of block will be equal to 3.
    Block will contain 6 elements, if size=2 offset=N with the same size of pruning mask.

    size = 1
    offset = 6 (head size)
    number of all channels = 18

    all channels              ------------------
    block #1, mask: 011111    x-----x-----x-----
    block #2, mask: 101111    -x-----x-----x----
    ...
    block #6, mask: 111110    -----x-----x-----x
    mask 110000               --xxxx--xxxx--xxxx
    """
    def __init__(self,
                 producer: MaskProducer,
                 size: int = 1,
                 offset: int = 0,
                 pruning_dimension: int = 0,
                 closed_branches: int = 0) -> None:
        """

        :param producer: descriptor of the producer
        :param size: number of sequent channels.
        :param offset: when not equal to 0, block is formed by taking `size` number of sequent channels,
        then skipping `offset` number of sequent channels and repeating the procedure for the rest of channels.
        :param pruning_dimension: axis number from 0 to N-1 in weights along which the dimension block defines pruning
        structure. N is total number of dimensions.
        :param closed_branches: number of branches where propagation block reached a consumer node on the passage from
        the producer nodes.
        """
        self.size = size
        self.offset = offset
        self.pruning_dimension = pruning_dimension
        self._producer = producer
        self._closed_branches = closed_branches
        self._group = None
        self._is_invalid = False

    def __eq__(self, other) -> bool:
        return self.pruning_dimension == other.pruning_dimension and \
            self.size == other.size and \
            self.offset == other.offset and \
            self._producer.id == other._producer.id

    def __str__(self) -> str:
        return f"S:{self.size}__O:{self.offset}__ID:{self._producer.id}"

    def __repr__(self) -> str:
        return self.__str__()

    def close_branch(self) -> None:
        """
        Increase the count of closed branches. It will be used to filter groups that have dimension blocks not reached
        consumers on all branches.
        """
        self._closed_branches += 1

    def set_group(self, group) -> None:
        self._group = group


class PropagationGroup:
    """
    Defines a group of propagation blocks and links it with the list of children groups.
    The group is initialized on producers of pruning and is propagated until consumers within PropagationMask.
    """

    def __init__(self, blocks: List[PropagationBlock]) -> None:
        self._blocks = blocks
        for block in blocks:
            block.set_group(self)
        self._children: List['PropagationGroup'] = []
        self.is_invalid = False

    def __str__(self) -> str:
        state = self.get_state()
        return json.dumps(state, separators=(',\n', ':'))

    def invalidate(self) -> None:
        """
        Invalidate all blocks in the group and do the same for child groups.
        """
        for block in self._blocks:
            block.is_invalid = True
        for child in self._children:
            child.invalidate()

    def get_blocks_on_leaves(self) -> List[List[PropagationBlock]]:
        """
        Traverses all children groups until the leaves and collects all propagation blocks at them.

        :return: the list of blocks from each leaf.
        """
        if not self._children:
            return [self._blocks]
        retval = []
        for child in self._children:
            groups = child.get_blocks_on_leaves()
            retval.append(groups[0] if len(groups) == 1 else groups)
        return retval

    def close_branch(self) -> None:
        """
        Increase the count of closed branches for the group.
        The counter will be used to filter groups that have propagation blocks not reached consumers on all branches.
        """
        for block in self._blocks:
            block.close_branch()

    @staticmethod
    def join_groups(*args: 'PropagationGroup') -> 'PropagationGroup':
        """
        Join block groups into a new one. The group combines all block and child groups from the given list of groups.

        :return: a new block group.
        """
        for group in args:
            assert isinstance(group, PropagationGroup), \
                f'Couldn\'t join args {args}, all elements should be BlockGroup instances'

        retval = PropagationGroup([])
        blocks = []
        for group in args:
            group.add_child(retval)
            for block in group.get_blocks():
                if block not in blocks:
                    blocks.append(block)
        for block in blocks:
            retval.add_block(block)
        return retval


    def get_state(self) -> str:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return list(map(str, self._blocks))

    def get_blocks(self) -> List[PropagationBlock]:
        return self._blocks.copy()

    def get_children(self) -> List['PropagationGroup']:
        return self._children.copy()

    def has_children(self) -> bool:
        return bool(self._children)

    def add_child(self, child: 'PropagationGroup') -> None:
        self._children.append(child)

    def add_block(self, block: PropagationBlock) -> None:
        self._blocks.append(block)
        block.set_group(self)


class PropagationMask:
    """
    Contains information about pruning in the current node:
    a group of propagation blocks per dimension for which they are applied.

    It's assumed that the propagation mask is initialized on producers and then propagated through the
    execution graph until consumer nodes.

    This helps to find possible ways of pruning nodes with tracking dependency between them.
    For example, to constrain a group of nodes to have the same structure or to find a
    specific pruning structure (PropagationBlock) that can be safely removed in all producers
    with retaining it in the consumers.
    """

    def __init__(self,
                 dim_groups_map: Dict[int, List[PropagationGroup]] = None) -> None:
        self.dim_groups_map = dim_groups_map if dim_groups_map is not None else {}

    def __str__(self) -> str:
        state = {dim: list(map(lambda x: x.get_state(), groups)) for dim, groups in self.dim_groups_map.items()}
        return json.dumps(state)

    def invalidate_groups(self) -> None:
        """
        Invalidate all blocks in the group and do the same for child groups.
        Can happen when propagation mask for some reason can't reach the consumer.
        For instance, when it's traversed until the node in ignored scope and that doesn't
        support pruning.
        """
        for groups in self.dim_groups_map.values():
            for group in groups:
                group.invalidate()


@dataclass
class PruningBlock:
    """
    Final and minimal representation of PropagationBlock after mask propagation.
    By analogy, it defines how much and which particular channels are supposed to be pruned for the node
    when a single element of pruning mask is 0.
    We assume that pruning mask is a vector with 1's and 0's. 1 retains the corresponding channel in weights,
    0 prunes it.
    """
    size: int
    offset: int
    producer_id: int
    pruning_dimension: int

    @classmethod
    def from_propagation_block(cls, block: PropagationBlock) -> 'PruningBlock':
        """
        Creates an object by taking all necessary information from PropagationBlock.
        """
        return cls(block.size, block.offset, block._producer.id, block.pruning_dimension)

    def __str__(self) -> str:
        return f'S{self.size}_O{self.offset}_PID{self.producer_id}'

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: 'PruningBlock') -> bool:
        return str(self) == str(other)


@dataclass
class PruningGroup:
    """
    Group of pruning blocks that is obtained after propagation.
    """
    dim_blocks: Set[PruningBlock]

    def __eq__(self, other: 'PruningGroup'):
        return self.dim_blocks == other.dim_blocks


def add_group_to_graph(graph: nx.DiGraph,
                       parent_group: PropagationGroup,
                       visited_block_ids_map: Dict[int, int]) -> None:
    """
    Recursive helper to traverse children of the given PropagationBlock with adding them to the graph.

    :param graph: hierarhy of the propagation groups/blocks.
    :param root_group: current group for traversing.
    :param visited_block_ids_map: helper mapping of group addresses to the id in the graph.
    """
    global counter
    parent_graph_id = counter
    graph.add_node(parent_graph_id, label=str(parent_group))
    visited_block_ids_map[id(parent_group)] = parent_graph_id
    if parent_group.has_children():
        for child_group in parent_group.get_children():
            child_id = id(child_group)
            if child_id not in visited_block_ids_map:
                counter += 1
                child_graph_id = counter
                visited_block_ids_map[id(child_group)] = child_graph_id
                graph.add_edge(parent_graph_id, child_graph_id)
                add_group_to_graph(graph, child_group, visited_block_ids_map)
            else:
                child_graph_id = visited_block_ids_map[child_id]
                graph.add_edge(parent_graph_id, child_graph_id)


counter = 0


def build_block_hierarchy(root_groups: List[PropagationGroup]) -> nx.DiGraph:
    """Creates hierarchy of propagation blocks/groups by traversing children in the given root groups.

    :param roots: list of the root groups
    :return: networkx graph that represents the hierarchy of propagation blocks/groups.
    """
    graph = nx.DiGraph()
    global counter
    counter = 0
    visited = dict()
    for root_group in root_groups:
        add_group_to_graph(graph, root_group, visited)
        counter += 1
    return graph


def get_pruning_groups(graph: NNCFGraph,
                       pruning_operations_metatypes: PruningOperationsMetatypeRegistry,
                       prune_operations_types: List[str],
                       dump_dir: Optional[Path] = None) -> List[PruningGroup]:
    """
    Determines how nodes of the given types should be pruned: which nodes should be pruned together, along which
    dimension, how many sequent channels with which offset. It's done by initializing PropagationMask's on the
    operations with prunable paramaters (producers of pruning) and propagating them through the execution graph.

    :param graph: nncf graph to initialize and propagate masks.
    :param pruning_operations_metatypes: registry with operation metatypes pruning algorithm is aware of, i.e.
    metatypes describing operations with common pruning mask application and propagation properties, e.g.
    IdentityMaskForwardOps unifies operations that propagate pruning masks as is (relu, swish etc.), whereas
    Convolution unifies different convolution operations (conv1d, conv2d, conv3d) which accepts some input masks and
    provide some output masks.
    :param prune_operations_types: types of operations with prunable parameters.
    :return: list of groups with parameters of pruning.
    """
    # 1. Initialize masks for producing nodes
    all_nodes_to_prune = graph.get_nodes_by_types(prune_operations_types)  # type: List[NNCFNode]
    roots = {}
    for node in all_nodes_to_prune:
        assert isinstance(node.layer_attributes, (LinearLayerAttributes, ConvolutionLayerAttributes))
        pruning_dim = node.layer_attributes.get_target_dim_for_compression()
        output_tensors_shapes = [x.tensor_shape for x in graph.get_output_edges(node)]
        assert len(output_tensors_shapes) == 1 or len( set(output_tensors_shapes)) <= 1, node.node_name
        output_tensors_shape = output_tensors_shapes[0]
        target_output_dim_for_compression = len(output_tensors_shape) - 1
        root_group = PropagationGroup(
            blocks=[
                PropagationBlock(
                    producer=MaskProducer(node.node_id),
                    pruning_dimension=pruning_dim
                )
            ]
        )
        mask = PropagationMask(
            dim_groups_map={
                target_output_dim_for_compression: [root_group]
            }
        )
        roots[node.node_id] = root_group
        node.data['output_mask'] = mask

    def get_attributes_fn(node: NNCFNode) -> Dict[str, Any]:
        result = {'metatype': str(node.metatype.name), 'node_id': str(node.node_id)}
        if node.layer_attributes:
            result.update(map(lambda pair: (pair[0], str(pair[1])), node.layer_attributes.__dict__.items()))
        if 'output_mask' in node.data:
            output_mask = node.data['output_mask']
            if output_mask:
                result['output_mask'] = str(output_mask)
        return result

    try:
        # 2. Propagate masks
        MaskPropagationAlgorithm(graph, pruning_operations_metatypes).mask_propagation()
    finally:
        if dump_dir is not None:
            block_graph = build_block_hierarchy(list(roots.values()))
            write_dot_graph(block_graph, dump_dir / 'latest_block_group_hierarchy.dot')
            save_for_netron(graph, str(dump_dir / 'latest_propagated_graph.xml'), get_attributes_fn=get_attributes_fn)

    # 3. Collect groups from producers
    blocks_map: Dict[int, List[List[PropagationBlock]]] = {}
    for id, group in roots.items():
        blocks_map[id] = group.get_blocks_on_leaves()

    # Filter non closing and duplicated groups
    pruning_groups = []  # type: List[PruningGroup]
    finished_producers = []
    for groups in blocks_map.values():
        for group in groups:
            blocks = []

            def collect_block_fn(x: PropagationBlock) -> PropagationBlock:
                blocks.append(x)
                return x

            objwalk(group, lambda x: isinstance(x, PropagationBlock), collect_block_fn)
            if all(block._closed_branches == 1 for block in blocks):
                for block in group:
                    assert not block._is_invalid, 'invalid groups are not handled'
                min_group = set(map(PruningBlock.from_propagation_block, group))
                all_not_finished = all(g.producer_id not in finished_producers for g in min_group)
                candidate_group = PruningGroup(min_group)
                if candidate_group not in pruning_groups and all_not_finished:
                    pruning_groups.append(candidate_group)
                    finished_producers.extend(g.producer_id for g in min_group)
                break  # iterate and choose first valid and not finished

    return pruning_groups
