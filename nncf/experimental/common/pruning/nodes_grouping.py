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
from typing import Dict
from typing import List
from typing import Set

import networkx as nx
from nncf.torch.nested_objects_traversal import objwalk
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.experimental.common.graph.netron import save_for_netron



class MaskProducer:
    def __init__(self, id_) -> None:
        self.id = id_


class DimensionBlock:
    def __init__(self,
                 producer: MaskProducer,
                 size: int = 1, offset: int = 0,
                 pruning_dimension: int = 0,
                 opened_branches: int = 1, closed_branches: int = 0) -> None:
        self.size = size
        self.offset = offset
        self.pruning_dimension = pruning_dimension
        self._producer = producer
        self._opened_branches = opened_branches
        self._closed_branches = closed_branches
        self._group = None
        self._is_invalid = False

    def __eq__(self, other):
        return self.pruning_dimension == other.pruning_dimension and \
            self.size == other.size and \
            self.offset == other.offset and \
            self._producer.id == other._producer.id

    def get_state(self):
        return f"S:{self.size}__O:{self.offset}__ID:{self._producer.id}"

    def split_by_reshape(self, shape_map: Dict[int, List[int]]) -> List['DimensionBlock']:
        """
        Reshape constraints creation:
            O -> [A, B, C, D] =>
            constraints:
            (size, offset):
            (1,D %E)
            (D, C*D % E),
            (C*D, B*C*D % E)
            (B*C*D, E % E = 0)
            E=A*B*C*D
        """
        if len(shape_map[1]) == 1:
            raise RuntimeError

        dot_product = reduce((lambda x, y: x * y), shape_map[1])
        assert dot_product == shape_map[0]

        size = dot_product
        blocks = []
        divided_shapes = filter(lambda x: x != 1, shape_map[1])
        for divided_shape in divided_shapes:
            offset = int(size % dot_product)
            size /= divided_shape
            block = DimensionBlock(
                size=int(size), offset=offset,
                pruning_dimension=self.pruning_dimension,
                producer=self._producer,
                opened_branches=self._opened_branches,
                closed_branches=self._closed_branches
            )
            blocks.append(block)
        return blocks

    def add_open_branch(self, num_open_branches=1):
        self._opened_branches += num_open_branches

    def close_branch(self):
        self._closed_branches += 1

    def set_group(self, group):
        self._group = group

    def __repr__(self):
        return self.get_state()


class BlockGroup:
    def __init__(self, blocks: List[DimensionBlock]) -> None:
        self._blocks = blocks
        for block in blocks:
            block.set_group(self)
        self._childs: List['BlockGroup'] = []
        self.is_invalid = False

    def get_state(self):
        return list(map(lambda x: x.get_state(), self._blocks))

    def invalidate(self):
        for block in self._blocks:
            block.is_invalid = True
        for child in self._childs:
            child.invalidate()

    def get_actual_groups(self) -> List[List[DimensionBlock]]:
        if not self._childs:
            return [self._blocks]
        retval = []
        for child in self._childs:
            groups = child.get_actual_groups()
            retval.append(groups[0] if len(groups) == 1 else groups)
        return retval

    def has_childs(self):
        return bool(self._childs)

    def add_childs(self, childs: List['BlockGroup']):
        self._childs.extend(childs)

    def add_block(self, block: DimensionBlock):
        self._blocks.append(block)
        block.set_group(self)

    def split_blocks_by_reshape(self, shape_map):
        if self._childs:
            raise NotImplementedError('Splitting BlockGroup with childs isn\'t implemented yet')

        new_blocks: List[List[DimensionBlock]] = []
        for block in self._blocks:
            new_blocks.append(block.split_by_reshape(shape_map))
        self._childs = []
        for group in zip(*new_blocks):
            self._childs.append(BlockGroup(blocks=list(group)))
        return self._childs.copy()

    def close_branch(self):
        for block in self._blocks:
            block.close_branch()

    def get_blocks(self) -> List[DimensionBlock]:
        return self._blocks.copy()

    @staticmethod
    def join_groups(*args: 'BlockGroup') -> 'BlockGroup':
        for group in args:
            assert isinstance(group, BlockGroup), \
                f'Couldn\'t join args {args}, all elements should be BlockGroup instances'

        retval = BlockGroup([])
        for group in args:
            group.add_childs([retval])
            for block in group.get_blocks():
                if block not in retval._blocks:
                    retval._blocks.append(block)
        return retval


class PropagationMask:
    def __init__(self,
                 dim_groups_map: Dict[int, List[BlockGroup]] = None):
        self.dim_groups_map = dim_groups_map if dim_groups_map is not None else {}

    def invalidate_groups(self):
        for groups in self.dim_groups_map.values():
            for group in groups:
                group.invalidate()

    def get_state(self):
        result = {}
        for dim, groups in self.dim_groups_map.items():
            groups_state = [group.get_state() for group in groups]
            result[dim] = groups_state
        return result


@dataclass
class MinimalDimensionBlock:
    size: int
    offset: int
    producer_id: int
    pruning_dimension: int

    @classmethod
    def from_dimension_block(cls, dim_block: DimensionBlock):
        return cls(dim_block.size, dim_block.offset, dim_block._producer.id, dim_block.pruning_dimension)

    def __str__(self):
        return f'S{self.size}_O{self.offset}_PID{self.producer_id}'

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other: 'MinimalDimensionBlock'):
        return str(self) == str(other)


@dataclass
class PruningNodeGroup:
    dim_blocks: Set[MinimalDimensionBlock]

    def __eq__(self, other: 'PruningNodeGroup'):
        return self.dim_blocks == other.dim_blocks


def block_group_to_graph(graph, root_group: 'BlockGroup', visited_block_ids_map):
    global counter
    parent_graph_id = counter
    graph.add_node(parent_graph_id, label=json.dumps(root_group.get_state(), separators=(',\n', ':')))
    visited_block_ids_map[id(root_group)] = parent_graph_id
    if root_group.has_childs():
        for child_group in root_group._childs:
            child_id = id(child_group)
            if child_id not in visited_block_ids_map:
                counter += 1
                child_graph_id = counter
                visited_block_ids_map[id(child_group)] = child_graph_id
                graph.add_edge(parent_graph_id, child_graph_id)
                block_group_to_graph(graph, child_group, visited_block_ids_map)
            else:
                child_graph_id = visited_block_ids_map[child_id]
                graph.add_edge(parent_graph_id, child_graph_id)


counter = 0


def build_nx_graph_from_roots(roots: Dict[int, 'BlockGroup']):
    graph = nx.DiGraph()
    global counter
    counter = 0
    visited = dict()
    for block_group in roots.values():
        block_group_to_graph(graph, block_group, visited)
        counter += 1
    return graph


def get_pruning_groups(graph: NNCFGraph,
                       pruning_operations_metatypes,
                       prune_operations_types) -> List[PruningNodeGroup]:
    # 1. Initialize masks for producing nodes
    all_nodes_to_prune = graph.get_nodes_by_types(prune_operations_types)  # type: List[NNCFNode]
    roots = {}
    for node in all_nodes_to_prune:
        assert isinstance(node.layer_attributes, (LinearLayerAttributes, ConvolutionLayerAttributes))
        pruning_dim = node.layer_attributes.get_target_dim_for_compression()
        root_group = BlockGroup([DimensionBlock(MaskProducer(node.node_id), pruning_dimension=pruning_dim)])
        roots[node.node_id] = root_group

        output_tensors_shapes = [x.tensor_shape for x in graph.get_output_edges(node)]
        assert len(output_tensors_shapes) == 1 or len(set(output_tensors_shapes)) <= 1, node.node_name
        output_tensors_shape = output_tensors_shapes[0]
        target_output_dim_for_compression = len(output_tensors_shape) - 1
        mask = PropagationMask({target_output_dim_for_compression: [root_group]})
        node.data['output_mask'] = mask

    def get_attributes_fn(node: NNCFNode) -> Dict[str, Any]:
        from nncf.experimental.common.pruning.nodes_grouping import PropagationMask
        result = {'metatype': str(node.metatype.name),
                  'node_id': str(node.node_id)}
        if node.layer_attributes:
            result.update(map(lambda pair: (pair[0], str(
                pair[1])), node.layer_attributes.__dict__.items()))
        if 'output_mask' in node.data:
            output_mask: PropagationMask = node.data['output_mask']
            if output_mask:
                result['output_mask'] = json.dumps(
                    output_mask.get_state(), indent=4)
        return result

    save_for_netron(graph, f'original.xml', get_attributes_fn=get_attributes_fn)

    try:
        # 2. Propagate masks
        MaskPropagationAlgorithm(graph, pruning_operations_metatypes).mask_propagation()
    finally:
        block_graph = build_nx_graph_from_roots(roots)
        write_dot_graph(block_graph, Path(f'latest_block_group_hierarchy.dot'))
        save_for_netron(graph, f'latest_propagated.xml', get_attributes_fn=get_attributes_fn)

    # 3. Collect groups from producers
    blocks_map: Dict[int, List[List[DimensionBlock]]] = {}
    for id, group in roots.items():
        blocks_map[id] = group.get_actual_groups()

    # Filter non closing and duplicated groups
    pruning_groups = []  # type: List[PruningNodeGroup]
    finished_producers = []
    for groups in blocks_map.values():
        for group in groups:
            blocks = []

            def collect_block_fn(x: DimensionBlock) -> DimensionBlock:
                blocks.append(x)
                return x

            objwalk(group, lambda x: isinstance(x, DimensionBlock), collect_block_fn)
            if all(block._closed_branches == 1 for block in blocks):
                for block in group:
                    assert not block._is_invalid, 'invalid groups are not handled'
                min_group = set(map(MinimalDimensionBlock.from_dimension_block, group))
                all_not_finished = all(g.producer_id not in finished_producers for g in min_group)
                candidate_group = PruningNodeGroup(min_group)
                if candidate_group not in pruning_groups and all_not_finished:
                    pruning_groups.append(candidate_group)
                    finished_producers.extend(g.producer_id for g in min_group)
                break  # iterate and choose first valid and not finished

    return pruning_groups
