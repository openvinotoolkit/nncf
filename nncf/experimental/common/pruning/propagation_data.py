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

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class ProducerInfo:
    """
    Defines a node that produce pruning masks. Operation that corresponds to the node has built-in weights.
    Pruning dimension is required in comparison with consumer info.

    :param node_id: id of the node that produce pruning
    :param pruning_dimension: axis number from 0 to N-1 in weights along which the pruning structure is defined.
        N is total number of dimensions.
    """

    node_id: int
    pruning_dimension: int = 0

    def __hash__(self) -> int:
        return hash((self.node_id, self.pruning_dimension))

    def __str__(self) -> str:
        return str(self.node_id)

    def __lt__(self, other: "ProducerInfo"):
        return self.node_id < other.node_id


@dataclass
class ConsumerInfo:
    """
    Defines the node that absorbs (consume) pruning masks. There are 2 types of consumer. First, have built-in weights,
    therefore can consume and produce pruning along the input channels. Consumers of the second type don't have built-in
    weights, instead weights are coming as a second input. In that case, pruning dimension is None.

    :param node_id: id of the node that consume pruning.
    :param pruning_dimension: axis number from 0 to N-1 in weights along which the pruning structure is defined.
        N is total number of dimensions. Equals None for the second type of consumers.
    """

    node_id: int
    pruning_dimension: Optional[int] = 1

    def __hash__(self) -> int:
        return hash((self.node_id, self.pruning_dimension))

    def __str__(self) -> str:
        return str(self.node_id)

    def __lt__(self, other: "ConsumerInfo"):
        return self.node_id < other.node_id


@dataclass
class PruningBlock:
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

    :param size: number of sequent channels.
    :param offset: when not equal to 0, block is formed by taking `size` number of sequent channels,
        then skipping `offset` number of sequent channels and repeating the procedure for the rest of channels.
    """

    size: int = 1
    offset: int = 0

    def __eq__(self, other) -> bool:
        return self.size == other.size and self.offset == other.offset

    def __str__(self) -> str:
        return f"S:{self.size}__O:{self.offset}"


class PropagationGroup:
    """
    Defines a group of propagation blocks and links it with the list of children groups.
    The group is initialized on producers of pruning and is propagated until consumers within PropagationMask.
    """

    def __init__(
        self,
        block: PruningBlock,
        producers: Optional[Set[ProducerInfo]] = None,
        consumers: Optional[Set[ConsumerInfo]] = None,
    ) -> None:
        self.block = block
        self._children: List["PropagationGroup"] = []
        self._is_invalid = False
        self._producers = set() if producers is None else producers
        self._consumers = set() if consumers is None else consumers

    @property
    def is_invalid(self):
        return self._is_invalid

    def __str__(self) -> str:
        producers = ",".join(map(str, sorted(self._producers)))
        consumers = ",".join(map(str, sorted(self._consumers)))
        return f"Block: {self.block}\nProducers: {producers}\nConsumers: {consumers}"

    def __repr__(self) -> str:
        producers = ",".join(map(str, self.get_producers()))
        consumers = ",".join(map(str, self.get_consumers()))
        return f"{str(self.block)}__P{producers}__C{consumers}"

    def invalidate(self) -> None:
        """
        Invalidate all blocks in the group and do the same for child groups.
        """
        self._is_invalid = True
        for child in self._children:
            child.invalidate()

    @staticmethod
    def join_groups(*args: "PropagationGroup") -> "PropagationGroup":
        """
        Join block groups into a new one. The group combines all block and child groups from the given list of groups.

        :return: a new block group.
        """
        first_block = args[0].block
        assert all(first_block == group.block for group in args), (
            "joining groups with different blocks is not "
            "supported. Need to implement merging of multiple block by choosing smallest common divisor as size.`"
        )

        new_group = PropagationGroup(block=first_block)
        for group in args:
            group.add_child(new_group)
            new_group.add_producers(group.get_producers())
            new_group.add_consumers(group.get_consumers())
        return new_group

    def add_consumers(self, consumers: Set[ConsumerInfo]):
        """
        Adds information about consumer to the group and to all children.
        The idea is to have up-to-date list of consumers in each leaf eventually.

        :param block: propagation block the corresponds to the consumer node.
        """
        self._consumers.update(consumers)
        for child in self._children:
            child.add_consumers(consumers)

    def add_producers(self, producers: Set[ProducerInfo]) -> None:
        self._producers.update(producers)

    def add_child(self, child: "PropagationGroup") -> None:
        self._children.append(child)

    def get_consumers(self) -> Set[ConsumerInfo]:
        return self._consumers.copy()

    def get_producers(self) -> Set[ProducerInfo]:
        return self._producers.copy()

    def get_children(self) -> List["PropagationGroup"]:
        return self._children.copy()

    def has_children(self) -> bool:
        return bool(self._children)


class PropagationMask:
    """
    Contains information about pruning in the current node - pruning block within propagation groups per dimension,
    for which they are applied.

    It's assumed that the propagation mask is initialized on producers and then propagated through the
    execution graph until consumer nodes.

    This helps to find possible ways of pruning nodes with tracking dependency between them.
    For example, to constrain a group of nodes to have the same structure or to find a
    specific pruning structure (PropagationBlock) that can be safely removed in all producers
    with retaining it in the consumers.
    """

    def __init__(self, dim_groups_map: Dict[int, List[PropagationGroup]] = None) -> None:
        self.dim_groups_map = dim_groups_map if dim_groups_map is not None else {}

    def __str__(self) -> str:
        state = {dim: list(map(repr, groups)) for dim, groups in self.dim_groups_map.items()}
        return json.dumps(state)

    def __bool__(self):
        return bool(self.dim_groups_map)

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
