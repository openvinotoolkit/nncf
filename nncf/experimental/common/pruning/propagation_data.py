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

from dataclasses import dataclass
import json
from typing import Dict, Optional, Set
from typing import List


# probably not needed
@dataclass
class MaskProducer:
    """
    Defines producer of the pruning.
    """
    # TODO: rename to node_id
    def __init__(self, id_) -> None:
        """
        :param id_: identification of the producer node in the NNCFGraph.
        """
        self.node_id = id_

@dataclass
class ConsumerInfo:
    """
    Defines the node that absorbs (consume) pruning
    """
    node_id: int
    pruning_dimension: int

    def __hash__(self) -> int:
        return hash((self.node_id, self.pruning_dimension))

    def __str__(self) -> str:
        return f'ID: {self.node_id}'

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
                 pruning_dimension: int = 0) -> None:
        """

        :param producer: descriptor of the producer
        :param size: number of sequent channels.
        :param offset: when not equal to 0, block is formed by taking `size` number of sequent channels,
        then skipping `offset` number of sequent channels and repeating the procedure for the rest of channels.
        :param pruning_dimension: axis number from 0 to N-1 in weights along which the dimension block defines pruning
        structure. N is total number of dimensions.
        """
        self.size = size
        self.offset = offset
        self.pruning_dimension = pruning_dimension
        self._producer = producer

    def __eq__(self, other) -> bool:
        return self.pruning_dimension == other.pruning_dimension and \
            self.size == other.size and \
            self.offset == other.offset and \
            self._producer.node_id == other._producer.id

    def __str__(self) -> str:
        return f"S:{self.size}__O:{self.offset}__ID:{self._producer.node_id}"

    def __repr__(self) -> str:
        return self.__str__()


class PropagationGroup:
    """
    Defines a group of propagation blocks and links it with the list of children groups.
    The group is initialized on producers of pruning and is propagated until consumers within PropagationMask.
    """
    # TODO: rename blocks to producers everywhere
    # TODO: make a set of blocks, since check for repeating and order is not important???
    # TODO: separate block and producers...
    def __init__(self, blocks: List[PropagationBlock], consumers: Optional[Set[ConsumerInfo]] = None) -> None:
        # TODO: all blocks inside the group should have the same, size and offset
        self._blocks = blocks
        if consumers is None:
            consumers = set()
        self._children: List['PropagationGroup'] = []
        self.is_invalid = False
        self._consumers = consumers

    def __str__(self) -> str:
        producers = '\n'.join(map(str, self._blocks))
        consumers = '\n'.join(map(str, self._consumers))
        return f'Producers\n{producers}\nConsumers\n{consumers}'

    def invalidate(self) -> None:
        """
        Invalidate all blocks in the group and do the same for child groups.
        """
        self.is_invalid = True
        for child in self._children:
            child.invalidate()

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
        consumers = []
        for group in args:
            group.add_child(retval)
            for block in group.get_blocks():
                if block not in blocks:
                    retval.add_block(block)
            for consumer in group.get_consumers():
                if consumer not in consumers:
                    retval.add_consumer(consumer)
        return retval

    def get_state(self) -> str:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return list(map(str, self._blocks))

    def add_consumer(self, consumer: ConsumerInfo):
        """
        Adds information about consumer to the group and to all children.
        The idea is to have up-to-date list of consumers in each leaf eventually.

        :param block: propagation block the corresponds to the consumer node.
        """
        self._consumers.add(consumer)
        for child in self._children:
            child.add_consumer(consumer)

    def get_blocks(self) -> List[PropagationBlock]:
        return self._blocks.copy()

    def get_children(self) -> List['PropagationGroup']:
        return self._children.copy()

    def get_consumers(self) -> Set[ConsumerInfo]:
        return self._consumers.copy()

    def has_children(self) -> bool:
        return bool(self._children)

    def add_child(self, child: 'PropagationGroup') -> None:
        self._children.append(child)

    def add_block(self, block: PropagationBlock) -> None:
        self._blocks.append(block)


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
