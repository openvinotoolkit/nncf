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
from typing import Dict, Optional, Set

from nncf.common.quantization.quantizer_propagation.structs import PropagatingQuantizer


class UnifiedScalePropagatingQuantizerGroupManager:
    """
    Keeps track of the groups of quantizers that have to have their scales unified in the final
    quantized model.
    """

    def __init__(self):
        self._next_gid = 0
        self._group_vs_prop_quants_dict: Dict[int, Set[PropagatingQuantizer]] = {}

    def _get_next_gid(self) -> int:
        retval = self._next_gid
        self._next_gid += 1
        return retval

    def register_group(self, prop_quants: Set[PropagatingQuantizer]) -> int:
        """
        Registers a set of propagating quantizers as a new group.

        :param prop_quants: A set of propagating quantizers to be registered.
        :return: The ID of the newly created group.
        """
        for pq in prop_quants:
            for gid, group in self._group_vs_prop_quants_dict.items():
                assert pq not in group, "Propagating quantizer #{} is already registered in a group {}!".format(
                    pq.id, gid
                )
        gid = self._get_next_gid()
        self._group_vs_prop_quants_dict[gid] = prop_quants
        return gid

    def add_to_group(self, target_gid: int, prop_quant: PropagatingQuantizer):
        """
        Adds a propagating quantizer to an already existing group.

        :param target_gid: The ID of the group to be extended.
        :param prop_quant: The propagating quantizer to be registered in the group. The quantizer
          must not be already registered in any group.
        """
        for gid, group in self._group_vs_prop_quants_dict.items():
            if target_gid != gid:
                assert prop_quant not in group, (
                    "Tried to add propagating quantizer #{} to group #{}, "
                    "but it is already registered in a group {}!".format(prop_quant.id, target_gid, gid)
                )
        self._group_vs_prop_quants_dict[target_gid].add(prop_quant)

    def remove_from_group(self, group: int, prop_quant: PropagatingQuantizer):
        """
        Removes a propagating quantizer from a group.

        :param group: The ID of the group from where a quantizer should be removed.
        :param prop_quant: The propagating quantizer to be removed from the group.
        """
        self._group_vs_prop_quants_dict[group].remove(prop_quant)

    def get_group_vs_prop_quants_dict(self) -> Dict[int, Set[PropagatingQuantizer]]:
        """
        :return: A dictionary of groups vs propagating quantizers currently associated with the corresponding group.
        """
        return copy(self._group_vs_prop_quants_dict)

    def get_group_id_by_propagating_quantizer_id(self, requested_pqid: int) -> Optional[int]:
        """
        If a propagating quantizer with a given ID is registered within a group,
        then this function will return the corresponding group ID; otherwise, None is returned.

        :param requested_pqid: The ID of the propagating quantizer to search for among groups.
        :return: The group ID of the quantizer, if found, otherwise None.
        """
        for gid, group in self._group_vs_prop_quants_dict.items():
            for pq in group:
                if pq.id == requested_pqid:
                    return gid
        return None

    def merge_groups(self, merge_to_gid: int, merge_from_gid: int):
        """
        Merges two groups into a single one. The `merge_to_gid` group retains its group ID.

        :param merge_to_gid: The ID of the group to merge into.
        :param merge_from_gid: The ID of the group to merge into the group defined by `merge_to_gid`
        """
        if merge_to_gid == merge_from_gid:
            return
        self._group_vs_prop_quants_dict[merge_to_gid].update(self._group_vs_prop_quants_dict[merge_from_gid])
        self._group_vs_prop_quants_dict.pop(merge_from_gid)


class QuantizersWaitingForMergeManager:
    """
    Tracks the quantizers that await a merge while trying to transition through a downward-branching node
    and corresponding node keys.
    """

    def __init__(self):
        self._branching_node_keys_vs_quantizers_waiting_for_merge: Dict[str, Set[PropagatingQuantizer]] = {}
        self._quantizers_vs_branching_node_keys: Dict[PropagatingQuantizer, str] = {}

    def add_propagating_quantizer_to_wait_on_node_key(self, pq: PropagatingQuantizer, branching_node_key: str):
        """
        Registers a propagating quantizer as "waiting" on a node in QuantizerPropagationStateGraph.

        :param pq: The propagating quantizer to be registered as "waiting".
        :param branching_node_key: The node key in QuantizerPropagationStateGraph to be waited upon, most likely
          the downward-branching node.
        """
        if branching_node_key not in self._branching_node_keys_vs_quantizers_waiting_for_merge:
            self._branching_node_keys_vs_quantizers_waiting_for_merge[branching_node_key] = set()
        self._branching_node_keys_vs_quantizers_waiting_for_merge[branching_node_key].add(pq)
        self._quantizers_vs_branching_node_keys[pq] = branching_node_key

    def get_blocking_node(self, pq: PropagatingQuantizer) -> str:
        """
        Returns the node key upon which the propagating quantizer is registered to be "waiting".

        :param pq: The propagating quantizer that has already been registered to be "waiting" on a node.
        :return: The node key in QuantizerPropagationStateGraph that the `pq` is registered to be waiting upon.
        """
        return self._quantizers_vs_branching_node_keys[pq]

    def get_waiting_quantizers_for_branching_node_key(self, node_key: str) -> Set[PropagatingQuantizer]:
        """
        Returns the set of all quantizers registered to be "waiting" on a given node key in
        QuantizerPropagationStateGraph.

        :param node_key: The node key in QuantizerPropagationStateGraph
        :return: The set of propagating quantizers registered as "waiting" on `node_key`.
        """
        return self._branching_node_keys_vs_quantizers_waiting_for_merge[node_key]

    def __contains__(self, item: PropagatingQuantizer):
        return item in self._quantizers_vs_branching_node_keys

    def resolve_merged_node(self, branching_node_key: str):
        """
        De-registers any quantizers that were previously registered to be "waiting" on a given node key.
        :param branching_node_key: The node key in QuantizerPropagationStateGraph that some propagating
          quantizers have previously been registered upon.
        """
        for pq in self._branching_node_keys_vs_quantizers_waiting_for_merge[branching_node_key]:
            self._quantizers_vs_branching_node_keys.pop(pq)
        self._branching_node_keys_vs_quantizers_waiting_for_merge.pop(branching_node_key)
