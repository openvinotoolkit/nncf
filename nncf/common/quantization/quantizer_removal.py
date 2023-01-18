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

from typing import Tuple
from typing import List

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype


def find_quantizer_nodes_to_cut(
        graph: NNCFGraph,
        quantizer_node: NNCFNode,
        quantizer_metatypes: List[OperatorMetatype],
        const_metatypes: List[OperatorMetatype],
        quantizable_metatypes: List[OperatorMetatype],
        quantize_agnostic_metatypes: List[OperatorMetatype]) -> Tuple[List[NNCFNode], List[NNCFNode]]:
    """
    Finds quantizer nodes that should be removed in addition to `quantizer_node` to get
    the correct model for inference. Returns the list of quantizer nodes (`quantizer_node` + nodes
    which were found) and the list of nodes that will be reverted to original precision if
    quantizer nodes are removed.

    :param graph: The NNCFGraph.
    :param quantizer_node: The quantizer node that we want to remove.
    :param quantizer_metatypes: List of quantizer metatypes.
    :param const_metatypes: List of constant metatypes.
    :param quantizable_metatypes: List of quantizable metatypes.
    :param quantize_agnostic_metatypes: List of quantize agnostic metatypes.
    :return: A tuple (quantizer_nodes, ops) where
        - `quantizer_nodes` is the list of quantizer nodes
        - `ops` is the list of nodes that will be reverted to original precision
        if `quantizer_nodes` are removed.
    """
    def _parse_node_relatives(node: NNCFNode, is_parents: bool):
        if node.metatype in quantizable_metatypes:
            ops_to_return_in_orig_prec.add(node)

        relatives = graph.get_previous_nodes(node) if is_parents else graph.get_next_nodes(node)
        for relative in relatives:
            if relative.metatype in quantizer_metatypes:
                if is_parents:
                    if relative in seen_children:
                        continue
                    if relative not in to_cut:
                        to_cut.append(relative)
                    to_see_children.append(relative)
                else:
                    seen_children.append(relative)
            elif relative.metatype not in const_metatypes:
                if relative not in seen_parents:
                    to_see_parents.append(relative)
                if relative not in seen_children and relative.metatype in quantize_agnostic_metatypes:
                    to_see_children.append(relative)

        seen_list = seen_parents if is_parents else seen_children
        seen_list.append(node)

    seen_children, seen_parents = [], []
    to_see_children, to_see_parents = [quantizer_node], []
    to_cut = [quantizer_node]
    ops_to_return_in_orig_prec = set()

    while to_see_parents or to_see_children:
        if to_see_children:
            _parse_node_relatives(to_see_children.pop(), is_parents=False)
        if to_see_parents:
            _parse_node_relatives(to_see_parents.pop(), is_parents=True)

    return to_cut, list(ops_to_return_in_orig_prec)
