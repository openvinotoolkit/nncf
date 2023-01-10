"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Callable
from typing import Tuple
from typing import List

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode


def find_fq_nodes_to_cut(graph: NNCFGraph,
                         fq_node: NNCFNode,
                         is_fake_quantize: Callable[[NNCFNode], bool],
                         is_const_node: Callable[[NNCFNode], bool],
                         is_quantizable: Callable[[NNCFNode], bool],
                         is_quantize_agnostic: Callable[[NNCFNode], bool]) -> Tuple[List[NNCFNode], List[NNCFNode]]:
    """
    Finds fake quantize nodes that addition to `fq_node` should be removed to get
    the correct model for inference. Returns the list of fake quantize nodes (`fq_node` + nodes
    which were founded) and the list of nodes that will be reverted to original precision when
    fake quantize nodes will be removed.

    :param graph: The NNCFGraph.
    :param fq_node: The fake quantize node which we want to remove.
    :param is_fq_node: A callable that returns `True` if a node is a fake quantize and `False` otherwise.
    :param is_const_node: A callable that returns `True` if a node is a constant and `False` otherwise.
    :param is_quantizable_node: A callable that returns `True` if a node is quantizable and `False` otherwise.
    :param is_quantize_agnostic_node: A callable that returns `True` if node is agnostic and `False` otherwise.
    :return: A tuple (fq_nodes, ops) where `fq_nodes` is the list of fake quantize nodes, ops are the list of
        nodes that will be reverted to original precision when `fq_nodes` will be removed.
    """
    def _parse_node_relatives(node: NNCFNode, is_parents: bool):
        if is_quantizable(node):
            ops_to_return_in_orig_prec.add(node)

        relatives = graph.get_previous_nodes(node) if is_parents else graph.get_next_nodes(node)
        for relative in relatives:
            if is_fake_quantize(relative):
                if is_parents:
                    if relative in seen_children:
                        continue
                    if relative not in to_cut:
                        to_cut.append(relative)
                    to_see_children.append(relative)
                else:
                    seen_children.append(relative)
            elif not is_const_node(relative):
                if relative not in seen_parents:
                    to_see_parents.append(relative)
                if relative not in seen_children and is_quantize_agnostic(relative):
                    to_see_children.append(relative)

        seen_list = seen_parents if is_parents else seen_children
        seen_list.append(node)

    seen_children, seen_parents = [], []
    to_see_children, to_see_parents = [fq_node], []
    to_cut = [fq_node]
    ops_to_return_in_orig_prec = set()

    while to_see_parents or to_see_children:
        if to_see_children:
            _parse_node_relatives(to_see_children.pop(), is_parents=False)
        if to_see_parents:
            _parse_node_relatives(to_see_parents.pop(), is_parents=True)

    return to_cut, ops_to_return_in_orig_prec
