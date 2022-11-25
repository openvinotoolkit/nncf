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

from typing import TypeVar, Callable, List, Union, Tuple, Optional, Any

import networkx as nx

Node = TypeVar('Node')


def traverse_graph(graph: nx.DiGraph,
                   traverse_function: Callable[[Node, List[Any]], Tuple[bool, List[Any]]],
                   start_nodes: Optional[List[Node]] = None,
                   traverse_forward: bool = True):
    """
    Traverses graph up or down starting form `curr_node` node.

    :param curr_node: Node from which traversal is started.
    :param traverse_function: Function describing condition of traversal continuation/termination.
    :param traverse_forward: Flag specifying direction of traversal.
    :return:
    """

    def _traverse_graph_recursive_helper(curr_node: Node,
                                         traverse_function: Callable[[Node, List[Any]], Tuple[bool, List[Any]]],
                                         output: List[Any], traverse_forward: bool):
        is_finished, output = traverse_function(curr_node, output)
        get_nodes_fn = graph.get_next_nodes if traverse_forward else graph.get_previous_nodes
        if not is_finished:
            for node in get_nodes_fn(curr_node):
                _traverse_graph_recursive_helper(node, traverse_function, output, traverse_forward)
        return output

    output = []
    if not start_nodes:
        start_nodes = graph.get_input_nodes()
    for node in start_nodes:
        _traverse_graph_recursive_helper(node, traverse_function, output, traverse_forward)
    return output
