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

from typing import TypeVar, Callable, List, Tuple, Optional, Any
from abc import ABC
from abc import abstractmethod

Node = TypeVar('Node')


class TraversableGraph(ABC):
    @abstractmethod
    def get_next_nodes(self, node: Node) -> List[Node]:
        """
        Returns the consumer nodes of a provided node.

        :param node: Producer node.
        :return: List of consumer nodes of provided node.
        """

    @abstractmethod
    def get_previous_nodes(self, node: Node) -> List[Node]:
        """
        Returns producer nodes of provided node.

        :param node: Consumer node.
        :return: List of producers nodes of provided node.
        """

    @abstractmethod
    def get_input_nodes(self) -> List[Node]:
        """
        Returns all input nodes of the graph.

        :return: A list of input nodes.
        """


def traverse_graph(graph: TraversableGraph,
                   traverse_function: Callable[[Node, List[Any]], Tuple[bool, List[Any]]],
                   start_nodes: Optional[List[Node]] = None,
                   traverse_forward: bool = True) -> List[Node]:
    """
    Traverses graph forward or backward starting from 'start_nodes'.
    The traversing starts iteratively from the 'start_nodes' list.
    If 'start_nodes' is None, then traversing starts from the input nodes of the graph.
    The order of input nodes is based on the returned list of get_input_nodes function of TraversableGraph.
    The traverse logic should be implemented through 'traverse_function'.

    :param graph: Any graph that implements get_next_nodes, get_previous_nodes and get_input_nodes functions.
    :param traverse_function: Function describing condition of traversal continuation/termination.
    :param start_nodes: Nodes from which traversal is started.
    :param traverse_forward: Flag specifying direction of traversal.
    If it is set to True the traversing will be done in forward direction, if - False in the backward direction.
    :return: The traversed path.
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
    if start_nodes is None:
        start_nodes = graph.get_input_nodes()
    for node in start_nodes:
        _traverse_graph_recursive_helper(node, traverse_function, output, traverse_forward)
    return output
