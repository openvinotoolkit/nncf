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

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from typing import Dict, List, NamedTuple, Set, TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.quantization.passes import transform_to_inference_graph

TModel = TypeVar("TModel")


class NodeOutputPort(NamedTuple):
    """
    Represents the output port of a node in the graph.

    :param node_name: The name of the node.
    :param output_port: The output port index.
    """

    node_name: str
    output_port: int


@dataclass
class LayerwiseStep:
    """
    Represents a single step in the layer-wise processing of the graph.

    :param target_node_map: Mapping of target nodes to their inputs/outputs ports
        and corresponding output ports in the subgraph.
    :param subgraph_inputs: List of input ports for the subgraph.
    :param subgraph_outputs: List of output ports for the subgraph.
    """

    target_node_map: Dict[NNCFNode, Dict[int, NodeOutputPort]]
    subgraph_inputs: List[NodeOutputPort]
    subgraph_outputs: List[NodeOutputPort]


@dataclass
class SimplePath:
    """
    Represents a simple path in the graph with input and output nodes.

    :param input_nodes: Set of input nodes.
    :param output_nodes: Set of output nodes.
    :param inputs: Set of input ports.
    """

    input_nodes: Set[NNCFNode]
    output_nodes: Set[NNCFNode]
    inputs: Set[NodeOutputPort] = field(default_factory=set)


class LayerwiseScheduler:
    """
    Scheduler that determines the order of layer-wise steps for processing a graph based on a given strategy.

    The scheduling algorithm works as follows:
    1. Initialize input nodes and create a copy of the graph for inference.
    2. Perform a topological traversal of the graph starting from the input nodes.
    3. For each node, if it is a target node and has not been processed, add the path from inputs to this node
       to the paths list.
    4. If `add_additional_outputs` is True, add all output nodes which was partial visited to remove inputs
       to target nodes.
    5. Merge paths that have overlapping input sets into single paths to minimize model extraction time.
    6. Create layer-wise steps by mapping target nodes to their input/output ports and corresponding output ports
       in the subgraph.
    7. Repeat the process until all target nodes have been processed.
    """

    def __init__(self, add_additional_outputs: bool = False):
        """
        :param strategy: The strategy to use for scheduling.
        """
        self.add_additional_outputs = add_additional_outputs

    def schedule(
        self, graph: NNCFGraph, target_nodes: List[NNCFNode], collect_inputs: bool = True
    ) -> List[LayerwiseStep]:
        """
        Schedules the execution of the graph in a layer-wise manner.

        :param graph: The graph to be scheduled.
        :param target_nodes: The nodes that are targeted for layer-wise processing.
        :param collect_inputs: If True (default) collects inputs for target nodes
            oterwise collects outputs of target nodes.
        :return: The scheduled steps for layer-wise processing.
        """
        # Initialize input nodes and create a copy of the graph for inference
        input_nodes = graph.get_input_nodes()
        inference_graph = transform_to_inference_graph(deepcopy(graph), input_nodes, [], [], [])

        steps = []
        visited_map = {node: False for node in inference_graph.get_all_nodes()}
        while not all([visited_map[node] for node in target_nodes]):
            input_nodes_map = {node: {node} for node in input_nodes}
            zero_indegree = input_nodes
            indegree_map = {}
            innode_map = {}
            paths = []

            # topological traversal of a graph starting from input nodes
            while zero_indegree:
                this_generation = zero_indegree
                zero_indegree = []
                for node in this_generation:
                    for next in inference_graph.get_next_nodes(node):
                        if next in input_nodes:
                            continue

                        if next not in input_nodes_map:
                            input_nodes_map[next] = set()
                        input_nodes_map[next] |= input_nodes_map[node]

                        if next not in indegree_map:
                            indegree_map[next] = len(inference_graph.get_previous_nodes(next))
                        indegree_map[next] -= 1

                        if indegree_map[next] == 0:
                            # if node is a target node and is not processed yet then
                            # the path from inputs to node is added to paths
                            if next not in target_nodes:
                                zero_indegree.append(next)
                            elif not visited_map[next]:
                                paths.append(SimplePath(input_nodes=input_nodes_map[next], output_nodes={next}))
                            visited_map[next] = True
                            del indegree_map[next]
                        else:
                            if next not in innode_map:
                                innode_map[next] = set()
                            innode_map[next].add(node)

            # if add_additional_outputs = True then add additional outputs to minimize computation cost,
            # otherwise reuse existing inputs.
            reuse_input_nodes = set()
            if self.add_additional_outputs:
                for node in indegree_map:
                    if not visited_map[node]:
                        paths.append(SimplePath(input_nodes_map[node], innode_map[node]))
            else:
                for node in indegree_map:
                    if not visited_map[node]:
                        reuse_input_nodes |= input_nodes_map[node]

            # fill input ports
            inputs_map = {}
            for node in input_nodes:
                inputs_map[node] = set()
                if collect_inputs and node in target_nodes:
                    for edge in inference_graph.get_input_edges(node):
                        inputs_map[node].add(NodeOutputPort(edge.from_node.node_name, edge.output_port_id))
                else:
                    for edge in inference_graph.get_output_edges(node):
                        inputs_map[node].add(NodeOutputPort(node.node_name, edge.output_port_id))

            for path in paths:
                for input_node in path.input_nodes:
                    path.inputs |= inputs_map[input_node]

            # merge paths to minimize model extraction time
            paths = self._merge_paths(paths)

            # create layerwise step by paths
            old_input_nodes = set()
            new_input_nodes = set()
            for p in paths:
                target_outputs = []
                additional_output_nodes = set()
                for output_node in p.output_nodes:
                    try:
                        target_node_index = target_nodes.index(output_node)
                        target_outputs.append((target_node_index, output_node))
                    except ValueError:
                        if output_node in p.input_nodes:
                            reuse_input_nodes.add(output_node)
                        else:
                            # filter additional output nodes
                            for prev_node in inference_graph.get_previous_nodes(output_node):
                                if prev_node not in p.output_nodes:
                                    additional_output_nodes.add(output_node)
                                    break
                if not target_outputs:
                    continue

                target_outputs.sort(key=lambda target_output: target_output[0])
                target_output_nodes = [output[1] for output in target_outputs]

                old_input_nodes |= p.input_nodes
                new_input_nodes |= set(target_output_nodes) | additional_output_nodes
                subgraph_inputs = list(p.inputs)
                step_target_nodes = OrderedDict()
                subgraph_outputs = []
                for node in target_output_nodes:
                    target_edge = {}
                    if collect_inputs:
                        for edge in inference_graph.get_input_edges(node):
                            output_id = NodeOutputPort(edge.from_node.node_name, edge.output_port_id)
                            target_edge[edge.input_port_id] = output_id
                            if output_id not in subgraph_outputs:
                                subgraph_outputs.append(output_id)
                    else:
                        for edge in inference_graph.get_output_edges(node):
                            output_id = NodeOutputPort(edge.from_node.node_name, edge.output_port_id)
                            target_edge[edge.output_port_id] = output_id
                            if output_id not in subgraph_outputs:
                                subgraph_outputs.append(output_id)
                    step_target_nodes[graph.get_node_by_name(node.node_name)] = target_edge
                for node in additional_output_nodes:
                    for edge in inference_graph.get_output_edges(node):
                        output_id = NodeOutputPort(edge.from_node.node_name, edge.output_port_id)
                        if output_id not in subgraph_outputs:
                            subgraph_outputs.append(output_id)
                steps.append(LayerwiseStep(step_target_nodes, subgraph_inputs, subgraph_outputs))

            # prepare input nodes for next steps
            if self.add_additional_outputs:
                reuse_input_nodes |= set(input_nodes) - old_input_nodes
            input_nodes = list(new_input_nodes | reuse_input_nodes)

        return steps

    def _merge_paths(self, paths: List[SimplePath]) -> List[SimplePath]:
        """
        Merges paths that have overlapping input sets into single paths.

        :param paths: List of paths to merge.
        :returns: The merged list of paths.
        """
        result = []
        for p in paths:
            for s in result:
                if not s.inputs.isdisjoint(p.inputs):
                    s.inputs |= p.inputs
                    s.input_nodes |= p.input_nodes
                    s.output_nodes |= p.output_nodes
                    break
            else:
                result.append(p)
        if len(paths) == len(result):
            return result
        return self._merge_paths(result)
