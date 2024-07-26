# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


# TODO(dlyakhov): Use torch.fx.graph.find_nodes method instead after
# torch version update (>= 2.4)
def get_graph_node_by_name(graph: torch.fx.Graph, name: str) -> torch.fx.Node:
    """
    Retrieves a node with the specified name from the grpah.
    Raises a runtime error if graph does not contain node with
    the given name.

    :param graph: Given torch fx graph.
    :param name: Target node name.
    :return: A graph node with the given name.
    """
    for node in graph.nodes:
        if node.name == name:
            return node
    raise RuntimeError(f"Node with name {name} is not found")
