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

from typing import Tuple, List, Optional

from functools import partial
from collections import Counter
import pytest

from nncf.common.graph.traversal import traverse_graph
from nncf.common.graph.traversal import Node
from nncf.common.graph.graph import NNCFGraph
from tests.common.quantization.test_filter_constant_nodes import ModelToTest1


def traverse_function(node: Node, output: List[Node], visited) -> Tuple[bool, List[Node]]:
    if visited[node.node_id]:
        return True, output
    visited[node.node_id] = True

    output.append(node)
    return False, output


class CaseForTest:
    def __init__(self, model, ref_traversing_output: List[str], start_nodes: Optional[List[str]] = None,
                 traverse_forward: Optional[bool] = True):
        self.model = model
        self.start_node_keys = start_nodes
        self.traverse_forward = traverse_forward
        self.ref_traversing_output = ref_traversing_output


@pytest.mark.parametrize('test_case', (CaseForTest(ModelToTest1(),
                                                   ['1 /Input_1_0', '2 /Conv_1_0', '5 /FC_1_0',
                                                    '6 /Identity_2_0', '7 /FC_2_0',
                                                    '8 /Output_1_0'], None, True),
                                       CaseForTest(ModelToTest1(),
                                                   ['5 /FC_1_0', '6 /Identity_2_0', '7 /FC_2_0',
                                                    '8 /Output_1_0'], ['5 /FC_1_0'], True),
                                       CaseForTest(ModelToTest1(),
                                                   ['1 /Input_1_0', '2 /Conv_1_0', '5 /FC_1_0',
                                                    '6 /Identity_2_0', '4 /Identity_1_0', '3 /Reshape_1_0'],
                                                   ['6 /Identity_2_0'],
                                                   False)
                                       ))
def test_graph_traversal(test_case: CaseForTest):
    nncf_graph = test_case.model.nncf_graph
    visited = {node_id: False for node_id in nncf_graph.get_all_node_ids()}
    partial_f = partial(traverse_function, visited=visited)
    start_nodes = None
    if test_case.start_node_keys is not None:
        start_nodes = list(map(nncf_graph.get_node_by_key, test_case.start_node_keys))
    output = traverse_graph(nncf_graph, partial_f, start_nodes=start_nodes,
                            traverse_forward=test_case.traverse_forward)
    output_node_keys = [node.data[NNCFGraph.KEY_NODE_ATTR] for node in output]
    assert Counter(output_node_keys) == Counter(test_case.ref_traversing_output)
