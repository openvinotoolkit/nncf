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

import json
from typing import List
from dataclasses import dataclass
from pathlib import Path

import pytest

from nncf.quantization.algorithms.accuracy_aware.utils import find_fq_nodes_to_cut
from nncf.common.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.experimental.openvino_native.graph.metatypes.common import QUANTIZE_AGNOSTIC_OPERATIONS
from nncf.experimental.openvino_native.graph.metatypes.common import QUANTIZABLE_OPERATIONS
from nncf.experimental.openvino_native.graph.metatypes.common import FAKE_QUANTIZE_OPERATIONS
from nncf.experimental.openvino_native.graph.metatypes.common import CONSTANT_OPERATIONS
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OV_OPERATION_METATYPES
from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT


DATA_DIR = OPENVINO_NATIVE_TEST_ROOT / 'data'


def load_graph(graph_path: Path) -> NNCFGraph:
    with open(graph_path) as f:
        data = json.load(f)

    graph = NNCFGraph()

    correspondences = {}
    for node_desc in data['nodes']:
        node_type = node_desc['type']
        node_id = node_desc['id']
        node_name = f'{node_type}_{node_id}'
        metatype = OV_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_type)
        node = graph.add_nncf_node(node_name, node_type, metatype)
        correspondences[node_id] = node.node_id

    for edge_desc in data['edges']:
        graph.add_edge_between_nncf_nodes(
            correspondences[edge_desc['from_node_id']],
            correspondences[edge_desc['to_node_id']],
            [],
            edge_desc['to_port'],
            edge_desc['from_port'],
            Dtype.FLOAT
        )

    return graph


@dataclass
class TestCase:
    fq_node_name: str  # Test input
    graph: NNCFGraph  # Test input
    fq_nodes: List[str]  # Expected output
    ops: List[str]  # Expected output


def generate_test_cases(graph_path: Path, data_path: Path) -> List[TestCase]:
    graph = load_graph(graph_path)

    with open(data_path) as f:
        data = json.load(f)

    test_cases = []
    for fq_node_name, expected in data.items():
        test_cases.append(
            TestCase(fq_node_name, graph, sorted(expected['fq_nodes']), sorted(expected['ops']))
        )
    return test_cases


TEST_CASES = []
TEST_CASES.extend(
    generate_test_cases(
        DATA_DIR / 'test_fq_remove_resnet18_graph.json',
        DATA_DIR / 'test_fq_remove_gt.json'
    )
)


@pytest.mark.parametrize('test_case', TEST_CASES)
def test_find_fq_nodes_to_cut(test_case: TestCase):
    fq_node = test_case.graph.get_node_by_name(test_case.fq_node_name)
    nodes, ops = find_fq_nodes_to_cut(
        test_case.graph,
        fq_node,
        lambda node: node.metatype in FAKE_QUANTIZE_OPERATIONS,
        lambda node: node.metatype in CONSTANT_OPERATIONS,
        lambda node: node.metatype in QUANTIZABLE_OPERATIONS,
        lambda node: node.metatype in QUANTIZE_AGNOSTIC_OPERATIONS
    )

    actual_fq_nodes = sorted([x.node_name for x in nodes])
    actual_ops = sorted([x.node_name for x in ops])

    assert actual_fq_nodes == test_case.fq_nodes
    assert actual_ops == test_case.ops
