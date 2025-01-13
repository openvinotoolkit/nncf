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

import networkx as nx
import numpy as np
import pytest

from nncf.common.factory import NNCFGraphFactory
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.model_utils import remove_friendly_name_duplicates
from nncf.openvino.graph.node_utils import create_bias_tensor
from tests.common.quantization.mock_graphs import NodeWithType
from tests.common.quantization.mock_graphs import create_mock_graph
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph
from tests.openvino.native.models import DuplicatedNamesModel


def get_nncf_graph_for_test(edge_shape, dtype):
    nodes = [
        NodeWithType("Input_1", None),
        NodeWithType("Conv_1", OVConvolutionMetatype),
        NodeWithType("Output_1", None),
    ]
    node_edges = [
        ("Input_1", "Conv_1"),
        ("Conv_1", "Output_1"),
    ]
    original_mock_graph = create_mock_graph(nodes, node_edges)
    nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)
    nncf_graph._nx_graph.out_edges[("1 /Conv_1_0", "2 /Output_1_0")][nncf_graph.ACTIVATION_SHAPE_EDGE_ATTR] = edge_shape
    nncf_graph._nx_graph.out_edges[("1 /Conv_1_0", "2 /Output_1_0")][nncf_graph.DTYPE_EDGE_ATTR] = dtype
    return nncf_graph


@pytest.mark.parametrize(
    "edge_shape,dtype,ref_shape",
    [((2, 3, 4, 5), np.float32, (1, 3, 1, 1)), ((1, 1, 2, 3), np.float64, (1, 1, 1, 1))],
)
def test_create_bias_tensor(edge_shape, dtype, ref_shape):
    graph = get_nncf_graph_for_test(edge_shape, dtype)
    val = create_bias_tensor(graph.get_node_by_name("/Conv_1_0"), graph, 5)
    assert val.shape == ref_shape
    assert np.equal(val, np.full(ref_shape, 5)).all()


@pytest.mark.parametrize(
    "model_instance,remove_duplicates,unique_layer_numbers",
    [
        (DuplicatedNamesModel().ov_model, False, 5),
        (DuplicatedNamesModel().ov_model, True, 6),
    ],
)
def test_remove_friendly_name_duplicates(model_instance, remove_duplicates, unique_layer_numbers):
    if remove_duplicates:
        model_instance = remove_friendly_name_duplicates(model_instance)

    model_uniqie_names = set([op.get_friendly_name() for op in model_instance.get_ops()])
    assert len(model_uniqie_names) == unique_layer_numbers

    nncf_graph = NNCFGraphFactory.create(model_instance)
    assert len(nncf_graph.get_all_nodes()) == unique_layer_numbers

    try:
        _ = InsertionPointGraph(nncf_graph)
        assert remove_duplicates
    except nx.NetworkXUnfeasible:
        assert not remove_duplicates
