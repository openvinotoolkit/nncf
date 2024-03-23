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

from enum import Enum
from pathlib import Path

import pytest

from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.quantization.passes import filter_constant_nodes
from nncf.quantization.passes import remove_nodes_and_reconnect_graph
from tests.post_training.test_templates.models import NNCFGraphDropoutRemovingCase
from tests.post_training.test_templates.models import NNCFGraphToTestConstantFiltering
from tests.shared.nx_graph import compare_nx_graph_with_reference
from tests.shared.paths import TEST_ROOT

DATA_ROOT = TEST_ROOT / "common" / "data" / "reference_graphs"


class TestModes(Enum):
    VALID = "valid"
    WRONG_TENSOR_SHAPE = "wrong_dropout_node"
    WRONG_PARALLEL_EDGES = "wrong_parallel_edges"


def _check_graphs(dot_file_name, nncf_graph) -> None:
    nx_graph = nncf_graph.get_graph_for_structure_analysis()
    path_to_dot = DATA_ROOT / dot_file_name
    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)


@pytest.mark.parametrize("mode", [TestModes.VALID, TestModes.WRONG_TENSOR_SHAPE, TestModes.WRONG_PARALLEL_EDGES])
def test_remove_nodes_and_reconnect_graph(mode: TestModes):
    dot_reference_path_before = Path("passes") / "dropout_synthetic_model_before.dot"
    dot_reference_path_after = Path("passes") / "dropout_synthetic_model_after.dot"
    dropout_metatype = "DROPOUT_METATYPE"
    kwargs = {}
    if mode != TestModes.VALID:
        kwargs.update({mode.value: True})

    nncf_graph = NNCFGraphDropoutRemovingCase(dropout_metatype, **kwargs).nncf_graph

    if mode != TestModes.VALID:
        with pytest.raises(AssertionError):
            remove_nodes_and_reconnect_graph(nncf_graph, [dropout_metatype])
        return

    _check_graphs(dot_reference_path_before, nncf_graph)
    remove_nodes_and_reconnect_graph(nncf_graph, [dropout_metatype])
    _check_graphs(dot_reference_path_after, nncf_graph)


@pytest.mark.parametrize("node_between_const_and_op", [False, True])
def test_filter_constant_nodes(node_between_const_and_op):
    dot_reference_path_before = (
        Path("passes") / f"test_constant_filtering_model_before{int(node_between_const_and_op)}.dot"
    )
    dot_reference_path_after = (
        Path("passes") / f"test_constant_filtering_model_after{int(node_between_const_and_op)}.dot"
    )

    class ConstantMetatype(OperatorMetatype):
        num_expected_input_edges = 0
        pass

    class NodeWithWeightMetatype(OperatorMetatype):
        num_expected_input_edges = 2

    nncf_graph = NNCFGraphToTestConstantFiltering(
        ConstantMetatype,
        NodeWithWeightMetatype,
        MultipleInputLayerAttributes(1, 3),
        node_between_const_and_op,
    ).nncf_graph

    additional_input_names = ["/Conv2_0", "/Concat_with_missed_input_0"]
    input_nodes = nncf_graph.get_input_nodes() + [nncf_graph.get_node_by_name(name) for name in additional_input_names]
    _check_graphs(dot_reference_path_before, nncf_graph)
    filter_constant_nodes(nncf_graph, input_nodes)
    _check_graphs(dot_reference_path_after, nncf_graph)
