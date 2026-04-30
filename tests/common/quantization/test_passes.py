# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.quantization.passes import bypass_noop_operation_nodes
from nncf.quantization.passes import find_constant_subgraphs
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.cross_fw.test_templates.models import NNCFGraphToTestConstantFiltering

DATA_ROOT = TEST_ROOT / "common" / "data" / "reference_graphs"


def _check_graphs(dot_file_name, nncf_graph) -> None:
    nx_graph = nncf_graph.get_graph_for_structure_analysis()
    path_to_dot = DATA_ROOT / dot_file_name
    compare_nx_graph_with_reference(nx_graph, path_to_dot, check_edge_attrs=True)


@pytest.mark.parametrize("node_between_const_and_op", [False, True])
def test_find_constant_subgraphs(node_between_const_and_op):
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
    constant_subgraphs = find_constant_subgraphs(nncf_graph, input_nodes)
    nncf_graph.remove_nodes_from(constant_subgraphs)
    _check_graphs(dot_reference_path_after, nncf_graph)


@dataclass
class ParamBypass:
    name: str
    graph_builder: Callable[[], NNCFGraph]

    def __str__(self):
        return self.name

    @property
    def ref_file(self) -> Path:
        return DATA_ROOT / "passes" / "bypass_noop" / f"{self.name}.dot"


class AnyTestMetaType(OperatorMetatype):
    pass


class NoopMetaType(OperatorMetatype):
    pass


def _build_one_to_one_graph():
    #  (input)
    #     |
    #  (bypass) <- should be bypassed
    #     |
    #  (output)
    graph = NNCFGraph()
    node_input = graph.add_nncf_node("Input_1", "any", AnyTestMetaType)
    node_bypass = graph.add_nncf_node("ByPass", "noop", NoopMetaType)
    node_output = graph.add_nncf_node("Output_1", "any", AnyTestMetaType)
    graph.add_edge_between_nncf_nodes(node_input.node_id, node_bypass.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass.node_id, node_output.node_id, (1,), 0, 0, Dtype.FLOAT)
    return graph


def _build_one_to_many_graph():
    #   (input)
    #      |
    #   (bypass)  <- should be bypassed
    #     /  \
    # (out1) (out2)
    graph = NNCFGraph()
    node_input = graph.add_nncf_node("Input_1", "any", AnyTestMetaType)
    node_bypass = graph.add_nncf_node("ByPass", "noop", NoopMetaType)
    node_output_1 = graph.add_nncf_node("Output_1", "any", AnyTestMetaType)
    node_output_2 = graph.add_nncf_node("Output_2", "any", AnyTestMetaType)
    graph.add_edge_between_nncf_nodes(node_input.node_id, node_bypass.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass.node_id, node_output_1.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass.node_id, node_output_2.node_id, (1,), 0, 0, Dtype.FLOAT)
    return graph


def _build_many_to_one_graph():
    # (input) (input)
    #      \  /
    #    (bypass)  <- should NOT be bypassed
    #       |
    #     (out1)
    graph = NNCFGraph()
    node_input_1 = graph.add_nncf_node("Input_1", "any", AnyTestMetaType)
    node_input_2 = graph.add_nncf_node("Input_2", "any", AnyTestMetaType)
    node_bypass = graph.add_nncf_node("ByPass", "noop", NoopMetaType)
    node_output_1 = graph.add_nncf_node("Output_1", "any", AnyTestMetaType)
    graph.add_edge_between_nncf_nodes(node_input_1.node_id, node_bypass.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_input_2.node_id, node_bypass.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass.node_id, node_output_1.node_id, (1,), 0, 0, Dtype.FLOAT)
    return graph


def _build_one_to_one_diff_dtype_graph():
    #  (input)
    #     | <float>
    #  (bypass)  <- should NOT be bypassed because of different dtypes
    #     | <int>
    #  (output)
    graph = NNCFGraph()
    node_input = graph.add_nncf_node("Input_1", "any", AnyTestMetaType)
    node_bypass = graph.add_nncf_node("ByPass", "noop", NoopMetaType)
    node_output = graph.add_nncf_node("Output_1", "any", AnyTestMetaType)
    graph.add_edge_between_nncf_nodes(node_input.node_id, node_bypass.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass.node_id, node_output.node_id, (1,), 0, 0, Dtype.INTEGER)
    return graph


def _build_one_to_one_diff_shape_graph():
    #  (input)
    #     | <(1,)>
    #  (bypass)  <- should NOT be bypassed because of different dtypes
    #     | <(1,1,)>
    #  (output)
    graph = NNCFGraph()
    node_input = graph.add_nncf_node("Input_1", "any", AnyTestMetaType)
    node_bypass = graph.add_nncf_node("ByPass", "noop", NoopMetaType)
    node_output = graph.add_nncf_node("Output_1", "any", AnyTestMetaType)
    graph.add_edge_between_nncf_nodes(node_input.node_id, node_bypass.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass.node_id, node_output.node_id, (1, 1), 0, 0, Dtype.FLOAT)
    return graph


def _build_one_to_none_graph():
    #  (input)
    #     |
    #  (bypass)  <- should be bypassed even if it has no output edges
    graph = NNCFGraph()
    node_input = graph.add_nncf_node("Input_1", "any", AnyTestMetaType)
    node_bypass = graph.add_nncf_node("ByPass", "noop", NoopMetaType)
    graph.add_edge_between_nncf_nodes(node_input.node_id, node_bypass.node_id, (1,), 0, 0, Dtype.FLOAT)
    return graph


def _build_none_to_one_graph():
    #  (bypass)  <- should be bypassed even if it has no input edges
    #     |
    #  (output)
    graph = NNCFGraph()
    node_bypass = graph.add_nncf_node("ByPass", "noop", NoopMetaType)
    node_output = graph.add_nncf_node("Output_1", "any", AnyTestMetaType)
    graph.add_edge_between_nncf_nodes(node_bypass.node_id, node_output.node_id, (1,), 0, 0, Dtype.FLOAT)
    return graph


def subgraph_with_same_noop_nodes_in_sequence():
    #  (input)
    #     |
    #  (bypass)
    #     |
    #  (bypass)
    #     |
    #  (output)
    graph = NNCFGraph()
    node_input = graph.add_nncf_node("Input_1", "any", AnyTestMetaType)
    node_bypass_1 = graph.add_nncf_node("ByPass_1", "noop", NoopMetaType)
    node_bypass_2 = graph.add_nncf_node("ByPass_2", "noop", NoopMetaType)
    node_output = graph.add_nncf_node("Output_1", "any", AnyTestMetaType)
    graph.add_edge_between_nncf_nodes(node_input.node_id, node_bypass_1.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass_1.node_id, node_bypass_2.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass_2.node_id, node_output.node_id, (1,), 0, 0, Dtype.FLOAT)
    return graph


def subgraph_with_same_noop_nodes():
    #      (input)
    #         |
    #      (bypass)
    #       /   \
    #  (bypass) (output)
    #     |
    #  (output)
    graph = NNCFGraph()
    node_input = graph.add_nncf_node("Input_1", "any", AnyTestMetaType)
    node_bypass_1 = graph.add_nncf_node("ByPass_1", "noop", NoopMetaType)
    node_bypass_2 = graph.add_nncf_node("ByPass_2", "noop", NoopMetaType)
    node_output_1 = graph.add_nncf_node("Output_1", "any", AnyTestMetaType)
    node_output_2 = graph.add_nncf_node("Output_2", "any", AnyTestMetaType)
    graph.add_edge_between_nncf_nodes(node_input.node_id, node_bypass_1.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass_1.node_id, node_bypass_2.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass_1.node_id, node_output_1.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass_2.node_id, node_output_2.node_id, (1,), 0, 0, Dtype.FLOAT)
    return graph


def subgraph_with_same_noop_nodes_one_output():
    #      (input)
    #         |
    #      (bypass)
    #       /   |
    #  (bypass) |
    #       \   |
    #     (output)
    graph = NNCFGraph()
    node_input = graph.add_nncf_node("Input_1", "any", AnyTestMetaType)
    node_bypass_1 = graph.add_nncf_node("ByPass_1", "noop", NoopMetaType)
    node_bypass_2 = graph.add_nncf_node("ByPass_2", "noop", NoopMetaType)
    node_output = graph.add_nncf_node("Output_1", "any", AnyTestMetaType)
    graph.add_edge_between_nncf_nodes(node_input.node_id, node_bypass_1.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass_1.node_id, node_bypass_2.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass_1.node_id, node_output.node_id, (1,), 0, 0, Dtype.FLOAT)
    graph.add_edge_between_nncf_nodes(node_bypass_2.node_id, node_output.node_id, (1,), 0, 0, Dtype.FLOAT)
    return graph


@pytest.mark.parametrize(
    "param",
    [
        ParamBypass(name="one_to_one", graph_builder=_build_one_to_one_graph),
        ParamBypass(name="one_to_many", graph_builder=_build_one_to_many_graph),
        ParamBypass(name="many_to_one", graph_builder=_build_many_to_one_graph),
        ParamBypass(name="one_to_one_diff_dtype", graph_builder=_build_one_to_one_diff_dtype_graph),
        ParamBypass(name="one_to_one_diff_shape", graph_builder=_build_one_to_one_diff_shape_graph),
        ParamBypass(name="one_to_none", graph_builder=_build_one_to_none_graph),
        ParamBypass(name="none_to_one", graph_builder=_build_none_to_one_graph),
        ParamBypass(name="same_noop_nodes_in_sequence", graph_builder=subgraph_with_same_noop_nodes_in_sequence),
        ParamBypass(name="subgraph_with_same_noop_nodes", graph_builder=subgraph_with_same_noop_nodes),
        ParamBypass(
            name="subgraph_with_same_noop_nodes_one_output", graph_builder=subgraph_with_same_noop_nodes_one_output
        ),
    ],
    ids=str,
)
def test_bypass_noop_operation_nodes(param: ParamBypass):
    graph = param.graph_builder()
    bypass_noop_operation_nodes(graph, [NoopMetaType])
    _check_graphs(param.ref_file, graph)
