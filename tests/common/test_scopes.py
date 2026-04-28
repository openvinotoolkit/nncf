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
import pytest

from nncf.common.graph import NNCFNode
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.scopes import get_not_matched_scopes
from nncf.common.scopes import propagate_ignored_constants_to_weighted_consumers
from nncf.scopes import IgnoredScope
from nncf.scopes import Subgraph
from nncf.scopes import get_difference_ignored_scope
from tests.common.quantization.metatypes import ConstantTestMetatype
from tests.common.quantization.metatypes import IdentityTestMetatype
from tests.common.quantization.metatypes import MatMulTestMetatype
from tests.common.quantization.metatypes import ParameterTestMetatype
from tests.common.quantization.metatypes import ReluTestMetatype
from tests.common.quantization.metatypes import ReshapeTestMetatype


@pytest.mark.parametrize(
    "scope, ref",
    [
        ("A", []),
        ("1", ["1"]),
        (["A", "B"], []),
        (["1", "2"], ["1", "2"]),
        ([r"{re}\d"], [r"{re}\d"]),
        ([r"{re}\w"], []),
        (["A", "B", "{re}.*", "1"], ["1"]),
        (IgnoredScope(names=["A", "B"]), []),
        (IgnoredScope(names=["1", "2"]), ["1", "2"]),
        (IgnoredScope(patterns=[r"\d"]), [r"{re}\d"]),
        (IgnoredScope(patterns=[r"\w"]), []),
    ],
)
def test_get_not_matched_scopes(scope, ref):
    node_lists = [
        NNCFNode({NNCFNode.ID_NODE_ATTR: 1, NNCFNode.NODE_NAME_ATTR: "A"}),
        NNCFNode({NNCFNode.ID_NODE_ATTR: 2, NNCFNode.NODE_NAME_ATTR: "B"}),
    ]
    not_matched = get_not_matched_scopes(scope, node_lists)
    assert not set(not_matched) - set(ref)


@pytest.mark.parametrize(
    "scope_1, scope_2, ref",
    (
        (
            IgnoredScope(
                names=["A_name", "B_name"],
                patterns=["A_pattern", "B_pattern"],
                types=["A_type", "B_type"],
                subgraphs=[
                    Subgraph(inputs=["A_input"], outputs=["A_output"]),
                    Subgraph(inputs=["B_input"], outputs=["B_output"]),
                ],
            ),
            IgnoredScope(
                names=["B_name", "C_name"],
                patterns=["B_pattern", "C_pattern"],
                types=["B_type", "C_type"],
                subgraphs=[
                    Subgraph(inputs=["B_input"], outputs=["B_output"]),
                    Subgraph(inputs=["C_input"], outputs=["C_output"]),
                ],
            ),
            IgnoredScope(
                names=["A_name"],
                patterns=["A_pattern"],
                types=["A_type"],
                subgraphs=[Subgraph(inputs=["A_input"], outputs=["A_output"])],
            ),
        ),
    ),
)
def test_ignored_scope_diff(scope_1, scope_2, ref):
    assert get_difference_ignored_scope(scope_1, scope_2) == ref


def _edge(graph: NNCFGraph, src: NNCFNode, dst: NNCFNode, input_port_id: int = 0) -> None:
    graph.add_edge_between_nncf_nodes(
        src.node_id,
        dst.node_id,
        tensor_shape=[1],
        input_port_id=input_port_id,
        output_port_id=0,
        dtype=Dtype.FLOAT,
    )


def _build_propagation_graph() -> NNCFGraph:
    """Graph exercising direct, passthrough, intermediate-op, and shared-weight cases.

    Layout:

        parameter -> matmul_a <- weight_a
                   -> reshape -> matmul_b <- weight_b (passthrough before op)
                   -> identity_chain_1 -> identity_chain_2 -> matmul_c <- weight_c
                                                           -> matmul_c2 <- weight_c (shared weight)
                   -> relu_before -> matmul_d <- weight_d (intermediate compute op)
        weight_e -> matmul_e (feeds nothing further - simple direct case)

    Relevant node metatypes:
      - Constants: weight_a..weight_e
      - Reshape/Identity: passthroughs used in IR-like patterns
      - MatMul: weighted ops
      - Relu: an arbitrary compute op that should NOT propagate
      - Parameter: a model input (source-only, not a Constant)
    """
    graph = NNCFGraph()
    parameter = graph.add_nncf_node("parameter", "parameter", ParameterTestMetatype)

    # Direct weight -> matmul_a
    weight_a = graph.add_nncf_node("weight_a", "constant", ConstantTestMetatype)
    matmul_a = graph.add_nncf_node("matmul_a", "matmul", MatMulTestMetatype)
    _edge(graph, parameter, matmul_a, input_port_id=0)
    _edge(graph, weight_a, matmul_a, input_port_id=1)

    # weight -> reshape -> matmul_b (single passthrough)
    weight_b = graph.add_nncf_node("weight_b", "constant", ConstantTestMetatype)
    reshape_b = graph.add_nncf_node("reshape_b", "reshape", ReshapeTestMetatype)
    matmul_b = graph.add_nncf_node("matmul_b", "matmul", MatMulTestMetatype)
    _edge(graph, parameter, matmul_b, input_port_id=0)
    _edge(graph, weight_b, reshape_b)
    _edge(graph, reshape_b, matmul_b, input_port_id=1)

    # weight -> identity -> identity -> matmul (two passthroughs)
    # same weight feeds two MatMuls (shared-weight case).
    weight_c = graph.add_nncf_node("weight_c", "constant", ConstantTestMetatype)
    id_c_1 = graph.add_nncf_node("identity_c_1", "identity", IdentityTestMetatype)
    id_c_2 = graph.add_nncf_node("identity_c_2", "identity", IdentityTestMetatype)
    matmul_c = graph.add_nncf_node("matmul_c", "matmul", MatMulTestMetatype)
    matmul_c2 = graph.add_nncf_node("matmul_c2", "matmul", MatMulTestMetatype)
    _edge(graph, parameter, matmul_c, input_port_id=0)
    _edge(graph, parameter, matmul_c2, input_port_id=0)
    _edge(graph, weight_c, id_c_1)
    _edge(graph, id_c_1, id_c_2)
    _edge(graph, id_c_2, matmul_c, input_port_id=1)
    _edge(graph, id_c_2, matmul_c2, input_port_id=1)

    # relu (intermediate compute) -> matmul. Must NOT propagate if user ignores relu.
    weight_d = graph.add_nncf_node("weight_d", "constant", ConstantTestMetatype)
    relu_before = graph.add_nncf_node("relu_before", "relu", ReluTestMetatype)
    matmul_d = graph.add_nncf_node("matmul_d", "matmul", MatMulTestMetatype)
    _edge(graph, parameter, relu_before)
    _edge(graph, relu_before, matmul_d, input_port_id=0)
    _edge(graph, weight_d, matmul_d, input_port_id=1)

    # Plain direct case used as a sanity control.
    weight_e = graph.add_nncf_node("weight_e", "constant", ConstantTestMetatype)
    matmul_e = graph.add_nncf_node("matmul_e", "matmul", MatMulTestMetatype)
    _edge(graph, parameter, matmul_e, input_port_id=0)
    _edge(graph, weight_e, matmul_e, input_port_id=1)

    return graph


_WEIGHTED = [MatMulTestMetatype]
_CONST = [ConstantTestMetatype]
_PASSTHROUGH_MT = [ReshapeTestMetatype, IdentityTestMetatype]


@pytest.mark.parametrize(
    "ignored_names, expected_added",
    [
        # Direct Constant -> weighted op.
        ({"weight_a"}, {"matmul_a"}),
        ({"weight_e"}, {"matmul_e"}),
        # Passthrough via one Reshape.
        ({"weight_b"}, {"matmul_b"}),
        # Two passthrough Identity nodes + shared weight feeds two matmuls.
        ({"weight_c"}, {"matmul_c", "matmul_c2"}),
        # Intermediate compute op (ReLU) feeding a weighted consumer must NOT propagate.
        ({"relu_before"}, set()),
        # Parameter is a source node but NOT a Constant - no propagation.
        ({"parameter"}, set()),
        # Already-weighted op name is untouched.
        ({"matmul_a"}, set()),
        # Multiple Constants at once.
        ({"weight_a", "weight_b"}, {"matmul_a", "matmul_b"}),
        # Mixed: Constant name and its consumer - idempotent, no duplicate add.
        ({"weight_a", "matmul_a"}, set()),
        # Empty input - empty result.
        (set(), set()),
    ],
)
def test_propagate_ignored_constants_to_weighted_consumers(ignored_names, expected_added):
    graph = _build_propagation_graph()
    result = propagate_ignored_constants_to_weighted_consumers(
        ignored_names,
        graph,
        weighted_metatypes=_WEIGHTED,
        constant_metatypes=_CONST,
        passthrough_metatypes=_PASSTHROUGH_MT,
    )
    assert result == ignored_names | expected_added


def test_propagate_ignored_constants_respects_passthrough_node_types():
    """If a node that would otherwise halt traversal is listed in
    ``passthrough_node_types``, BFS should continue through it."""
    graph = NNCFGraph()
    weight = graph.add_nncf_node("w", "constant", ConstantTestMetatype)
    # A "relu" would normally stop the walk, but we mark it passthrough by node_type here.
    relu = graph.add_nncf_node("relu_passthrough", "symmetric_quantize", ReluTestMetatype)
    matmul = graph.add_nncf_node("mm", "matmul", MatMulTestMetatype)
    _edge(graph, weight, relu)
    _edge(graph, relu, matmul)
    result = propagate_ignored_constants_to_weighted_consumers(
        {"w"},
        graph,
        weighted_metatypes=_WEIGHTED,
        constant_metatypes=_CONST,
        passthrough_node_types=["symmetric_quantize"],
    )
    assert result == {"w", "mm"}


def test_propagate_ignored_constants_stops_at_non_passthrough_non_weighted():
    """If a non-weighted, non-passthrough node sits between the Constant and the
    weighted op, traversal halts there and the weighted op is NOT added."""
    graph = NNCFGraph()
    weight = graph.add_nncf_node("w", "constant", ConstantTestMetatype)
    relu = graph.add_nncf_node("relu", "relu", ReluTestMetatype)
    matmul = graph.add_nncf_node("mm", "matmul", MatMulTestMetatype)
    _edge(graph, weight, relu)
    _edge(graph, relu, matmul)
    result = propagate_ignored_constants_to_weighted_consumers(
        {"w"},
        graph,
        weighted_metatypes=_WEIGHTED,
        constant_metatypes=_CONST,
        passthrough_metatypes=[ReshapeTestMetatype],  # does NOT include Relu
    )
    assert result == {"w"}


def test_propagate_ignored_constants_empty_metatype_lists():
    """With no constant_metatypes, nothing propagates even for source-only nodes."""
    graph = NNCFGraph()
    weight = graph.add_nncf_node("w", "constant", ConstantTestMetatype)
    matmul = graph.add_nncf_node("mm", "matmul", MatMulTestMetatype)
    _edge(graph, weight, matmul)
    result = propagate_ignored_constants_to_weighted_consumers(
        {"w"},
        graph,
        weighted_metatypes=_WEIGHTED,
        constant_metatypes=[],
    )
    assert result == {"w"}
