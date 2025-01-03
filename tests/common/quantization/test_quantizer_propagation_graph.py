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
from abc import ABC
from abc import abstractmethod
from collections import Counter
from collections import namedtuple
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pytest

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.insertion_point_graph import PostHookInsertionPoint
from nncf.common.insertion_point_graph import PreHookInsertionPoint
from nncf.common.quantization.quantizer_propagation.graph import QuantizerPropagationStateGraph as QPSG
from nncf.common.quantization.quantizer_propagation.structs import PropagationPath
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.quantization.quantizer_propagation.structs import QuantizerPropagationStateGraphNodeType
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizerSetup
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import UnifiedScaleType
from tests.common.quantization.metatypes import WEIGHT_LAYER_METATYPES
from tests.common.quantization.metatypes import CatTestMetatype
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.mock_graphs import get_ip_graph_for_test
from tests.common.quantization.mock_graphs import get_mock_nncf_node_attrs
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph
from tests.common.quantization.mock_graphs import get_two_branch_mock_model_graph
from tests.common.quantization.mock_graphs import mark_input_ports_lexicographically_based_on_input_node_key


def get_edge_paths(graph, start_node_key, finish_node_key) -> List[List[Tuple]]:
    node_paths = list(nx.all_simple_paths(graph, start_node_key, finish_node_key))
    edge_paths = []
    for path in node_paths:
        edge_paths.append([(path[i], path[i + 1]) for i in range(0, len(path) - 1)])
    return edge_paths


def get_edge_paths_for_propagation(graph, start_node_key, finish_node_key) -> List[List[Tuple]]:
    paths = get_edge_paths(graph, start_node_key, finish_node_key)
    return [list(reversed(path)) for path in paths]


class TestQuantizerPropagationStateGraph:
    @staticmethod
    @pytest.fixture()
    def mock_qp_graph():
        ip_graph = get_ip_graph_for_test(get_two_branch_mock_model_graph())
        qpsg = QPSG(ip_graph)

        qpsg.nodes["5 /F_0"][QPSG.OPERATOR_METATYPE_NODE_ATTR] = CatTestMetatype
        qpsg.nodes["6 /G_0"][QPSG.OPERATOR_METATYPE_NODE_ATTR] = Conv2dTestMetatype
        qpsg.skip_check = False
        yield qpsg
        if not qpsg.skip_check:
            qpsg.run_consistency_check()

    def test_build_quantizer_propagation_state_graph_from_ip_graph(self):
        ip_graph = get_ip_graph_for_test(get_two_branch_mock_model_graph())
        quant_prop_graph = QPSG(ip_graph)
        assert len(ip_graph.nodes) == len(quant_prop_graph.nodes)
        assert len(ip_graph.edges) == len(quant_prop_graph.edges)

        for ip_graph_node_key, ip_graph_node in ip_graph.nodes.items():
            qpg_node = quant_prop_graph.nodes[ip_graph_node_key]
            assert qpg_node[QPSG.NODE_TYPE_NODE_ATTR] == QPSG.ipg_node_type_to_qpsg_node_type(
                ip_graph_node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
            )
            qpg_node_type = qpg_node[QPSG.NODE_TYPE_NODE_ATTR]
            if QPSG.is_insertion_point(qpg_node_type):
                assert qpg_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None
                assert not qpg_node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                qip = qpg_node[QPSG.QUANT_INSERTION_POINT_DATA_NODE_ATTR]
                ip = ip_graph_node[InsertionPointGraph.INSERTION_POINT_NODE_ATTR]
                assert ip.target_node_name == qip.target_node_name
                assert isinstance(qip, ActivationQuantizationInsertionPoint)
                if isinstance(ip, PreHookInsertionPoint):
                    assert ip.input_port_id == qip.input_port_id
                else:
                    assert isinstance(ip, PostHookInsertionPoint)
            elif qpg_node_type == QuantizerPropagationStateGraphNodeType.OPERATOR:
                assert not qpg_node[QPSG.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR]
                assert qpg_node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] == QuantizationTrait.NON_QUANTIZABLE
                assert not qpg_node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        for from_node, to_node, edge_data in ip_graph.edges(data=True):
            qpg_edge_data = quant_prop_graph.edges[from_node, to_node]
            assert not qpg_edge_data[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            for key, value in edge_data.items():
                assert qpg_edge_data[key] == value

        quant_prop_graph.run_consistency_check()

    def test_add_propagating_quantizer(self, mock_qp_graph):
        ref_qconf_list = [QuantizerConfig(), QuantizerConfig(num_bits=6)]

        target_node_key = "5 /F_0"
        target_ip_node_key = InsertionPointGraph.get_pre_hook_node_key(target_node_key)
        prop_quant = mock_qp_graph.add_propagating_quantizer(ref_qconf_list, target_ip_node_key)
        assert prop_quant.potential_quant_configs == ref_qconf_list
        assert prop_quant.current_location_node_key == target_ip_node_key
        assert prop_quant.affected_ip_nodes == {target_ip_node_key}
        assert prop_quant.last_accepting_location_node_key is None
        assert prop_quant.affected_edges == {(target_ip_node_key, target_node_key)}
        assert not prop_quant.propagation_path

        for node_key, node in mock_qp_graph.nodes.items():
            if node_key == target_node_key:
                assert node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]
            elif node_key == target_ip_node_key:
                assert node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] == prop_quant
            else:
                assert not node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                if QPSG.is_insertion_point(node[QPSG.NODE_TYPE_NODE_ATTR]):
                    assert node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None

            target_ip_node = mock_qp_graph.nodes[target_ip_node_key]
            assert target_ip_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] == prop_quant

        for from_node, to_node, edge_data in mock_qp_graph.edges.data():
            if (from_node, to_node) == (target_ip_node_key, target_node_key):
                assert edge_data[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]
            else:
                assert not edge_data[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        with pytest.raises(nncf.InternalError):
            _ = mock_qp_graph.add_propagating_quantizer(
                ref_qconf_list, InsertionPointGraph.get_post_hook_node_key(target_node_key)
            )

    START_IP_NODES_AND_PATHS_TO_DOMINATING_IP_NODES = [
        # Non-branching case - starting from "4 /E_0" pre-hook
        (
            InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
            [
                [
                    (
                        InsertionPointGraph.get_post_hook_node_key("2 /C_0"),
                        InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
                    )
                ]
            ],
        ),
        # Non-branching case - starting from "2 /C_0" post-hook
        (
            InsertionPointGraph.get_post_hook_node_key("2 /C_0"),
            [
                [
                    ("2 /C_0", InsertionPointGraph.get_post_hook_node_key("2 /C_0")),
                    (InsertionPointGraph.get_pre_hook_node_key("2 /C_0"), "2 /C_0"),
                ]
            ],
        ),
        # Branching case - starting from "5 /F_0" pre-hook port 0
        (
            InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
            [
                [
                    (
                        InsertionPointGraph.get_post_hook_node_key("4 /E_0"),
                        InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
                    )
                ]
            ],
        ),
        # Branching case - starting from "5 /F_0" pre-hook port 1
        (
            InsertionPointGraph.get_pre_hook_node_key("5 /F_0", input_port_id=1),
            [
                [
                    (
                        InsertionPointGraph.get_post_hook_node_key("3 /D_0"),
                        InsertionPointGraph.get_pre_hook_node_key("5 /F_0", input_port_id=1),
                    )
                ]
            ],
        ),
    ]

    @staticmethod
    @pytest.fixture(params=START_IP_NODES_AND_PATHS_TO_DOMINATING_IP_NODES)
    def start_ip_node_and_path_to_dominating_node(request):
        return request.param

    def test_get_paths_to_immediately_dominating_insertion_points(
        self, start_ip_node_and_path_to_dominating_node, mock_qp_graph
    ):
        start_node = start_ip_node_and_path_to_dominating_node[0]
        ref_paths = start_ip_node_and_path_to_dominating_node[1]
        test_paths = mock_qp_graph.get_paths_to_immediately_dominating_insertion_points(start_node)

        def get_cat_path_list(path_list: List[List[Tuple[str, str]]]):
            str_paths = [[str(edge[0]) + " -> " + str(edge[1]) for edge in path] for path in path_list]
            cat_paths = [";".join(path) for path in str_paths]
            return cat_paths

        assert Counter(get_cat_path_list(ref_paths)) == Counter(get_cat_path_list(test_paths))

    class DomIPGroupedByUnifiedScalesTestStruct:
        def __init__(self, start_ip_node_key: str, ref_groups_vs_paths: Dict[Optional[int], List[PropagationPath]]):
            self.start_ip_node_key = start_ip_node_key
            self.ref_groups_vs_paths = ref_groups_vs_paths

    START_IP_NODES_AND_GROUPED_PATHS_TO_DOM_IP_NODES = [
        DomIPGroupedByUnifiedScalesTestStruct(
            start_ip_node_key=InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
            ref_groups_vs_paths={
                None: [
                    [
                        (
                            InsertionPointGraph.get_post_hook_node_key("2 /C_0"),
                            InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
                        )
                    ]
                ]
            },
        ),
        DomIPGroupedByUnifiedScalesTestStruct(
            start_ip_node_key=InsertionPointGraph.get_post_hook_node_key("2 /C_0"),
            ref_groups_vs_paths={
                None: [
                    [
                        ("2 /C_0", InsertionPointGraph.get_post_hook_node_key("2 /C_0")),
                        (InsertionPointGraph.get_pre_hook_node_key("2 /C_0"), "2 /C_0"),
                    ]
                ]
            },
        ),
        DomIPGroupedByUnifiedScalesTestStruct(
            start_ip_node_key=InsertionPointGraph.get_pre_hook_node_key("5 /F_0", input_port_id=1),
            ref_groups_vs_paths={
                None: [
                    [
                        (
                            InsertionPointGraph.get_post_hook_node_key("3 /D_0"),
                            InsertionPointGraph.get_pre_hook_node_key("5 /F_0", input_port_id=1),
                        )
                    ]
                ]
            },
        ),
        DomIPGroupedByUnifiedScalesTestStruct(
            start_ip_node_key=InsertionPointGraph.get_post_hook_node_key("5 /F_0"),
            ref_groups_vs_paths={
                0: [
                    [
                        ("5 /F_0", InsertionPointGraph.get_post_hook_node_key("5 /F_0")),
                        (InsertionPointGraph.get_pre_hook_node_key("5 /F_0", input_port_id=0), "5 /F_0"),
                    ],
                    [
                        ("5 /F_0", InsertionPointGraph.get_post_hook_node_key("5 /F_0")),
                        (InsertionPointGraph.get_pre_hook_node_key("5 /F_0", input_port_id=1), "5 /F_0"),
                    ],
                ]
            },
        ),
    ]

    @staticmethod
    @pytest.fixture(params=START_IP_NODES_AND_GROUPED_PATHS_TO_DOM_IP_NODES)
    def start_ip_node_and_dom_node_grouped_paths(request):
        return request.param

    def test_get_paths_to_immediately_dominating_insertion_points_grouped_by_unified_scales(
        self, mock_qp_graph, start_ip_node_and_dom_node_grouped_paths: DomIPGroupedByUnifiedScalesTestStruct
    ):
        start_node_key = start_ip_node_and_dom_node_grouped_paths.start_ip_node_key
        ref_groups_vs_paths = start_ip_node_and_dom_node_grouped_paths.ref_groups_vs_paths
        test_groups_vs_paths = (
            mock_qp_graph.get_paths_to_immediately_dominating_insertion_points_grouped_by_unified_scales(
                start_node_key, {CatTestMetatype}, {CatTestMetatype: WEIGHT_LAYER_METATYPES}
            )
        )

        def get_cat_path_list(path_list: List[List[Tuple[str, str]]]):
            str_paths = [[str(edge[0]) + " -> " + str(edge[1]) for edge in path] for path in path_list]
            cat_paths = [";".join(path) for path in str_paths]
            return cat_paths

        processed_ref_groups = {k: get_cat_path_list(v) for k, v in ref_groups_vs_paths.items()}
        processed_test_groups = {k: get_cat_path_list(v) for k, v in test_groups_vs_paths.items()}

        assert processed_ref_groups == processed_test_groups

    START_TARGET_NODES = [
        (InsertionPointGraph.get_pre_hook_node_key("7 /H_0"), InsertionPointGraph.get_post_hook_node_key("6 /G_0")),
        (InsertionPointGraph.get_pre_hook_node_key("7 /H_0"), InsertionPointGraph.get_pre_hook_node_key("5 /F_0")),
        (
            InsertionPointGraph.get_pre_hook_node_key("5 /F_0", input_port_id=1),
            InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
        ),
        (InsertionPointGraph.get_pre_hook_node_key("5 /F_0"), InsertionPointGraph.get_post_hook_node_key("1 /B_0")),
    ]

    @staticmethod
    @pytest.fixture(params=START_TARGET_NODES)
    def start_target_nodes(request):
        return request.param

    @pytest.mark.dependency(name="propagate_via_path")
    def test_quantizers_can_propagate_via_path(self, start_target_nodes, mock_qp_graph):
        start_ip_node_key = start_target_nodes[0]
        target_ip_node_key = start_target_nodes[1]

        # From "target" to "start" since propagation direction is inverse to edge direction
        rev_paths = get_edge_paths_for_propagation(mock_qp_graph, target_ip_node_key, start_ip_node_key)

        assert rev_paths
        for path in rev_paths:
            working_graph = deepcopy(mock_qp_graph)
            ref_prop_quant = working_graph.add_propagating_quantizer([QuantizerConfig()], start_ip_node_key)
            ref_affected_edges = deepcopy(ref_prop_quant.affected_edges)
            ref_affected_edges.update(set(path))
            ref_affected_ip_nodes = deepcopy(ref_prop_quant.affected_ip_nodes)
            prop_quant = working_graph.propagate_quantizer_via_path(ref_prop_quant, path)
            final_node_key, _ = path[-1]
            for from_node_key, to_node_key in path:
                edge_data = working_graph.edges[from_node_key, to_node_key]
                assert edge_data[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [ref_prop_quant]
                to_node = working_graph.nodes[to_node_key]
                if QPSG.is_insertion_point(to_node[QPSG.NODE_TYPE_NODE_ATTR]):
                    assert to_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None
                from_node = working_graph.nodes[from_node_key]
                if QPSG.is_insertion_point(from_node[QPSG.NODE_TYPE_NODE_ATTR]):
                    ref_affected_ip_nodes.add(from_node_key)
            working_graph.run_consistency_check()

            final_node_key, _ = path[-1]
            final_node = working_graph.nodes[final_node_key]
            assert final_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] == ref_prop_quant

            assert prop_quant.current_location_node_key == final_node_key
            assert prop_quant.propagation_path == path
            assert prop_quant.affected_edges == ref_affected_edges
            assert prop_quant.affected_ip_nodes == ref_affected_ip_nodes

    START_TARGET_ACCEPTING_NODES = [
        (
            InsertionPointGraph.get_pre_hook_node_key("7 /H_0"),
            InsertionPointGraph.get_pre_hook_node_key("6 /G_0"),
            InsertionPointGraph.get_post_hook_node_key("6 /G_0"),
        ),
        (
            InsertionPointGraph.get_pre_hook_node_key("6 /G_0"),
            InsertionPointGraph.get_post_hook_node_key("5 /F_0"),
            InsertionPointGraph.get_post_hook_node_key("5 /F_0"),
        ),
        (
            InsertionPointGraph.get_pre_hook_node_key("5 /F_0", input_port_id=1),
            InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
            InsertionPointGraph.get_post_hook_node_key("3 /D_0"),
        ),
        (
            InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
            InsertionPointGraph.get_pre_hook_node_key("1 /B_0"),
            InsertionPointGraph.get_post_hook_node_key("1 /B_0"),
        ),
    ]

    @staticmethod
    @pytest.fixture(params=START_TARGET_ACCEPTING_NODES)
    def start_target_accepting_nodes(request):
        return request.param

    @pytest.mark.dependency(depends=["propagate_via_path"])
    def test_backtrack_propagation_until_accepting_location(self, start_target_accepting_nodes, mock_qp_graph):
        start_ip_node_key = start_target_accepting_nodes[0]
        target_ip_node_key = start_target_accepting_nodes[1]
        forced_last_accepting_location = start_target_accepting_nodes[2]

        prop_quant = mock_qp_graph.add_propagating_quantizer([QuantizerConfig()], start_ip_node_key)
        ref_affected_edges = deepcopy(prop_quant.affected_edges)

        # Here, the tested graph should have such a structure that there is only one path from target to start
        path = get_edge_paths_for_propagation(mock_qp_graph, target_ip_node_key, start_ip_node_key)[0]
        prop_quant = mock_qp_graph.propagate_quantizer_via_path(prop_quant, path)
        prop_quant.last_accepting_location_node_key = forced_last_accepting_location
        if forced_last_accepting_location is not None:
            resulting_path = get_edge_paths_for_propagation(
                mock_qp_graph, forced_last_accepting_location, start_ip_node_key
            )[0]
            ref_affected_edges.update(set(resulting_path))

        prop_quant = mock_qp_graph.backtrack_propagation_until_accepting_location(prop_quant)

        assert prop_quant.current_location_node_key == forced_last_accepting_location
        assert prop_quant.affected_edges == ref_affected_edges
        assert prop_quant.propagation_path == resulting_path

        target_node = mock_qp_graph.nodes[target_ip_node_key]
        accepting_node = mock_qp_graph.nodes[forced_last_accepting_location]
        if forced_last_accepting_location != target_ip_node_key:
            assert target_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None
            assert target_ip_node_key not in prop_quant.affected_ip_nodes
        assert accepting_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] == prop_quant

    @pytest.mark.dependency(depends=["propagate_via_path"])
    def test_clone_propagating_quantizer(self, mock_qp_graph, start_target_nodes):
        start_ip_node_key = start_target_nodes[0]
        target_ip_node_key = start_target_nodes[1]

        # From "target" to "start" since propagation direction is inverse to edge direction
        # Only take one path out of possible paths for this test
        rev_path = get_edge_paths_for_propagation(mock_qp_graph, target_ip_node_key, start_ip_node_key)[0]

        ref_prop_quant = mock_qp_graph.add_propagating_quantizer([QuantizerConfig()], start_ip_node_key)

        prop_quant = mock_qp_graph.propagate_quantizer_via_path(ref_prop_quant, rev_path)

        cloned_prop_quant = mock_qp_graph.clone_propagating_quantizer(prop_quant)

        assert cloned_prop_quant.affected_ip_nodes == prop_quant.affected_ip_nodes
        assert cloned_prop_quant.affected_edges == prop_quant.affected_edges
        assert cloned_prop_quant.propagation_path == prop_quant.propagation_path
        assert cloned_prop_quant.current_location_node_key == prop_quant.current_location_node_key

        for ip_node_key in prop_quant.affected_ip_nodes:
            node = mock_qp_graph.nodes[ip_node_key]
            assert cloned_prop_quant in node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
        for from_node_key, to_node_key in prop_quant.affected_edges:
            edge = mock_qp_graph.edges[from_node_key, to_node_key]
            assert cloned_prop_quant in edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        # The cloned quantizer had not been put into any IP (cannot have multiple PQs in one IP right now)
        mock_qp_graph.skip_check = True

    START_TARGET_NODES_FOR_TWO_QUANTIZERS = [
        (
            InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
            InsertionPointGraph.get_post_hook_node_key("2 /C_0"),
            InsertionPointGraph.get_pre_hook_node_key("7 /H_0"),
            InsertionPointGraph.get_post_hook_node_key("6 /G_0"),
        ),
        (
            InsertionPointGraph.get_pre_hook_node_key("2 /C_0"),
            InsertionPointGraph.get_post_hook_node_key("0 /A_0"),
            InsertionPointGraph.get_pre_hook_node_key("7 /H_0"),
            InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
        ),
        # Simulated quantizer merging result
        (
            InsertionPointGraph.get_pre_hook_node_key("6 /G_0"),
            InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
            InsertionPointGraph.get_pre_hook_node_key("6 /G_0"),
            InsertionPointGraph.get_post_hook_node_key("3 /D_0"),
        ),
    ]

    @staticmethod
    @pytest.fixture(params=START_TARGET_NODES_FOR_TWO_QUANTIZERS)
    def start_target_nodes_for_two_quantizers(request):
        return request.param

    @pytest.mark.dependency(depends=["propagate_via_path"])
    def test_remove_propagating_quantizer(self, mock_qp_graph, start_target_nodes_for_two_quantizers):
        start_ip_node_key_remove = start_target_nodes_for_two_quantizers[0]
        target_ip_node_key_remove = start_target_nodes_for_two_quantizers[1]

        start_ip_node_key_keep = start_target_nodes_for_two_quantizers[2]
        target_ip_node_key_keep = start_target_nodes_for_two_quantizers[3]

        # From "target" to "start" since propagation direction is inverse to edge direction
        # Only take one path out of possible paths for this test
        rev_path_remove = get_edge_paths_for_propagation(
            mock_qp_graph, target_ip_node_key_remove, start_ip_node_key_remove
        )[0]
        rev_path_keep = get_edge_paths_for_propagation(mock_qp_graph, target_ip_node_key_keep, start_ip_node_key_keep)[
            0
        ]

        prop_quant_to_remove = mock_qp_graph.add_propagating_quantizer([QuantizerConfig()], start_ip_node_key_remove)
        prop_quant_to_remove = mock_qp_graph.propagate_quantizer_via_path(prop_quant_to_remove, rev_path_remove)

        prop_quant_to_keep = mock_qp_graph.add_propagating_quantizer([QuantizerConfig()], start_ip_node_key_keep)

        prop_quant_to_keep = mock_qp_graph.propagate_quantizer_via_path(prop_quant_to_keep, rev_path_keep)

        affected_ip_nodes = deepcopy(prop_quant_to_remove.affected_ip_nodes)
        affected_op_nodes = deepcopy(prop_quant_to_remove.affected_operator_nodes)
        affected_edges = deepcopy(prop_quant_to_keep.affected_edges)
        last_location = prop_quant_to_remove.current_location_node_key
        ref_quant_to_keep_state_dict = deepcopy(prop_quant_to_keep.__dict__)

        mock_qp_graph.remove_propagating_quantizer(prop_quant_to_remove)

        last_node = mock_qp_graph.nodes[last_location]
        assert last_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None

        for ip_node_key in affected_ip_nodes:
            node = mock_qp_graph.nodes[ip_node_key]
            assert prop_quant_to_remove not in node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        for op_node_key in affected_op_nodes:
            node = mock_qp_graph.nodes[op_node_key]
            assert prop_quant_to_remove not in node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        for from_node_key, to_node_key in affected_edges:
            edge = mock_qp_graph.edges[from_node_key, to_node_key]
            assert prop_quant_to_remove not in edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        assert prop_quant_to_keep.__dict__ == ref_quant_to_keep_state_dict

        for ip_node_key in prop_quant_to_keep.affected_ip_nodes:
            node = mock_qp_graph.nodes[ip_node_key]
            assert prop_quant_to_keep in node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

        for from_node_key, to_node_key in prop_quant_to_keep.affected_edges:
            edge = mock_qp_graph.edges[from_node_key, to_node_key]
            assert prop_quant_to_keep in edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

    QUANTIZABLE_NODES_START_NODES_DOMINATED_NODES = [
        (
            ["3 /D_0", "4 /E_0", "5 /F_0"],
            {
                "1 /B_0": {"3 /D_0", "4 /E_0"},
                InsertionPointGraph.get_pre_hook_node_key("1 /B_0"): {"3 /D_0", "4 /E_0"},
                "4 /E_0": {"5 /F_0"},
                InsertionPointGraph.get_post_hook_node_key("3 /D_0"): {"5 /F_0"},
                "0 /A_0": {"3 /D_0", "4 /E_0"},
                InsertionPointGraph.get_pre_hook_node_key("6 /G_0"): set(),
            },
        ),
        (
            ["2 /C_0", "4 /E_0", "7 /H_0"],
            {
                InsertionPointGraph.get_pre_hook_node_key("2 /C_0"): {"2 /C_0"},
                InsertionPointGraph.get_post_hook_node_key("2 /C_0"): {"4 /E_0"},
                "3 /D_0": {"7 /H_0"},
                # corner case - has a branch without quantizers
                InsertionPointGraph.get_pre_hook_node_key("1 /B_0"): {"2 /C_0", "7 /H_0"},
                InsertionPointGraph.get_post_hook_node_key("7 /H_0"): set(),
            },
        ),
    ]

    @staticmethod
    @pytest.fixture(params=QUANTIZABLE_NODES_START_NODES_DOMINATED_NODES)
    def dominated_nodes_test_struct(request):
        return request.param

    @staticmethod
    def mark_nodes_with_traits(qpsg: QPSG, node_keys_vs_traits_dict: Dict[str, QuantizationTrait]) -> QPSG:
        for node_key, node in qpsg.nodes.items():
            if node_key in node_keys_vs_traits_dict:
                node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = node_keys_vs_traits_dict[node_key]
        return qpsg

    def test_get_quantizable_op_nodes_immediately_dominated_by_node(self, mock_qp_graph, dominated_nodes_test_struct):
        nodes_to_mark_as_quantizable = dominated_nodes_test_struct[0]

        node_keys_vs_trait_dict = {}
        for node_key in mock_qp_graph.nodes:
            node_keys_vs_trait_dict[node_key] = QuantizationTrait.QUANTIZATION_AGNOSTIC

        traits_to_mark_with = [QuantizationTrait.INPUTS_QUANTIZABLE, QuantizationTrait.NON_QUANTIZABLE]

        for trait in traits_to_mark_with:
            for node_key in nodes_to_mark_as_quantizable:
                node_keys_vs_trait_dict[node_key] = trait

            mock_qp_graph = self.mark_nodes_with_traits(mock_qp_graph, node_keys_vs_trait_dict)
            for start_node_key, ref_dominated_quantizable_nodes_set in dominated_nodes_test_struct[1].items():
                dominated_quantizable_nodes_list = (
                    mock_qp_graph.get_non_quant_agnostic_op_nodes_immediately_dominated_by_node(start_node_key)
                )
                assert set(dominated_quantizable_nodes_list) == ref_dominated_quantizable_nodes_set

    @staticmethod
    def get_model_graph() -> NNCFGraph:
        mock_graph = nx.DiGraph()

        #      (0 /A_0)
        #         |
        #      (1 /B_0)
        #    /        \
        # (2 /C_0)     (3 /D_0)
        #    |         |
        # (5 /F_0)     (4 /E_0)
        #
        #

        node_keys = ["A", "B", "C", "D", "E", "F"]
        for node_key in node_keys:
            mock_node_attrs = get_mock_nncf_node_attrs(op_name=node_key)
            mock_graph.add_node(node_key, **mock_node_attrs)

        mock_graph.add_edges_from([("A", "B"), ("B", "C"), ("B", "D"), ("D", "E"), ("C", "F")])
        mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
        return get_nncf_graph_from_mock_nx_graph(mock_graph)

    StateQuantizerTestStruct = namedtuple(
        "StateQuantizerTestStruct",
        (
            "init_node_to_trait_and_configs_dict",
            "starting_quantizer_ip_node",
            "target_node_for_quantizer",
            "is_merged",
            "prop_path",
        ),
    )

    SetQuantizersTestStruct = namedtuple("SetQuantizersTestStruct", ("start_set_quantizers", "expected_set_quantizers"))

    MERGE_QUANTIZER_INTO_PATH_TEST_CASES = [
        SetQuantizersTestStruct(
            start_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict={
                        "4 /E_0": (QuantizationTrait.INPUTS_QUANTIZABLE, [QuantizerConfig()])
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key("1 /B_0"),
                    is_merged=False,
                    prop_path=None,
                ),
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict={
                        "5 /F_0": (QuantizationTrait.INPUTS_QUANTIZABLE, [QuantizerConfig()]),
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key("2 /C_0"),
                    is_merged=True,
                    prop_path=[
                        (
                            InsertionPointGraph.get_post_hook_node_key("1 /B_0"),
                            InsertionPointGraph.get_pre_hook_node_key("2 /C_0"),
                        )
                    ],
                ),
            ],
            expected_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict={
                        InsertionPointGraph.get_pre_hook_node_key("1 /B_0"): (
                            QuantizationTrait.INPUTS_QUANTIZABLE,
                            [QuantizerConfig()],
                        )
                    },
                    starting_quantizer_ip_node=["4 /E_0", "5 /F_0"],
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key("1 /B_0"),
                    is_merged=False,
                    prop_path=None,
                )
            ],
        ),
        SetQuantizersTestStruct(
            start_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict={
                        "4 /E_0": (QuantizationTrait.INPUTS_QUANTIZABLE, [QuantizerConfig()])
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key("1 /B_0"),
                    is_merged=False,
                    prop_path=None,
                ),
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict={
                        "5 /F_0": (QuantizationTrait.INPUTS_QUANTIZABLE, [QuantizerConfig()]),
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
                    target_node_for_quantizer=InsertionPointGraph.get_post_hook_node_key("1 /B_0"),
                    is_merged=True,
                    prop_path=[("1 /B_0", InsertionPointGraph.get_post_hook_node_key("1 /B_0"))],
                ),
            ],
            expected_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict={
                        "1 /B_0": (QuantizationTrait.INPUTS_QUANTIZABLE, [QuantizerConfig()])
                    },
                    starting_quantizer_ip_node=["4 /E_0", "5 /F_0"],
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key("1 /B_0"),
                    is_merged=False,
                    prop_path=None,
                )
            ],
        ),
        SetQuantizersTestStruct(
            start_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict={
                        "4 /E_0": (QuantizationTrait.INPUTS_QUANTIZABLE, [QuantizerConfig()])
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
                    target_node_for_quantizer=InsertionPointGraph.get_post_hook_node_key("1 /B_0"),
                    is_merged=False,
                    prop_path=None,
                ),
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict={
                        "5 /F_0": (QuantizationTrait.INPUTS_QUANTIZABLE, [QuantizerConfig()]),
                    },
                    starting_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
                    target_node_for_quantizer=InsertionPointGraph.get_pre_hook_node_key("2 /C_0"),
                    is_merged=True,
                    prop_path=[
                        (
                            InsertionPointGraph.get_post_hook_node_key("1 /B_0"),
                            InsertionPointGraph.get_pre_hook_node_key("2 /C_0"),
                        )
                    ],
                ),
            ],
            expected_set_quantizers=[
                StateQuantizerTestStruct(
                    init_node_to_trait_and_configs_dict={
                        "1 /B_0": (QuantizationTrait.INPUTS_QUANTIZABLE, [QuantizerConfig()])
                    },
                    starting_quantizer_ip_node=["4 /E_0", "5 /F_0"],
                    target_node_for_quantizer=InsertionPointGraph.get_post_hook_node_key("1 /B_0"),
                    is_merged=False,
                    prop_path=None,
                )
            ],
        ),
    ]

    @staticmethod
    @pytest.fixture(params=MERGE_QUANTIZER_INTO_PATH_TEST_CASES)
    def merge_quantizer_into_path_test_struct(request):
        return request.param

    @pytest.fixture
    def model_graph_qpsg(self):
        mock_graph = self.get_model_graph()
        ip_graph = get_ip_graph_for_test(mock_graph)
        quant_prop_graph = QPSG(ip_graph)
        return quant_prop_graph

    def test_merge_quantizer_into_path(self, model_graph_qpsg, merge_quantizer_into_path_test_struct):
        quant_prop_graph = model_graph_qpsg

        for quantizers_test_struct in merge_quantizer_into_path_test_struct.start_set_quantizers:
            init_node_to_trait_and_configs_dict = quantizers_test_struct.init_node_to_trait_and_configs_dict
            starting_quantizer_ip_node = quantizers_test_struct.starting_quantizer_ip_node
            target_node = quantizers_test_struct.target_node_for_quantizer
            is_merged = quantizers_test_struct.is_merged
            prop_path = quantizers_test_struct.prop_path
            node_key_vs_trait_dict: Dict[str, QuantizationTrait] = {}
            for node_key in quant_prop_graph.nodes:
                node_key_vs_trait_dict[node_key] = QuantizationTrait.QUANTIZATION_AGNOSTIC
            primary_prop_quant = None
            merged_prop_quant = []
            for node_key, trait_and_configs_tuple in init_node_to_trait_and_configs_dict.items():
                trait = trait_and_configs_tuple[0]
                node_key_vs_trait_dict[node_key] = trait
            quant_prop_graph = self.mark_nodes_with_traits(quant_prop_graph, node_key_vs_trait_dict)

            for node_key, trait_and_configs_tuple in init_node_to_trait_and_configs_dict.items():
                trait = trait_and_configs_tuple[0]
                qconfigs = trait_and_configs_tuple[1]
                if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                    ip_node_key = InsertionPointGraph.get_pre_hook_node_key(node_key)
                    prop_quant = quant_prop_graph.add_propagating_quantizer(qconfigs, ip_node_key)
                    if ip_node_key == starting_quantizer_ip_node:
                        primary_prop_quant = prop_quant

            path = get_edge_paths_for_propagation(quant_prop_graph, target_node, starting_quantizer_ip_node)
            primary_prop_quant = quant_prop_graph.propagate_quantizer_via_path(primary_prop_quant, path[0])
            if is_merged:
                merged_prop_quant.append((primary_prop_quant, prop_path))

            quant_prop_graph.run_consistency_check()

        for prop_quant, prop_path in merged_prop_quant:
            quant_prop_graph.merge_quantizer_into_path(prop_quant, prop_path)
            quant_prop_graph.run_consistency_check()

        expected_quantizers_test_struct = merge_quantizer_into_path_test_struct.expected_set_quantizers
        self.check_final_state_qpsg(quant_prop_graph, expected_quantizers_test_struct)

    @staticmethod
    def check_final_state_qpsg(final_quant_prop_graph, expected_quantizers_test_struct):
        for quantizer_param in expected_quantizers_test_struct:
            from_node_key = quantizer_param.target_node_for_quantizer
            expected_prop_path = set()
            target_node = quantizer_param.target_node_for_quantizer
            for start_node in quantizer_param.starting_quantizer_ip_node:
                added_path = get_edge_paths_for_propagation(final_quant_prop_graph, target_node, start_node)
                expected_prop_path.update(added_path[0])

            quantizer = final_quant_prop_graph.nodes[from_node_key][QPSG.PROPAGATING_QUANTIZER_NODE_ATTR]

            assert quantizer is not None

            for from_node_key, to_node_key in expected_prop_path:
                assert (
                    quantizer
                    in final_quant_prop_graph.edges[(from_node_key, to_node_key)][
                        QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR
                    ]
                )
                from_node = final_quant_prop_graph.nodes[from_node_key]
                from_node_type = from_node[QPSG.NODE_TYPE_NODE_ATTR]
                if QPSG.is_insertion_point(from_node_type):
                    assert (
                        quantizer
                        in final_quant_prop_graph.nodes[from_node_key][QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
                    )

            assert quantizer.affected_edges == expected_prop_path


class TestRedundantQuantizerMerge:
    class RedundantQuantizerMergeTestStruct(ABC):
        ref_remaining_pq_positions: Set[str]
        operator_node_key_vs_trait_dict: Dict[str, QuantizationTrait]

        def prepare_qpsg_state(self, qpsg: QPSG) -> QPSG:
            qpsg = TestQuantizerPropagationStateGraph.mark_nodes_with_traits(qpsg, self.operator_node_key_vs_trait_dict)
            return self._setup_and_propagate_quantizers(qpsg)

        @abstractmethod
        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pass

    class NoConnectingPathsState0(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {
            InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
            InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
        }
        operator_node_key_vs_trait_dict = {
            "3 /D_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "5 /F_0": QuantizationTrait.INPUTS_QUANTIZABLE,
        }

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("3 /D_0"))
            qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("5 /F_0"))
            return qpsg

    class NoConnectingPathsState1(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {
            InsertionPointGraph.get_pre_hook_node_key("2 /C_0"),
            InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
        }
        operator_node_key_vs_trait_dict = {
            "2 /C_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "5 /F_0": QuantizationTrait.INPUTS_QUANTIZABLE,
        }

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("2 /C_0"))
            qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("5 /F_0"))
            return qpsg

    class NoConnectingPathsState2(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {
            InsertionPointGraph.get_pre_hook_node_key("1 /B_0"),
            InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
        }
        operator_node_key_vs_trait_dict = {
            "1 /B_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "2 /C_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "3 /D_0": QuantizationTrait.INPUTS_QUANTIZABLE,
        }

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("1 /B_0"))
            qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("4 /E_0"))
            return qpsg

    class NoConnectingPathsState3(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {
            InsertionPointGraph.get_post_hook_node_key("1 /B_0"),
            InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
        }
        operator_node_key_vs_trait_dict = {
            "1 /B_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "2 /C_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "3 /D_0": QuantizationTrait.INPUTS_QUANTIZABLE,
        }

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("2 /C_0")
            )
            pq_2 = qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("3 /D_0")
            )
            qpsg.merge_quantizers_for_branching_node(
                [pq_1, pq_2], [QuantizerConfig()], [None, None], InsertionPointGraph.get_post_hook_node_key("1 /B_0")
            )
            qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("4 /E_0"))
            return qpsg

    class BranchHandlingState0(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {
            InsertionPointGraph.get_pre_hook_node_key("8 /I_0", input_port_id=0),
            InsertionPointGraph.get_pre_hook_node_key("8 /I_0", input_port_id=1),
            InsertionPointGraph.get_pre_hook_node_key("2 /C_0"),
            InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
        }
        operator_node_key_vs_trait_dict = {
            "8 /I_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "2 /C_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "6 /G_0": QuantizationTrait.NON_QUANTIZABLE,
        }

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            # This case will fail if, after going depth-first through the '3 /D_0' branch of the graph,
            # the merge traversal function state is not reset (which is incorrect behavior)
            # when starting to traverse the '2 /C_0' branch.
            qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("8 /I_0", input_port_id=0)
            )
            qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("8 /I_0", input_port_id=1)
            )
            qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("2 /C_0"))
            qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("3 /D_0"))
            return qpsg

    class MergeState0(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {InsertionPointGraph.get_post_hook_node_key("2 /C_0")}
        operator_node_key_vs_trait_dict = {"5 /F_0": QuantizationTrait.INPUTS_QUANTIZABLE}

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("5 /F_0")
            )

            qpsg.propagate_quantizer_via_path(
                pq_1,
                [
                    (
                        InsertionPointGraph.get_post_hook_node_key("2 /C_0"),
                        InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
                    )
                ],
            )
            _ = qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("5 /F_0"))
            return qpsg

    class MergeState1(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {InsertionPointGraph.get_post_hook_node_key("1 /B_0")}
        operator_node_key_vs_trait_dict = {
            "1 /B_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "2 /C_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "3 /D_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "4 /E_0": QuantizationTrait.INPUTS_QUANTIZABLE,
        }

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("2 /C_0")
            )
            pq_2 = qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("3 /D_0")
            )
            qpsg.merge_quantizers_for_branching_node(
                [pq_1, pq_2], [QuantizerConfig()], [None, None], InsertionPointGraph.get_post_hook_node_key("1 /B_0")
            )
            pq_3 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(per_channel=True)],
                InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),  # sic!
            )
            # pq_3 should be considered redundant w.r.t the upstream per-tensor quantizer
            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
                InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_3, path)
            return qpsg

    class NoRedundancyState0(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {
            InsertionPointGraph.get_post_hook_node_key("2 /C_0"),
            InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
        }
        operator_node_key_vs_trait_dict = {"5 /F_0": QuantizationTrait.INPUTS_QUANTIZABLE}

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("5 /F_0")
            )

            qpsg.propagate_quantizer_via_path(
                pq_1,
                [
                    (
                        InsertionPointGraph.get_post_hook_node_key("2 /C_0"),
                        InsertionPointGraph.get_pre_hook_node_key("5 /F_0"),
                    )
                ],
            )
            _ = qpsg.add_propagating_quantizer(
                [QuantizerConfig(num_bits=6)], InsertionPointGraph.get_pre_hook_node_key("5 /F_0")
            )
            return qpsg

    class NoRedundancyState1(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {
            InsertionPointGraph.get_post_hook_node_key("1 /B_0"),
            InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
        }
        operator_node_key_vs_trait_dict = {
            "1 /B_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "2 /C_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "3 /D_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "4 /E_0": QuantizationTrait.INPUTS_QUANTIZABLE,
        }

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(per_channel=True)], InsertionPointGraph.get_pre_hook_node_key("2 /C_0")
            )
            pq_2 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(per_channel=True)], InsertionPointGraph.get_pre_hook_node_key("3 /D_0")
            )
            qpsg.merge_quantizers_for_branching_node(
                [pq_1, pq_2],
                [QuantizerConfig(per_channel=True)],
                [None, None],
                InsertionPointGraph.get_post_hook_node_key("1 /B_0"),
            )
            pq_3 = qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("4 /E_0")
            )
            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
                InsertionPointGraph.get_pre_hook_node_key("4 /E_0"),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_3, path)
            return qpsg

    class NoRedundancyState2(RedundantQuantizerMergeTestStruct):
        ref_remaining_pq_positions = {
            InsertionPointGraph.get_post_hook_node_key("8 /I_0"),
            InsertionPointGraph.get_post_hook_node_key("12 /L_0"),
        }
        operator_node_key_vs_trait_dict = {
            "9 /J_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "10 /K_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "12 /L_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "13 /N_0": QuantizationTrait.INPUTS_QUANTIZABLE,
        }

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("9 /J_0")
            )
            pq_2 = qpsg.add_propagating_quantizer(
                [QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("13 /N_0")
            )
            qpsg.propagate_quantizer_via_path(
                pq_1,
                [
                    (
                        InsertionPointGraph.get_post_hook_node_key("8 /I_0"),
                        InsertionPointGraph.get_pre_hook_node_key("9 /J_0"),
                    )
                ],
            )
            qpsg.propagate_quantizer_via_path(
                pq_2,
                [
                    (
                        InsertionPointGraph.get_post_hook_node_key("12 /L_0"),
                        InsertionPointGraph.get_pre_hook_node_key("13 /N_0"),
                    )
                ],
            )
            return qpsg

    REDUNDANT_QUANTIZER_MERGE_TEST_CASES = [
        # No connecting quantization-agnostic paths between quantizers
        NoConnectingPathsState0(),
        NoConnectingPathsState1(),
        NoConnectingPathsState2(),
        NoConnectingPathsState3(),
        BranchHandlingState0(),
        MergeState0(),
        MergeState1(),
        NoRedundancyState0(),
        NoRedundancyState1(),
        NoRedundancyState2(),
    ]

    @pytest.fixture(params=REDUNDANT_QUANTIZER_MERGE_TEST_CASES)
    def redundant_pq_merge_test_struct(self, request):
        return request.param

    @pytest.fixture
    def model_graph_qpsg(self):
        mock_graph = self.get_model_graph()
        ip_graph = get_ip_graph_for_test(mock_graph)
        quant_prop_graph = QPSG(ip_graph)
        return quant_prop_graph

    @staticmethod
    def get_model_graph() -> NNCFGraph:
        mock_graph = nx.DiGraph()

        #      (0 /A_0)
        #        |
        #      (1 /B_0)
        #    /        \
        # (2 /C_0)     (3 /D_0)
        #   |            |
        # (5 /F_0)     (4 /E_0)
        #   |            |
        # (6 /G_0)       |
        #   |            |
        # (7 /H_0)       |
        #    \          /
        #      (8 /I_0)
        #    /          \
        # (9 /J_0)    (10 /K_0)
        #                |
        # (11 /M_0)      |
        #     \\        /
        #      (12 /L_0)
        #         |
        #      (13 /N_0)
        node_keys = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
        for node_key in node_keys:
            mock_node_attrs = get_mock_nncf_node_attrs(op_name=node_key)
            mock_graph.add_node(node_key, **mock_node_attrs)

        mock_graph.add_edges_from(
            [
                ("A", "B"),
                ("B", "C"),
                ("B", "D"),
                ("D", "E"),
                ("C", "F"),
                ("F", "G"),
                ("G", "H"),
                ("H", "I"),
                ("E", "I"),
                ("I", "J"),
                ("I", "K"),
                ("M", "L"),
                ("K", "L"),
                ("L", "N"),
            ]
        )
        mock_graph.edges[("K", "L")][NNCFGraph.DTYPE_EDGE_ATTR] = Dtype.INTEGER

        mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
        return get_nncf_graph_from_mock_nx_graph(mock_graph)

    def test_merge_redundant_subsequent_quantizers_across_graph(
        self, model_graph_qpsg: QPSG, redundant_pq_merge_test_struct: RedundantQuantizerMergeTestStruct
    ):
        model_graph_qpsg = redundant_pq_merge_test_struct.prepare_qpsg_state(model_graph_qpsg)
        model_graph_qpsg.merge_redundant_subsequent_quantizers_across_graph()
        remaining_pqs = model_graph_qpsg.collect_all_propagating_quantizers()
        ref_remaining_pq_positions = redundant_pq_merge_test_struct.ref_remaining_pq_positions
        for pq in remaining_pqs:
            assert pq.current_location_node_key in ref_remaining_pq_positions

        assert len(remaining_pqs) == len(ref_remaining_pq_positions)


class TestUnifinedScaleTypeAfterMergeQuantizers:
    @staticmethod
    def setup_and_propagate_quantizers(qpsg: QPSG) -> QPSG:
        pq_1 = qpsg.add_propagating_quantizer(
            [QuantizerConfig(per_channel=True)],
            InsertionPointGraph.get_pre_hook_node_key("4 /E_0", input_port_id=0),
            unified_scale_type=UnifiedScaleType.UNIFY_ALWAYS,
        )
        g_id = qpsg.get_unified_scale_group_id_by_propagating_quantizer_id(pq_1.id)
        pq_2 = qpsg.add_propagating_quantizer(
            [QuantizerConfig(per_channel=True)],
            InsertionPointGraph.get_pre_hook_node_key("4 /E_0", input_port_id=1),
            unified_scale_type=UnifiedScaleType.UNIFY_ALWAYS,
            unified_scale_group_id_override=g_id,
        )

        paths_1 = get_edge_paths_for_propagation(
            qpsg,
            InsertionPointGraph.get_pre_hook_node_key("2 /C_0"),
            InsertionPointGraph.get_pre_hook_node_key("4 /E_0", input_port_id=0),
        )

        paths_2 = get_edge_paths_for_propagation(
            qpsg,
            InsertionPointGraph.get_pre_hook_node_key("3 /D_0"),
            InsertionPointGraph.get_pre_hook_node_key("4 /E_0", input_port_id=1),
        )

        qpsg.propagate_quantizer_via_path(pq_1, paths_1[0])
        qpsg.propagate_quantizer_via_path(pq_2, paths_2[0])

        qpsg.merge_quantizers_for_branching_node(
            [pq_1, pq_2], [QuantizerConfig(per_channel=True)], [None, None], "1 /B_0"
        )

    @staticmethod
    def get_model_graph_with_split_node() -> QPSG:
        mock_graph = nx.DiGraph()

        #         (0 /A_0)
        #            |
        #         (1 /B_0)
        #        /        \
        #    (2 /C_0)   (3 /D_0)
        #        \        /
        #         (4 /E_0)

        node_keys = ["A", "B", "C", "D", "E"]
        for node_key in node_keys:
            mock_node_attrs = get_mock_nncf_node_attrs(op_name=node_key)
            if node_key == "B":
                # Split have no POST_HOOK
                mock_node_attrs[NNCFNode.NODE_TYPE_ATTR] = "split"
            mock_graph.add_node(node_key, **mock_node_attrs)

        mock_graph.add_edges_from([("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E")])
        mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
        mock_graph = get_nncf_graph_from_mock_nx_graph(mock_graph)

        ip_graph = get_ip_graph_for_test(mock_graph)
        qpsg = QPSG(ip_graph)

        operator_node_key_vs_trait_dict = {
            "0 /A_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "1 /B_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "2 /C_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "3 /D_0": QuantizationTrait.CONCAT,
            "4 /E_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
        }
        model_graph_qpsg = TestQuantizerPropagationStateGraph.mark_nodes_with_traits(
            qpsg, operator_node_key_vs_trait_dict
        )
        return model_graph_qpsg

    def test_unified_scale_type_for_split_node(self):
        model_graph_qpsg = self.get_model_graph_with_split_node()
        self.setup_and_propagate_quantizers(model_graph_qpsg)

        quantizers = model_graph_qpsg.collect_all_propagating_quantizers()
        assert len(quantizers) == 1

        pq = quantizers.pop()
        assert pq.unified_scale_type == UnifiedScaleType.UNIFY_ALWAYS

        setup = model_graph_qpsg.create_quantizer_setup({})
        assert len(setup.unified_scale_groups) == 1

        gid = setup.get_unified_scale_group_id(pq.id)
        assert gid == 0


def create_graph_for_output_quant_as_weights() -> NNCFGraph:
    mock_graph = nx.DiGraph()

    #      (0 A/A)       (1 B/B)
    #        |             |
    #      (2 C/C)         |
    #    /         \      /
    # (3 D/D)       (4 E/E)
    #   |             |
    # (6 G/G)       (5 F/F)
    #   |             |
    # (7 H/H)         |
    #   |             |
    # (8 I/I)         |
    #    \           /
    #     \         /
    #       (9 J/J)
    node_keys = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    for node_key in node_keys:
        mock_node_attrs = get_mock_nncf_node_attrs(op_name=node_key, scope_str=node_key)
        mock_graph.add_node(node_key, **mock_node_attrs)

    mock_graph.add_edges_from(
        [
            ("A", "C"),
            ("C", "D"),
            ("C", "E"),
            ("D", "G"),
            ("G", "H"),
            ("H", "I"),
            ("I", "J"),
            ("B", "E"),
            ("E", "F"),
            ("F", "J"),
        ]
    )
    mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
    return get_nncf_graph_from_mock_nx_graph(mock_graph)


MODEL_GRAPH: NNCFGraph = create_graph_for_output_quant_as_weights()


class OutputQuantAsWeightsSetupTestStruct(ABC):
    operator_node_key_vs_trait_dict: Dict[str, QuantizationTrait]
    quantizable_module_node_names_vs_qconfigs: Dict[NNCFNodeName, List[QuantizerConfig]]

    def prepare_qpsg_state(self, qpsg: QPSG) -> QPSG:
        qpsg = TestQuantizerPropagationStateGraph.mark_nodes_with_traits(qpsg, self.operator_node_key_vs_trait_dict)
        return self._setup_and_propagate_quantizers(qpsg)

    @abstractmethod
    def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
        pass

    @abstractmethod
    def ref_quantizer_setup(self) -> MultiConfigQuantizerSetup:
        pass


class LinearPropagation(OutputQuantAsWeightsSetupTestStruct):
    operator_node_key_vs_trait_dict = {
        "5 F/F_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
        "8 G/G_0": QuantizationTrait.INPUTS_QUANTIZABLE,
        "2 C/C_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
    }
    quantizable_module_node_names_vs_qconfigs = {"C/C_0": [QuantizerConfig()]}

    def ref_quantizer_setup(self) -> MultiConfigQuantizerSetup:
        setup = MultiConfigQuantizerSetup()
        setup.quantization_points[0] = MultiConfigQuantizationPoint(
            WeightQuantizationInsertionPoint(target_node_name="C/C_0"),
            possible_qconfigs=[QuantizerConfig()],
            directly_quantized_operator_node_names=["C/C_0", "G/G_0"],
        )
        setup.shared_input_operation_set_groups[0] = {0}
        return setup

    def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
        pq_1 = qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("6 G/G_0"))
        paths = get_edge_paths_for_propagation(
            qpsg,
            InsertionPointGraph.get_post_hook_node_key("2 C/C_0"),
            InsertionPointGraph.get_pre_hook_node_key("6 G/G_0"),
        )
        path = paths[0]
        qpsg.propagate_quantizer_via_path(pq_1, path)
        qpsg.mark_act_quantizer_as_dependent_on_weights(pq_1, "2 C/C_0")
        return qpsg


class TestOutputQuantAsWeightsSetup:
    class LinearPropagationWithConfigSubspaceSelection(OutputQuantAsWeightsSetupTestStruct):
        operator_node_key_vs_trait_dict = {
            "4 D/D_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "8 G/G_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "2 C/C_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
        }
        quantizable_module_node_names_vs_qconfigs = {
            "C/C_0": [QuantizerConfig(num_bits=6), QuantizerConfig(num_bits=8)]
        }

        def ref_quantizer_setup(self) -> MultiConfigQuantizerSetup:
            setup = MultiConfigQuantizerSetup()
            setup.quantization_points[0] = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(target_node_name="C/C_0"),
                possible_qconfigs=[QuantizerConfig(num_bits=8)],
                directly_quantized_operator_node_names=["C/C_0", "G/G_0"],
            )
            setup.shared_input_operation_set_groups[0] = {0}
            return setup

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=4)],
                InsertionPointGraph.get_pre_hook_node_key("6 G/G_0"),
            )
            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_post_hook_node_key("2 C/C_0"),
                InsertionPointGraph.get_pre_hook_node_key("6 G/G_0"),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_1, path)
            qpsg.mark_act_quantizer_as_dependent_on_weights(pq_1, "2 C/C_0")
            return qpsg

    class LinearPropagationWithFailedMerge(OutputQuantAsWeightsSetupTestStruct):
        operator_node_key_vs_trait_dict = {
            "4 D/D_0": QuantizationTrait.QUANTIZATION_AGNOSTIC,
            "8 G/G_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "2 C/C_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
        }
        quantizable_module_node_names_vs_qconfigs = {
            "C/C_0": [QuantizerConfig(num_bits=6), QuantizerConfig(num_bits=8)]
        }

        def ref_quantizer_setup(self) -> MultiConfigQuantizerSetup:
            setup = MultiConfigQuantizerSetup()
            setup.quantization_points[0] = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(target_node_name="C/C_0"),
                possible_qconfigs=[QuantizerConfig(num_bits=6), QuantizerConfig(num_bits=8)],
                directly_quantized_operator_node_names=["C/C_0"],
            )
            setup.quantization_points[1] = MultiConfigQuantizationPoint(
                ActivationQuantizationInsertionPoint(target_node_name="C/C_0"),
                possible_qconfigs=[QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=5)],
                directly_quantized_operator_node_names=["G/G_0"],
            )
            setup.shared_input_operation_set_groups = {0: {0, 1}}
            return setup

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=5)],
                InsertionPointGraph.get_pre_hook_node_key("6 G/G_0"),
            )
            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_post_hook_node_key("2 C/C_0"),
                InsertionPointGraph.get_pre_hook_node_key("6 G/G_0"),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_1, path)
            qpsg.mark_act_quantizer_as_dependent_on_weights(pq_1, "2 C/C_0")
            return qpsg

    class LinearPropagationWithUnifiedScales(OutputQuantAsWeightsSetupTestStruct):
        operator_node_key_vs_trait_dict = {
            "9 J/J_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "8 I/I_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
        }
        quantizable_module_node_names_vs_qconfigs = {"I/I_0": [QuantizerConfig(num_bits=4)]}

        def ref_quantizer_setup(self) -> MultiConfigQuantizerSetup:
            setup = MultiConfigQuantizerSetup()
            setup.quantization_points[0] = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(target_node_name="I/I_0"),
                possible_qconfigs=[QuantizerConfig(num_bits=4)],
                directly_quantized_operator_node_names=["I/I_0", "J/J_0"],
            )
            setup.quantization_points[1] = MultiConfigQuantizationPoint(
                ActivationQuantizationInsertionPoint(target_node_name="J/J_0", input_port_id=1),
                possible_qconfigs=[QuantizerConfig(num_bits=4)],
                directly_quantized_operator_node_names=["J/J_0"],
            )
            setup.unified_scale_groups = {0: {0, 1}}
            setup.shared_input_operation_set_groups = {0: {0, 1}}
            return setup

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(num_bits=4)],
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=0),
            )
            pq_2 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(num_bits=4)],
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=1),
            )
            qpsg.unify_pq_scales(pq_1, pq_2)
            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_post_hook_node_key("8 I/I_0"),
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=0),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_1, path)
            qpsg.mark_act_quantizer_as_dependent_on_weights(pq_1, "8 I/I_0")
            return qpsg

    class BranchingPropagationWithUnifiedScales(OutputQuantAsWeightsSetupTestStruct):
        operator_node_key_vs_trait_dict = {
            "9 J/J_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "8 I/I_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
            "5 F/F_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
        }
        quantizable_module_node_names_vs_qconfigs = {
            "I/I_0": [QuantizerConfig(mode=QuantizationMode.ASYMMETRIC)],
            "F/F_0": [QuantizerConfig(mode=QuantizationMode.ASYMMETRIC)],
        }

        def ref_quantizer_setup(self) -> MultiConfigQuantizerSetup:
            setup = MultiConfigQuantizerSetup()
            setup.quantization_points[0] = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(target_node_name="I/I_0"),
                possible_qconfigs=[QuantizerConfig(mode=QuantizationMode.ASYMMETRIC)],
                directly_quantized_operator_node_names=["I/I_0", "J/J_0"],
            )
            setup.quantization_points[1] = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(target_node_name="F/F_0"),
                possible_qconfigs=[QuantizerConfig(mode=QuantizationMode.ASYMMETRIC)],
                directly_quantized_operator_node_names=["F/F_0", "J/J_0"],
            )
            setup.unified_scale_groups = {0: {0, 1}}
            setup.shared_input_operation_set_groups = {0: {0, 1}}
            return setup

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(mode=QuantizationMode.ASYMMETRIC)],
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=0),
            )
            pq_2 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(mode=QuantizationMode.ASYMMETRIC)],
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=1),
            )
            qpsg.unify_pq_scales(pq_1, pq_2)

            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_post_hook_node_key("8 I/I_0"),
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=0),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_1, path)
            qpsg.mark_act_quantizer_as_dependent_on_weights(pq_1, "8 I/I_0")

            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_post_hook_node_key("5 F/F_0"),
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=1),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_2, path)
            qpsg.mark_act_quantizer_as_dependent_on_weights(pq_2, "5 F/F_0")
            return qpsg

    class BranchingPropagationWithPerChannelFiltering(OutputQuantAsWeightsSetupTestStruct):
        operator_node_key_vs_trait_dict = {
            "9 J/J_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "8 I/I_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
            "5 F/F_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
        }
        quantizable_module_node_names_vs_qconfigs = {
            "I/I_0": [QuantizerConfig(mode=QuantizationMode.ASYMMETRIC, per_channel=True), QuantizerConfig(num_bits=4)],
            "F/F_0": [QuantizerConfig(mode=QuantizationMode.ASYMMETRIC, per_channel=True), QuantizerConfig(num_bits=4)],
        }

        def ref_quantizer_setup(self) -> MultiConfigQuantizerSetup:
            setup = MultiConfigQuantizerSetup()
            setup.quantization_points[0] = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(target_node_name="I/I_0"),
                possible_qconfigs=[QuantizerConfig(num_bits=4)],
                directly_quantized_operator_node_names=["I/I_0", "J/J_0"],
            )
            setup.quantization_points[1] = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(target_node_name="F/F_0"),
                possible_qconfigs=[QuantizerConfig(num_bits=4)],
                directly_quantized_operator_node_names=["F/F_0", "J/J_0"],
            )
            setup.unified_scale_groups = {0: {0, 1}}
            setup.shared_input_operation_set_groups = {0: {0, 1}}
            return setup

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(mode=QuantizationMode.ASYMMETRIC, per_channel=True), QuantizerConfig(num_bits=4)],
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=0),
            )
            pq_2 = qpsg.add_propagating_quantizer(
                [QuantizerConfig(num_bits=4, per_channel=True), QuantizerConfig(num_bits=4)],
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=1),
            )
            qpsg.unify_pq_scales(pq_1, pq_2)

            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_post_hook_node_key("8 I/I_0"),
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=0),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_1, path)
            qpsg.mark_act_quantizer_as_dependent_on_weights(pq_1, "8 I/I_0")

            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_post_hook_node_key("5 F/F_0"),
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=1),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_2, path)
            qpsg.mark_act_quantizer_as_dependent_on_weights(pq_2, "5 F/F_0")
            return qpsg

    class BranchingPropagationWithPerChannelFilteringAndUnifiedScales(OutputQuantAsWeightsSetupTestStruct):
        operator_node_key_vs_trait_dict = {
            "9 J/J_0": QuantizationTrait.INPUTS_QUANTIZABLE,
            "8 I/I_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
            "5 F/F_0": QuantizationTrait.OUTPUT_QUANTIZATION_AS_WEIGHTS,
        }
        quantizable_module_node_names_vs_qconfigs = {
            "I/I_0": [
                QuantizerConfig(mode=QuantizationMode.ASYMMETRIC),
                QuantizerConfig(num_bits=6, signedness_to_force=True),
                QuantizerConfig(num_bits=4, signedness_to_force=True),
            ],
            "F/F_0": [
                QuantizerConfig(mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                QuantizerConfig(num_bits=4),
                QuantizerConfig(),
            ],
        }

        def ref_quantizer_setup(self) -> MultiConfigQuantizerSetup:
            setup = MultiConfigQuantizerSetup()
            setup.quantization_points[0] = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(target_node_name="I/I_0"),
                possible_qconfigs=[QuantizerConfig(mode=QuantizationMode.ASYMMETRIC)],
                directly_quantized_operator_node_names=["I/I_0", "J/J_0"],
            )
            setup.quantization_points[1] = MultiConfigQuantizationPoint(
                WeightQuantizationInsertionPoint(target_node_name="F/F_0"),
                possible_qconfigs=[
                    QuantizerConfig(mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                    QuantizerConfig(num_bits=4),
                    QuantizerConfig(),
                ],
                directly_quantized_operator_node_names=["F/F_0"],
            )
            setup.quantization_points[2] = MultiConfigQuantizationPoint(
                ActivationQuantizationInsertionPoint(target_node_name="F/F_0"),
                possible_qconfigs=[QuantizerConfig(mode=QuantizationMode.ASYMMETRIC)],
                directly_quantized_operator_node_names=["J/J_0"],
            )
            setup.unified_scale_groups = {0: {0, 2}}
            setup.shared_input_operation_set_groups = {0: {0, 1, 2}}
            return setup

        def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
            pq_1 = qpsg.add_propagating_quantizer(
                [
                    QuantizerConfig(mode=QuantizationMode.ASYMMETRIC),
                    QuantizerConfig(num_bits=6, signedness_to_force=False),
                ],
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=0),
            )
            pq_2 = qpsg.add_propagating_quantizer(
                [
                    QuantizerConfig(num_bits=4, per_channel=True),
                    QuantizerConfig(num_bits=4),
                    QuantizerConfig(mode=QuantizationMode.ASYMMETRIC),
                ],
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=1),
            )
            qpsg.unify_pq_scales(pq_1, pq_2)

            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_post_hook_node_key("8 I/I_0"),
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=0),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_1, path)
            qpsg.mark_act_quantizer_as_dependent_on_weights(pq_1, "8 I/I_0")

            paths = get_edge_paths_for_propagation(
                qpsg,
                InsertionPointGraph.get_post_hook_node_key("5 F/F_0"),
                InsertionPointGraph.get_pre_hook_node_key("9 J/J_0", input_port_id=1),
            )
            path = paths[0]
            qpsg.propagate_quantizer_via_path(pq_2, path)
            qpsg.mark_act_quantizer_as_dependent_on_weights(pq_2, "5 F/F_0")
            return qpsg

    OUTPUT_QUANT_AS_WEIGHTS_TEST_CASES = [
        LinearPropagation(),
        LinearPropagationWithConfigSubspaceSelection(),
        LinearPropagationWithFailedMerge(),
        LinearPropagationWithUnifiedScales(),
        BranchingPropagationWithUnifiedScales(),
        BranchingPropagationWithPerChannelFiltering(),
        BranchingPropagationWithPerChannelFilteringAndUnifiedScales(),
    ]

    @pytest.fixture(params=OUTPUT_QUANT_AS_WEIGHTS_TEST_CASES)
    def output_quant_as_weights_test_struct(self, request):
        return request.param

    @pytest.fixture
    def model_graph_qpsg(self):
        ip_graph = get_ip_graph_for_test(MODEL_GRAPH)
        quant_prop_graph = QPSG(ip_graph)
        return quant_prop_graph

    def test_create_quantizer_setup_with_output_quant_as_weights_ops(
        self, model_graph_qpsg: QPSG, output_quant_as_weights_test_struct: OutputQuantAsWeightsSetupTestStruct
    ):
        prepped_qpsg = output_quant_as_weights_test_struct.prepare_qpsg_state(model_graph_qpsg)
        test_quantizer_setup = prepped_qpsg.create_quantizer_setup(
            output_quant_as_weights_test_struct.quantizable_module_node_names_vs_qconfigs
        )
        ref_quantizer_setup = output_quant_as_weights_test_struct.ref_quantizer_setup()
        assert test_quantizer_setup.equivalent_to(ref_quantizer_setup)


@pytest.mark.parametrize(
    "weight_configs, activation_configs, reference_configs",
    [
        (
            # Weights #1
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=True, per_channel=False
                ),
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=True, per_channel=True
                ),
            ],
            # Activations #1
            [
                QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
            ],
            # Reference #1
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=True, per_channel=False
                ),
            ],
        ),
        (
            # Weights #2
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=True, per_channel=False
                ),
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=True, per_channel=True
                ),
            ],
            # Activations #2
            [
                QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False),
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=True, per_channel=False
                ),
            ],
            # Reference #2
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=True, per_channel=False
                ),
            ],
        ),
        (
            # Weights #3
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=True, per_channel=False
                ),
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=True, per_channel=True
                ),
            ],
            # Activations #3
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=True, per_channel=False
                ),
            ],
            # Reference #3
            [],
        ),
        (
            # Weights #4
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=True, per_channel=False
                ),
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=True, per_channel=True
                ),
            ],
            # Activations #4
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=False, per_channel=False
                ),
            ],
            # Reference #4
            [],
        ),
        (
            # Weights #5
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=False, per_channel=False
                ),
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=True, per_channel=True
                ),
            ],
            # Activations #5
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=None, per_channel=False
                ),
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=None, per_channel=False
                ),
            ],
            # Reference #5
            [
                QuantizerConfig(
                    num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=False, per_channel=False
                ),
            ],
        ),
        (
            # Weights #6
            [
                QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
            ],
            # Activations #6
            [
                QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
            ],
            # Reference #6
            [
                QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
            ],
        ),
    ],
)
def test_get_weight_and_activation_qconfig_list_intersection(weight_configs, activation_configs, reference_configs):
    resulted_configs = QPSG._get_weight_and_activation_qconfig_list_intersection(weight_configs, activation_configs)
    assert resulted_configs == reference_configs


class LinearPropagationForRemovalTest(LinearPropagation):
    def _setup_and_propagate_quantizers(self, qpsg: QPSG) -> QPSG:
        pq_1 = qpsg.add_propagating_quantizer([QuantizerConfig()], InsertionPointGraph.get_pre_hook_node_key("6 G/G_0"))
        paths = get_edge_paths_for_propagation(
            qpsg,
            InsertionPointGraph.get_post_hook_node_key("2 C/C_0"),
            InsertionPointGraph.get_pre_hook_node_key("6 G/G_0"),
        )
        path = paths[0]
        qpsg.propagate_quantizer_via_path(pq_1, path)
        qpsg.mark_act_quantizer_as_dependent_on_weights(pq_1, "2 C/C_0")
        return qpsg, pq_1


def test_remove_pq_from_pqs_after_weight_dependent_output_quantized_nodes():
    ip_graph = get_ip_graph_for_test(MODEL_GRAPH)
    quant_prop_graph = QPSG(ip_graph)
    linear_propagation = LinearPropagationForRemovalTest()
    quant_prop_graph, pq_1 = linear_propagation.prepare_qpsg_state(quant_prop_graph)
    qpsg, pq_1 = linear_propagation._setup_and_propagate_quantizers(quant_prop_graph)
    qpsg.remove_propagating_quantizer(pq_1)
    assert pq_1 not in qpsg._pqs_after_weight_dependent_output_quantized_nodes
