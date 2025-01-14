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


from collections import Counter
from collections import namedtuple
from dataclasses import dataclass
from itertools import permutations
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import pytest

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.quantization.quantizer_propagation.graph import QuantizerPropagationStateGraph as QPSG
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationRule
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.quantizer_propagation.solver import TransitionStatus
from nncf.common.quantization.quantizer_propagation.structs import IgnoreReason
from nncf.common.quantization.quantizer_propagation.structs import PropagatingQuantizer
from nncf.common.quantization.quantizer_propagation.structs import PropagationPath
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.quantization.quantizer_setup import ActivationQuantizationInsertionPoint
from nncf.common.quantization.quantizer_setup import MultiConfigQuantizationPoint
from nncf.common.quantization.quantizer_setup import QuantizationPointId
from nncf.common.quantization.quantizer_setup import WeightQuantizationInsertionPoint
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from tests.common.quantization.metatypes import DEFAULT_TEST_QUANT_TRAIT_MAP
from tests.common.quantization.metatypes import BatchNormTestMetatype
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import DropoutTestMetatype
from tests.common.quantization.metatypes import GeluTestMetatype
from tests.common.quantization.metatypes import MatMulTestMetatype
from tests.common.quantization.metatypes import MaxPool2dTestMetatype
from tests.common.quantization.metatypes import MinTestMetatype
from tests.common.quantization.metatypes import ScaledDotProductAttentionMetatype
from tests.common.quantization.metatypes import SoftmaxTestMetatype
from tests.common.quantization.mock_graphs import get_ip_graph_for_test
from tests.common.quantization.mock_graphs import get_mock_nncf_node_attrs
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph
from tests.common.quantization.mock_graphs import get_sequentially_connected_model_graph
from tests.common.quantization.mock_graphs import mark_input_ports_lexicographically_based_on_input_node_key
from tests.common.quantization.test_quantizer_propagation_graph import get_edge_paths_for_propagation


class TwoFcAfterDropout:
    DROPOUT_OP_TYPE_STR = "dropout"
    DROPOUT_NODE_NAME = f"TwoFcAfterDropoutModel/{DROPOUT_OP_TYPE_STR}_0"

    FC_OP_TYPE_STR = "linear"
    FC_1_NODE_NAME = f"TwoFcAfterDropoutModel/NNCFLinear[branch1]/{FC_OP_TYPE_STR}_0"

    FC_2_SCOPE_STR = "TwoFcAfterDropoutModel/NNCFLinear[branch2]"
    FC_2_NODE_NAME = f"{FC_2_SCOPE_STR}/linear_0"

    @staticmethod
    def get_graph():
        graph = nx.DiGraph()
        dropout_node_attrs = {
            NNCFNode.NODE_NAME_ATTR: TwoFcAfterDropout.DROPOUT_NODE_NAME,
            NNCFNode.NODE_TYPE_ATTR: TwoFcAfterDropout.DROPOUT_OP_TYPE_STR,
        }

        fc_1_node_attrs = {
            NNCFNode.NODE_NAME_ATTR: TwoFcAfterDropout.FC_1_NODE_NAME,
            NNCFNode.NODE_TYPE_ATTR: TwoFcAfterDropout.FC_OP_TYPE_STR,
        }

        fc_2_node_attrs = {
            NNCFNode.NODE_NAME_ATTR: TwoFcAfterDropout.FC_2_NODE_NAME,
            NNCFNode.NODE_TYPE_ATTR: TwoFcAfterDropout.FC_OP_TYPE_STR,
        }

        graph.add_node("dropout", **dropout_node_attrs)
        graph.add_node("fc_1", **fc_1_node_attrs)
        graph.add_node("fc_2", **fc_2_node_attrs)
        graph.add_edge("dropout", "fc_1")
        graph.add_edge("dropout", "fc_2")

        mark_input_ports_lexicographically_based_on_input_node_key(graph)
        return graph


def get_branching_model_graph() -> NNCFGraph:
    mock_graph = nx.DiGraph()

    #              (0 /O)  <-- treating this as an auxiliary "input" node
    #                 |
    #              (1 /A)
    #                 |
    #            /-(2 /B)---------\
    #           /     |           |
    #       (3 /C)  (4 /D)      (5 /E)
    #          |                /    \
    #       (6 /F)           (7 /G) (8 /H)
    #      /    |               \    /
    # (11 /K)  (12 /L)          (9 /I)
    #    |       |                 |
    # (13 /M)  (14 /N)          (10 /J)
    #    |       |
    # (15 /P)  (16 /Q)

    node_keys = ["O", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q"]
    for node_key in node_keys:
        mock_node_attrs = get_mock_nncf_node_attrs(op_name=node_key)
        mock_graph.add_node(node_key, **mock_node_attrs)

    mock_graph.add_edges_from(
        [
            ("O", "A"),
            ("A", "B"),
            ("B", "C"),
            ("B", "D"),
            ("B", "E"),
            ("C", "F"),
            ("E", "G"),
            ("E", "H"),
            ("G", "I"),
            ("H", "I"),
            ("I", "J"),
            ("F", "K"),
            ("F", "L"),
            ("K", "M"),
            ("L", "N"),
            ("M", "P"),
            ("N", "Q"),
        ]
    )

    mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
    return get_nncf_graph_from_mock_nx_graph(mock_graph)


def get_scaled_dot_product_graph():
    mock_graph = nx.DiGraph()

    node_keys = ["input", "branch_node", "reshape", "reshape_1", "reshape_2", "scaled_dot_product_attention"]
    for node_key in node_keys:
        mock_node_attrs = get_mock_nncf_node_attrs(op_name=node_key)
        mock_graph.add_node(node_key, **mock_node_attrs)

    mock_graph.add_edges_from(
        [
            ("input", "branch_node"),
            ("branch_node", "reshape"),
            ("branch_node", "reshape_1"),
            ("branch_node", "reshape_2"),
            ("reshape", "scaled_dot_product_attention"),
            ("reshape_1", "scaled_dot_product_attention"),
            ("reshape_2", "scaled_dot_product_attention"),
        ]
    )

    mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
    return get_nncf_graph_from_mock_nx_graph(mock_graph)


class MultiQPSerializedDataForTest:
    def __init__(
        self,
        target_type: TargetType,
        node_name: NNCFNodeName,
        qconfigs: List[QuantizerConfig],
        input_port_id: int = None,
        directly_quantized_op_node_names: List[NNCFNodeName] = None,
    ):
        self.target_type = target_type
        self.node_name = node_name
        self.qconfigs = qconfigs
        self.input_port_id = input_port_id
        if directly_quantized_op_node_names is None:
            self.directly_quantized_op_node_names = []
        else:
            self.directly_quantized_op_node_names = directly_quantized_op_node_names


class RunOnIpGraphTestStruct:
    def __init__(
        self,
        base_nx_graph: nx.DiGraph,
        retval_qp_data: Dict[QuantizationPointId, MultiQPSerializedDataForTest],
        retval_unified_scale_qp_groups: List[Set[QuantizationPointId]],
        retval_shared_input_operation_set_groups: List[Set[QuantizationPointId]],
        expected_count_finished_quant: int,
        expected_count_active_quant: int,
        ignored_scopes: Optional[List[str]],
    ):
        self.base_graph = get_nncf_graph_from_mock_nx_graph(base_nx_graph)
        self.retval_unified_scale_qp_groups = retval_unified_scale_qp_groups
        self.retval_shared_input_operation_set_groups = retval_shared_input_operation_set_groups
        self.expected_count_finished_quant = expected_count_finished_quant
        self.expected_count_active_quant = expected_count_active_quant
        self.ignored_scopes = ignored_scopes
        self.retval_qps: Dict[QuantizationPointId, MultiConfigQuantizationPoint] = {}
        for id_, qp_data in retval_qp_data.items():
            if qp_data.target_type is TargetType.OPERATION_WITH_WEIGHTS:
                qip = WeightQuantizationInsertionPoint(qp_data.node_name)
            else:
                assert qp_data.target_type in [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK]
                qip = ActivationQuantizationInsertionPoint(qp_data.node_name, qp_data.input_port_id)

            self.retval_qps[id_] = MultiConfigQuantizationPoint(
                qip,
                possible_qconfigs=qp_data.qconfigs,
                directly_quantized_operator_node_names=qp_data.directly_quantized_op_node_names,
            )


class TestQuantizerPropagationSolver:
    def test_setup_initial_quantizers_in_quant_prop_graph(self):
        ops_to_quantize = [
            BatchNormTestMetatype.name,
            Conv2dTestMetatype.name,
            MatMulTestMetatype.name,
            GeluTestMetatype.name,
        ]
        ops_not_to_quantize = [
            MaxPool2dTestMetatype.name,
            DropoutTestMetatype.name,
            MinTestMetatype.name,
            SoftmaxTestMetatype.name,
        ]
        node_keys = [MODEL_INPUT_OP_NAME] + ops_to_quantize + ops_not_to_quantize
        mock_graph = get_sequentially_connected_model_graph(node_keys)
        nncf_graph = get_nncf_graph_from_mock_nx_graph(mock_graph)
        ip_graph = get_ip_graph_for_test(nncf_graph)

        qp_graph = QPSG(ip_graph)
        quant_prop_solver = QuantizerPropagationSolver(
            run_consistency_checks=True, default_trait_to_metatype_map=DEFAULT_TEST_QUANT_TRAIT_MAP
        )
        qp_graph = quant_prop_solver.set_allowed_quantization_types_for_operator_nodes(qp_graph)
        qp_graph = quant_prop_solver.setup_initial_quantizers(qp_graph)
        qp_graph.run_consistency_check()

        for node_key in ops_to_quantize:
            actual_key = nncf_graph.get_node_key_by_id(nncf_graph.get_node_by_name("/" + node_key + "_0").node_id)
            pred_ip_key = next(qp_graph.predecessors(actual_key))
            node = qp_graph.nodes[actual_key]
            pred_ip_node = qp_graph.nodes[pred_ip_key]
            prop_quant = pred_ip_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR]
            assert prop_quant is not None
            assert node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]

            edge = qp_graph.edges[pred_ip_key, actual_key]
            assert edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]

        for node_key in ops_not_to_quantize:
            nncf_node = nncf_graph.get_node_by_name("/" + node_key + "_0")
            actual_key = nncf_graph.get_node_key_by_id(nncf_node.node_id)
            pred_ip_key = next(qp_graph.predecessors(actual_key))
            node = qp_graph.nodes[actual_key]
            pred_ip_node = qp_graph.nodes[pred_ip_key]
            assert pred_ip_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None

            assert not node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            edge = qp_graph.edges[pred_ip_key, actual_key]
            assert not edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

    def test_setup_initial_quantizers_sdpa(self):
        nncf_graph = get_scaled_dot_product_graph()
        ip_graph = get_ip_graph_for_test(nncf_graph)

        qp_graph = QPSG(ip_graph)

        sdpa_node_key = "5 /scaled_dot_product_attention_0"
        quant_prop_solver = QuantizerPropagationSolver(
            run_consistency_checks=True,
            default_trait_to_metatype_map=DEFAULT_TEST_QUANT_TRAIT_MAP,
        )

        qp_graph = quant_prop_solver.set_allowed_quantization_types_for_operator_nodes(qp_graph)
        qp_graph = quant_prop_solver.setup_initial_quantizers(qp_graph)
        qp_graph.run_consistency_check()

        for port_id, pred_ip_key in enumerate(qp_graph.predecessors(sdpa_node_key)):
            node = qp_graph.nodes[sdpa_node_key]
            pred_ip_node = qp_graph.nodes[pred_ip_key]
            prop_quant = pred_ip_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR]
            if port_id in ScaledDotProductAttentionMetatype.target_input_ports:
                assert prop_quant is not None
                assert node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR][port_id] == prop_quant

                edge = qp_graph.edges[pred_ip_key, sdpa_node_key]
                assert edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]
            else:
                assert prop_quant is None

                edge = qp_graph.edges[pred_ip_key, sdpa_node_key]
                assert edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == []

    MergeQConfigSolution = namedtuple(
        "MergeQConfigSolution", ("merge_qconfig_list", "branch_qconfig_lists_after_merge")
    )
    MergeQConfigTestStruct = namedtuple(
        "MergeQConfigTestStruct", ("branch_qconfig_lists_before_merge", "strategy_vs_solution_dict")
    )
    QCONFIG_PRIMARY_SECONDARY_BEFORE_AND_AFTER_MERGING = [
        # Compatible configs on all branches
        # 0
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=8), ],
                [QuantizerConfig(num_bits=8), ],
                [QuantizerConfig(num_bits=8), ]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8), ],
                        [QuantizerConfig(num_bits=8), ],
                        [QuantizerConfig(num_bits=8), ]
                    ], ),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None])
            }),

        # 1
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), ]
            ],

            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), ]
                    ]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None,
                                                      None]
                ),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None,
                                                      None]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None,
                                                      None])
            }),

        # 2
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(num_bits=8)],
                [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(num_bits=8)],
                [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(num_bits=8)],
            ],

            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(num_bits=8)],
                        [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(num_bits=8)],
                        [QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(num_bits=8)],
                    ]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC),
                                        QuantizerConfig(num_bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None]
                ),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC),
                                        QuantizerConfig(num_bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None]),

                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC),
                                        QuantizerConfig(num_bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None])
            }),

        # Requantization necessary for merge
        # 3
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=6), ],
                [QuantizerConfig(num_bits=4), ],
                [QuantizerConfig(num_bits=5), ]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[[QuantizerConfig(num_bits=6), ],
                                                      [QuantizerConfig(num_bits=4), ],
                                                      [QuantizerConfig(num_bits=5), ]]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[[QuantizerConfig(num_bits=6), ],
                                                      [QuantizerConfig(num_bits=4), ],
                                                      [QuantizerConfig(num_bits=5), ]]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=6)],
                    branch_qconfig_lists_after_merge=[None,
                                                      [QuantizerConfig(num_bits=4), ],
                                                      [QuantizerConfig(num_bits=5), ]]),

                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[[QuantizerConfig(num_bits=6), ],
                                                      [QuantizerConfig(num_bits=4), ],
                                                      [QuantizerConfig(num_bits=5), ]])
            }),

        # 4
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                [QuantizerConfig(num_bits=4), ],
                [QuantizerConfig(num_bits=5), ]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(num_bits=4), ],
                        [QuantizerConfig(num_bits=5), ]]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(num_bits=4), ],
                        [QuantizerConfig(num_bits=5), ]]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                    branch_qconfig_lists_after_merge=[None,
                                                      [QuantizerConfig(num_bits=4), ],
                                                      [QuantizerConfig(num_bits=5), ]]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(num_bits=4), ],
                        [QuantizerConfig(num_bits=5), ]])
            }),

        # A branch will either have to be requantized or not for the merge, based on the
        # final choice of the branch qconfig w.r.t. the merge qconfig
        # 5
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=6)],
                [QuantizerConfig(num_bits=7), ],
                [QuantizerConfig(num_bits=8), ],
                [QuantizerConfig(num_bits=7), ]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=6)],
                        [QuantizerConfig(num_bits=7), ],
                        [QuantizerConfig(num_bits=8), ],
                        [QuantizerConfig(num_bits=7), ]]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=6)],
                        [QuantizerConfig(num_bits=7), ],
                        [QuantizerConfig(num_bits=8), ],
                        [QuantizerConfig(num_bits=7), ]]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=8), ],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=6)],
                        [QuantizerConfig(num_bits=7), ],
                        None,
                        [QuantizerConfig(num_bits=7), ]
                    ]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=6)],
                        [QuantizerConfig(num_bits=7), ],
                        [QuantizerConfig(num_bits=8), ],
                        [QuantizerConfig(num_bits=7), ]
                    ])
            }),

        # 6
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)]]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)]]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        None,
                        None
                    ]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[
                        None,
                        None,
                        None
                    ])
            }),

        # 7
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=6), ],
                [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6)],
                [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)],
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6), ],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)], ]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6), ],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)], ]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=8), ],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6), ],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)],
                    ]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6), ],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)],
                    ])
            }),

        # 8
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=6), ],
                [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6),
                 QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5),
                 QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6), ],
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5),
                         QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)]]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6), ],
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5),
                         QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)]]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6), ],
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5),
                         QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)],
                    ]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=6), ],
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5),
                         QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)],
                    ])
            }),

        # 9
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=3)],
                [QuantizerConfig(num_bits=7),
                 QuantizerConfig(num_bits=6),
                 QuantizerConfig(num_bits=6,
                                 mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(num_bits=8),
                 QuantizerConfig(num_bits=5),
                 QuantizerConfig(num_bits=4,
                                 mode=QuantizationMode.ASYMMETRIC)]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=3)],
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5),
                         QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)]]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=3)],
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5),
                         QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)]]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=3), ],
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5),
                         QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)],
                    ]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=3), ],
                        [QuantizerConfig(num_bits=7), QuantizerConfig(num_bits=6),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5),
                         QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)],
                    ])
            }),

        # 10
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC),
                 QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)]]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)]]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)]
                    ]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=4, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC),
                         QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=5)]
                    ])
            }),

        # Real-world scenarios
        # 11
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
                 QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                 QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                 QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                 QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)]]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)]]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[
                        QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                        QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False),
                        QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True), ],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)]
                    ]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                    branch_qconfig_lists_after_merge=[
                        None,
                        None])
            }),

        # 12
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC),
                 QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC),
                 QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
                 QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)]
            ],
            strategy_vs_solution_dict={
                QuantizerPropagationRule.DO_NOT_MERGE_BRANCHES: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)]]),
                QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME : MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)]]),
                QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)]
                    ]),
                QuantizerPropagationRule.MERGE_ALL_IN_ONE: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[
                        None,
                        None,
                        None
                    ])
            })
    ]  # fmt: skip

    @staticmethod
    @pytest.fixture(params=QCONFIG_PRIMARY_SECONDARY_BEFORE_AND_AFTER_MERGING)
    def qconfig_merge_test_struct(request):
        return request.param

    def test_get_merged_qconfigs(self, qconfig_merge_test_struct: MergeQConfigTestStruct):
        for strategy in QuantizerPropagationRule:
            quant_prop_solver = QuantizerPropagationSolver(propagation_strategy=strategy, run_consistency_checks=True)
            solution_for_strategy = qconfig_merge_test_struct.strategy_vs_solution_dict[strategy]
            ref_merge_qconfig_list = solution_for_strategy.merge_qconfig_list
            ref_branch_qconfig_lists_after_merge = solution_for_strategy.branch_qconfig_lists_after_merge

            (
                merge_qconfig_list,
                branch_qconfig_lists_after_merge,
            ) = quant_prop_solver.get_merged_qconfigs_for_downward_branching_case(
                qconfig_merge_test_struct.branch_qconfig_lists_before_merge
            )

            assert ref_merge_qconfig_list == merge_qconfig_list
            assert ref_branch_qconfig_lists_after_merge == branch_qconfig_lists_after_merge

    def test_merged_qconfig_list_is_independent_of_branch_qconfig_list_order(
        self, qconfig_merge_test_struct: MergeQConfigTestStruct
    ):
        quant_prop_solver = QuantizerPropagationSolver(
            propagation_strategy=QuantizerPropagationRule.MERGE_WITH_POTENTIAL_REQUANTIZATION
        )
        branch_qconfig_lists_before_merge = qconfig_merge_test_struct.branch_qconfig_lists_before_merge
        ref_merge_qconfig_list, _ = quant_prop_solver.get_merged_qconfigs_for_downward_branching_case(
            branch_qconfig_lists_before_merge
        )

        for permutation in permutations(branch_qconfig_lists_before_merge):
            test_merge_qconfig_list, _ = quant_prop_solver.get_merged_qconfigs_for_downward_branching_case(permutation)
            assert ref_merge_qconfig_list == test_merge_qconfig_list

    BRANCHING_MODEL_GRAPH = get_branching_model_graph()

    class InitNodeTestStruct:
        def __init__(self, quantization_trait, config, op_meta=UnknownMetatype):
            self.quantization_trait = quantization_trait
            self.config = config
            self.op_meta = op_meta

    @dataclass
    class BranchTransitionTestStruct:
        # Unspecified nodes are marked as quantization agnostic
        init_node_to_trait_and_configs_dict: Dict[str, "TestQuantizerPropagationSolver.InitNodeTestStruct"]
        starting_primary_quantizer_ip_node: str
        target_branching_node_for_primary_quantizer: str
        expected_status: TransitionStatus
        nncf_graph_builder: Callable[[], NNCFGraph] = None

    BRANCH_TRANSITION_TEST_CASES = [
        # Scaled dot product attention case
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '5 /scaled_dot_product_attention_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                                             QuantizerConfig(), ScaledDotProductAttentionMetatype),
            },
            starting_primary_quantizer_ip_node=
                InsertionPointGraph.get_pre_hook_node_key('5 /scaled_dot_product_attention_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('1 /branch_node_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION,
            nncf_graph_builder=get_scaled_dot_product_graph
        ),

        # Downward branches are quantization-agnostic
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                                             QuantizerConfig()),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('4 /D_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_TRANSITION
        ),

        # Downward branches have quantizers that are still propagating
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                                             [QuantizerConfig()]),
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                                             [QuantizerConfig()]),
                '5 /E_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                                             [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('6 /F_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=7)]),
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=6)]),
                '5 /E_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=4)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('5 /E_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)]),
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)]),
                '5 /E_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=4), QuantizerConfig(num_bits=6)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('5 /E_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=4), QuantizerConfig(num_bits=6)]),

                '3 /C_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)]),

                '5 /E_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=8), QuantizerConfig(num_bits=6, mode=QuantizationMode.ASYMMETRIC)]),

                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=4)]),
                '7 /G_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=4)]),
                '8 /H_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=4)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('4 /D_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=5)]),
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=6)]),
                '5 /E_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=4), QuantizerConfig(num_bits=6)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('5 /E_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=7, mode=QuantizationMode.ASYMMETRIC)]),
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=6), QuantizerConfig(num_bits=8)]),
                '5 /E_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=4), QuantizerConfig(num_bits=6)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('6 /F_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        # A branch has a non-quantizable op
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.NON_QUANTIZABLE,
                      []),
                '5 /E_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('5 /E_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.NON_QUANTIZABLE,
                      []),
                '10 /J_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('6 /F_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),


        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.NON_QUANTIZABLE,
                      []),
                '5 /E_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('5 /E_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                '4 /D_0': InitNodeTestStruct(QuantizationTrait.NON_QUANTIZABLE,
                      []),
                '5 /E_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('5 /E_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Transition impacts a concat node that won't be quantized
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '7 /G_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                '8 /H_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      []),
                '9 /I_0': InitNodeTestStruct(QuantizationTrait.CONCAT,
                      []),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('7 /G_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('5 /E_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Transition impacts a concat node that is, by this point in time, quantized
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '7 /G_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                '8 /H_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      []),
                '9 /I_0': InitNodeTestStruct(QuantizationTrait.CONCAT,
                      [QuantizerConfig(num_bits=6)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('7 /G_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('5 /E_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      []),
                '11 /K_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      [], OutputNoopMetatype),
                '12 /L_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('12 /L_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('6 /F_0'),
            expected_status=TransitionStatus.SHOULD_TRANSITION
        ),
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      []),
                '11 /K_0': InitNodeTestStruct(QuantizationTrait.NON_QUANTIZABLE,
                      []),
                '12 /L_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('12 /L_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('6 /F_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      []),
                '13 /M_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      [], OutputNoopMetatype),
                '12 /L_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                '11 /K_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      []),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('12 /L_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('6 /F_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                '6 /F_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      []),
                '13 /M_0': InitNodeTestStruct(QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      [], OutputNoopMetatype),
                '12 /L_0': InitNodeTestStruct(QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                '11 /K_0': InitNodeTestStruct(QuantizationTrait.NON_QUANTIZABLE,
                      []),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('12 /L_0'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('6 /F_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        )
    ]  # fmt: skip

    @staticmethod
    @pytest.fixture(params=BRANCH_TRANSITION_TEST_CASES)
    def branch_transition_test_struct(request):
        return request.param

    def test_check_branching_transition(self, branch_transition_test_struct: BranchTransitionTestStruct):
        init_node_to_trait_and_configs_dict = branch_transition_test_struct.init_node_to_trait_and_configs_dict
        starting_primary_quantizer_ip_node = branch_transition_test_struct.starting_primary_quantizer_ip_node
        target_node = branch_transition_test_struct.target_branching_node_for_primary_quantizer
        expected_status = branch_transition_test_struct.expected_status

        # Graph preparation
        if branch_transition_test_struct.nncf_graph_builder is None:
            nncf_graph = get_branching_model_graph()
        else:
            nncf_graph = branch_transition_test_struct.nncf_graph_builder()
        ip_graph = get_ip_graph_for_test(nncf_graph)

        # Metatypes must be assigned before QPSG creation, because
        # QPSG detects outputs based on the metatype
        metatypes = {k: v.op_meta for k, v in init_node_to_trait_and_configs_dict.items()}
        for node_key, metatype in metatypes.items():
            node = ip_graph.nodes[node_key]
            node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR].attributes[NNCFNode.METATYPE_ATTR] = metatype

        quant_prop_graph = QPSG(ip_graph)
        for node in quant_prop_graph.nodes.values():
            node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.QUANTIZATION_AGNOSTIC

        primary_prop_quant = None
        for node_key, init_node_struct in init_node_to_trait_and_configs_dict.items():
            qconfigs = init_node_struct.config
            trait = init_node_struct.quantization_trait
            quant_prop_graph.nodes[node_key][QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = trait
            if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                target_input_ports = [0]
                metatype = quant_prop_graph.nodes[node_key]["op_meta"]
                if metatype.target_input_ports is not None:
                    target_input_ports = metatype.target_input_ports

                for input_port_id in target_input_ports:
                    ip_node_key = InsertionPointGraph.get_pre_hook_node_key(node_key, input_port_id=input_port_id)
                    prop_quant = quant_prop_graph.add_propagating_quantizer(qconfigs, ip_node_key)
                    if ip_node_key == starting_primary_quantizer_ip_node:
                        primary_prop_quant = prop_quant
            elif trait == QuantizationTrait.CONCAT and qconfigs:
                # Assuming two-port concat nodes are used in the test graph, adjust as necessary
                for input_port_id in [0, 1]:
                    ip_node_key = InsertionPointGraph.get_pre_hook_node_key(node_key, input_port_id=input_port_id)
                    quant_prop_graph.add_propagating_quantizer(qconfigs, ip_node_key)

        path = get_edge_paths_for_propagation(quant_prop_graph, target_node, starting_primary_quantizer_ip_node)

        primary_prop_quant = quant_prop_graph.propagate_quantizer_via_path(primary_prop_quant, path[0])
        quant_prop_graph.run_consistency_check()

        # The propagating quantizers are in place, now check the transition
        solver = QuantizerPropagationSolver(run_consistency_checks=True)
        status = solver.check_branching_transition(quant_prop_graph, primary_prop_quant, target_node)
        assert status == expected_status

    PathTransitionTestStruct = namedtuple(
        "PathTransitionTestStruct",
        (
            "init_node_to_trait_configs_and_target_node_dict",
            # Unspecified nodes are marked as quantization agnostic
            "starting_primary_quantizer_ip_node",
            "primary_quantizer_qconfigs",
            "target_node_for_primary_quantizer",
            "expected_status",
        ),
    )

    @staticmethod
    def prepare_propagation_graph_state(
        ip_graph: InsertionPointGraph, init_node_to_trait_configs_and_target_node_dict: Dict[str, Tuple]
    ) -> Tuple[List[PropagatingQuantizer], QPSG]:
        quant_prop_graph = QPSG(ip_graph)
        prop_quantizers = []
        for node in quant_prop_graph.nodes.values():
            node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.QUANTIZATION_AGNOSTIC

        for node_key, trait_configs_and_target_tuple in init_node_to_trait_configs_and_target_node_dict.items():
            trait = trait_configs_and_target_tuple[0]
            qconfigs = trait_configs_and_target_tuple[1]
            target_node = trait_configs_and_target_tuple[2]
            quant_prop_graph.nodes[node_key][QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = trait
            if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                ip_node_key = InsertionPointGraph.get_pre_hook_node_key(node_key)
                prop_quant = quant_prop_graph.add_propagating_quantizer(qconfigs, ip_node_key)
                if target_node is not None:
                    path = get_edge_paths_for_propagation(quant_prop_graph, target_node, ip_node_key)
                    prop_quant = quant_prop_graph.propagate_quantizer_via_path(prop_quant, path[0])
                prop_quantizers.append(prop_quant)

        quant_prop_graph.run_consistency_check()
        return prop_quantizers, quant_prop_graph

    PATH_TRANSITION_TEST_CASES = [
        # Transition cases

        # Single propagating quantizer
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('10 /J_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('5 /E_0'),
            expected_status=TransitionStatus.SHOULD_TRANSITION
        ),

        # Non-intersecting paths, no branch influence
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '4 /D_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_post_hook_node_key('1 /A_0')),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('6 /F_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('3 /C_0'),
            expected_status=TransitionStatus.SHOULD_TRANSITION
        ),

        # Non-intersecting paths, branch influence
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '6 /F_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=6)],
                      InsertionPointGraph.get_pre_hook_node_key('3 /C_0')),
                '7 /G_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('5 /E_0'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('4 /D_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('1 /A_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        # Non-intersecting paths, branch influence with downward branch config narrowing
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '6 /F_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=6), QuantizerConfig(num_bits=8)],
                      InsertionPointGraph.get_pre_hook_node_key('3 /C_0')),
                '7 /G_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=6), QuantizerConfig(num_bits=5,
                                                                    mode=QuantizationMode.ASYMMETRIC)],
                      InsertionPointGraph.get_pre_hook_node_key('5 /E_0'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('4 /D_0'),
            primary_quantizer_qconfigs=[QuantizerConfig(num_bits=6)],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('1 /A_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        # Merge cases
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '5 /E_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('1 /A_0'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('3 /C_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_MERGE
        ),

        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '7 /G_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('1 /A_0')),
                '6 /F_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=2)],
                      InsertionPointGraph.get_pre_hook_node_key('3 /C_0'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('8 /H_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_MERGE
        ),

        # No transition cases:

        # Path blocked by a quantizer
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '3 /C_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('2 /B_0')),
                '10 /J_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=4)],
                      InsertionPointGraph.get_pre_hook_node_key('9 /I_0')),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('6 /F_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('3 /C_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Path blocked by a non-quantizable node
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '10 /J_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('8 /H_0')),
                '3 /C_0': (QuantizationTrait.NON_QUANTIZABLE,
                      [],
                      None)
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('6 /F_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('1 /A_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),


        # A downward branch node was marked as non-quantizable
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '7 /G_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('5 /E_0')),
                '4 /D_0': (QuantizationTrait.NON_QUANTIZABLE,
                      [],
                      None)
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('3 /C_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Incompatible upstream quantizer
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '5 /E_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(6)],
                      InsertionPointGraph.get_post_hook_node_key('1 /A_0')),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('4 /D_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '5 /E_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(6)],
                      InsertionPointGraph.get_post_hook_node_key('1 /A_0'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('4 /D_0'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('2 /B_0'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Incompatible downstream quantizers
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '6 /F_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=6), QuantizerConfig(num_bits=8)],
                      InsertionPointGraph.get_pre_hook_node_key('3 /C_0')),
                '7 /G_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(num_bits=6), QuantizerConfig(num_bits=5,
                                                                    mode=QuantizationMode.ASYMMETRIC)],
                      InsertionPointGraph.get_pre_hook_node_key('5 /E_0'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('4 /D_0'),
            primary_quantizer_qconfigs=[QuantizerConfig(num_bits=4)],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('1 /A_0'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),
    ]  # fmt: skip

    @staticmethod
    @pytest.fixture(params=PATH_TRANSITION_TEST_CASES)
    def path_transition_test_struct(request):
        return request.param

    def test_check_transition_via_path(self, path_transition_test_struct: PathTransitionTestStruct):
        init_node_to_trait_configs_and_target_node_dict = (
            path_transition_test_struct.init_node_to_trait_configs_and_target_node_dict
        )
        starting_primary_quantizer_ip_node = path_transition_test_struct.starting_primary_quantizer_ip_node
        primary_quantizer_qconfigs = path_transition_test_struct.primary_quantizer_qconfigs
        target_node = path_transition_test_struct.target_node_for_primary_quantizer
        ref_status = path_transition_test_struct.expected_status

        # Graph preparation
        mock_graph = get_branching_model_graph()
        ip_graph = get_ip_graph_for_test(mock_graph)
        _, quant_prop_graph = self.prepare_propagation_graph_state(
            ip_graph, init_node_to_trait_configs_and_target_node_dict
        )

        primary_prop_quant = quant_prop_graph.add_propagating_quantizer(
            primary_quantizer_qconfigs, starting_primary_quantizer_ip_node
        )
        quant_prop_graph.run_consistency_check()
        path = get_edge_paths_for_propagation(quant_prop_graph, target_node, starting_primary_quantizer_ip_node)[0]

        solver = QuantizerPropagationSolver(run_consistency_checks=True)
        status = solver.check_transition_via_path(primary_prop_quant, path, quant_prop_graph)
        assert status == ref_status

    PropagationStepTestStruct = namedtuple(
        "PropagationStepTestStruct",
        (
            "init_node_to_trait_configs_and_target_node_dict",
            "expected_finished_status",
            "current_location_node_key_for_propagated_quant",
            "added_quantizer_location_node_keys",
        ),
    )
    PROPAGATION_STEP_TEST_CASES = [
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '6 /F_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_post_hook_node_key('0 /O_0'))
            },
            expected_finished_status=True,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_post_hook_node_key('0 /O_0'),
            added_quantizer_location_node_keys=[]
        ),
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                '6 /F_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('3 /C_0'))
            },
            expected_finished_status=False,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_pre_hook_node_key('3 /C_0'),
            added_quantizer_location_node_keys=[]
        ),
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict={
                '6 /F_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('1 /A_0')),
                '7 /G_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('5 /E_0')),
                '10 /J_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('9 /I_0'))
            },
            expected_finished_status=False,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_pre_hook_node_key('1 /A_0'),
            added_quantizer_location_node_keys=[]
        ),

        # Covers the case where the quantizer should be cloned
        # (i.e. when passing through an upward branching node)
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict={
                '10 /J_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_post_hook_node_key('9 /I_0')),
                '6 /F_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('1 /A_0')),
                '7 /G_0': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('5 /E_0')),
            },
            expected_finished_status=False,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_post_hook_node_key('9 /I_0'),
            added_quantizer_location_node_keys=[InsertionPointGraph.get_pre_hook_node_key('9 /I_0', input_port_id=1)]
        )
    ]  # fmt: skip

    @staticmethod
    @pytest.fixture(params=PROPAGATION_STEP_TEST_CASES)
    def propagation_step_test_struct(request):
        return request.param

    def test_propagation_step(self, propagation_step_test_struct: PropagationStepTestStruct):
        init_node_to_trait_configs_and_target_node_dict = (
            propagation_step_test_struct.init_node_to_trait_configs_and_target_node_dict
        )
        expected_finished_status = propagation_step_test_struct.expected_finished_status
        current_location_node_key_for_propagated_quant = (
            propagation_step_test_struct.current_location_node_key_for_propagated_quant
        )
        # Graph preparation
        mock_graph = get_branching_model_graph()
        ip_graph = get_ip_graph_for_test(mock_graph)
        quant_prop_solver = QuantizerPropagationSolver(run_consistency_checks=True)
        prop_quantizers, quant_prop_graph = self.prepare_propagation_graph_state(
            ip_graph, init_node_to_trait_configs_and_target_node_dict
        )
        untouched_quantizers = []
        quant_prop = None
        for pq in prop_quantizers:
            if pq.current_location_node_key == current_location_node_key_for_propagated_quant:
                quant_prop = pq
            else:
                untouched_quantizers.append(pq)

        assert quant_prop is not None
        quant_prop_graph = quant_prop_solver.propagation_step(quant_prop, quant_prop_graph)
        quant_prop_graph.run_consistency_check()

        if expected_finished_status:
            finished_propagating_quantizers = quant_prop_solver.get_finished_propagating_quantizers()
            assert quant_prop in finished_propagating_quantizers
        else:
            active_propagating_quantizers_queue = quant_prop_solver.get_active_propagating_quantizers_queue()
            assert quant_prop in active_propagating_quantizers_queue

        for pq in untouched_quantizers:
            assert pq not in quant_prop_solver.get_active_propagating_quantizers_queue()
            assert pq not in quant_prop_solver.get_finished_propagating_quantizers()

        # The quantizers that were added during preparation were not registered
        # as active for the solvers; but the ones that may have appeared due to an upward
        # branching transition will be registered, and so will the propagated quantizer
        quantizers_count_after_step = quant_prop_solver.get_total_quantizer_count()
        # Should be true for non-merge cases
        assert quantizers_count_after_step == 1 + len(propagation_step_test_struct.added_quantizer_location_node_keys)

    def test_handling_upward_branching_path_with_no_transition_creates_no_extra_quantizers(self, mocker):
        # Graph preparation
        mock_graph = get_branching_model_graph()
        ip_graph = get_ip_graph_for_test(mock_graph)
        quant_prop_solver = QuantizerPropagationSolver()
        prep_data_dict = {
            "9 /I_0": (QuantizationTrait.NON_QUANTIZABLE, [], None),
            "10 /J_0": (
                QuantizationTrait.INPUTS_QUANTIZABLE,
                [QuantizerConfig()],
                InsertionPointGraph.get_post_hook_node_key("9 /I_0"),
            ),
        }

        prop_quantizers, quant_prop_graph = self.prepare_propagation_graph_state(ip_graph, prep_data_dict)
        assert len(prop_quantizers) == 1
        pq = prop_quantizers[0]
        mocker.spy(quant_prop_graph, "remove_propagating_quantizer")
        mocker.spy(quant_prop_graph, "clone_propagating_quantizer")
        _ = quant_prop_solver.propagation_step(pq, quant_prop_graph)
        finished_pqs = quant_prop_solver.get_finished_propagating_quantizers()

        assert quant_prop_graph.remove_propagating_quantizer.call_count == 1
        assert quant_prop_graph.clone_propagating_quantizer.call_count == 1
        assert len(finished_pqs) == 1
        assert finished_pqs[0] is pq
        assert not quant_prop_solver.get_active_propagating_quantizers_queue()
        for edge in quant_prop_graph.edges():
            edge_attrs = quant_prop_graph.edges[edge]
            affecting_quantizers = edge_attrs[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            assert (not affecting_quantizers) or (len(affecting_quantizers) == 1 and pq in affecting_quantizers)
        for node_attrs in quant_prop_graph.nodes.values():
            if QPSG.is_insertion_point(node_attrs[QPSG.NODE_TYPE_NODE_ATTR]):
                affecting_pq = node_attrs[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR]
                assert (affecting_pq is pq) or (affecting_pq is None)

    RUN_ON_IP_GRAPH_TEST_CASES = [
        RunOnIpGraphTestStruct(
            base_nx_graph=get_sequentially_connected_model_graph(['conv2d', 'batch_norm']),
            retval_qp_data={1: MultiQPSerializedDataForTest(TargetType.OPERATOR_POST_HOOK,
                                                            "/conv2d_0",
                                                            [QuantizerConfig()],
                                                            directly_quantized_op_node_names=[
                                                                "/batch_norm_0"
                                                            ])},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}],
            expected_count_finished_quant=1,
            expected_count_active_quant=0,
            ignored_scopes=None
        ),
        RunOnIpGraphTestStruct(
            base_nx_graph=get_sequentially_connected_model_graph(['conv2d', 'gelu', 'conv2d']),
            retval_qp_data={1: MultiQPSerializedDataForTest(TargetType.OPERATOR_POST_HOOK,
                                                            "/conv2d_0",
                                                            [QuantizerConfig()],
                                                            directly_quantized_op_node_names=[
                                                                "/gelu_0"
                                                            ]
                                                            ),
                            2: MultiQPSerializedDataForTest(TargetType.OPERATOR_POST_HOOK,
                                                            "/gelu_0",
                                                            [QuantizerConfig()],
                                                            directly_quantized_op_node_names=[
                                                                "/conv2d_1"])},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}, {2}],
            expected_count_finished_quant=2,
            expected_count_active_quant=0,
            ignored_scopes=None
        ),
        RunOnIpGraphTestStruct(
            base_nx_graph=get_sequentially_connected_model_graph(['conv2d', 'matmul', 'gelu', 'softmax']),
            retval_qp_data={1: MultiQPSerializedDataForTest(TargetType.OPERATOR_POST_HOOK,
                                                            "/conv2d_0", [QuantizerConfig()],
                                                            directly_quantized_op_node_names=["/matmul_0"])},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}],
            expected_count_finished_quant=1,
            expected_count_active_quant=0,
            ignored_scopes={'/gelu_0': IgnoreReason.USER_REQUESTED, '/conv2d_0': IgnoreReason.USER_REQUESTED}
        ),
        RunOnIpGraphTestStruct(
            base_nx_graph=get_sequentially_connected_model_graph(['conv2d', 'matmul']),
            retval_qp_data={1: MultiQPSerializedDataForTest(TargetType.OPERATOR_POST_HOOK,
                                                            "/conv2d_0",
                                                            [QuantizerConfig()],
                                                            directly_quantized_op_node_names=["/matmul_0"])},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}],
            expected_count_finished_quant=1,
            expected_count_active_quant=0,
            ignored_scopes={'/conv2d_0': IgnoreReason.USER_REQUESTED}
        ),
        RunOnIpGraphTestStruct(
            base_nx_graph=get_sequentially_connected_model_graph(['conv2d', 'matmul']),
            retval_qp_data={},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[],
            expected_count_finished_quant=0,
            expected_count_active_quant=0,
            ignored_scopes={'/conv2d_0': IgnoreReason.USER_REQUESTED, '/matmul_0': IgnoreReason.USER_REQUESTED}
        ),
        RunOnIpGraphTestStruct(
            base_nx_graph=TwoFcAfterDropout.get_graph(),
            retval_qp_data={1: MultiQPSerializedDataForTest(TargetType.OPERATOR_PRE_HOOK,
                                                            TwoFcAfterDropout.FC_1_NODE_NAME, [QuantizerConfig()],
                                                            input_port_id=0,
                                                            directly_quantized_op_node_names=[
                                                                TwoFcAfterDropout.FC_1_NODE_NAME]
                                                            )},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}],
            expected_count_finished_quant=1,
            expected_count_active_quant=0,
            ignored_scopes={TwoFcAfterDropout.FC_2_NODE_NAME: IgnoreReason.USER_REQUESTED}
        )
    ]  # fmt: skip

    @staticmethod
    @pytest.fixture(params=RUN_ON_IP_GRAPH_TEST_CASES)
    def run_on_ip_graph_test_struct(request):
        return request.param

    def test_run_on_ip_graph(self, run_on_ip_graph_test_struct: RunOnIpGraphTestStruct):
        expected_count_finished_quant = run_on_ip_graph_test_struct.expected_count_finished_quant
        expected_count_active_quant = run_on_ip_graph_test_struct.expected_count_active_quant

        # Graph preparation
        nncf_graph = run_on_ip_graph_test_struct.base_graph
        ip_graph = get_ip_graph_for_test(nncf_graph)

        if run_on_ip_graph_test_struct.ignored_scopes is not None:
            weight_ignored_scopes = list(run_on_ip_graph_test_struct.ignored_scopes.keys())
        else:
            weight_ignored_scopes = None
        quant_prop_solver = QuantizerPropagationSolver(
            activation_ignored_scopes=run_on_ip_graph_test_struct.ignored_scopes,
            weight_ignored_scopes=weight_ignored_scopes,
            default_trait_to_metatype_map=DEFAULT_TEST_QUANT_TRAIT_MAP,
            run_consistency_checks=True,
        )
        retval = quant_prop_solver.run_on_ip_graph(ip_graph)

        assert retval.quantizer_setup.quantization_points == run_on_ip_graph_test_struct.retval_qps
        assert (
            list(retval.quantizer_setup.unified_scale_groups.values())
            == run_on_ip_graph_test_struct.retval_unified_scale_qp_groups
        )
        assert (
            list(retval.quantizer_setup.shared_input_operation_set_groups.values())
            == run_on_ip_graph_test_struct.retval_shared_input_operation_set_groups
        )

        assert len(quant_prop_solver.get_active_propagating_quantizers_queue()) == expected_count_active_quant
        assert len(quant_prop_solver.get_finished_propagating_quantizers()) == expected_count_finished_quant

    @pytest.fixture()
    def ip_graph_with_int_edges(self):
        mock_graph = nx.DiGraph()

        # Double edges stand for integer data flows
        #        (0 /O_0)  <-- treating this as an auxiliary "input" node
        #          ||
        #        (1 /A_0)
        #           |
        #        (2 /B_0)
        #       //    \
        #    (3 /C_0) (4 /D_0)
        #       \\    /
        #        (5 /E_0)
        #          |
        #        (6 /F_0)

        node_keys = ["O", "A", "B", "C", "D", "E", "F"]
        for node_key in node_keys:
            mock_node_attrs = get_mock_nncf_node_attrs(op_name=node_key)
            mock_graph.add_node(node_key, **mock_node_attrs)

        mock_graph.add_edges_from([("O", "A"), ("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"), ("E", "F")])

        for edge in [("O", "A"), ("B", "C"), ("C", "E")]:
            mock_graph.edges[edge][NNCFGraph.DTYPE_EDGE_ATTR] = Dtype.INTEGER

        mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
        nncf_graph = get_nncf_graph_from_mock_nx_graph(mock_graph)
        ip_graph = get_ip_graph_for_test(nncf_graph)
        return ip_graph

    class IntEdgePropagationTestStruct:
        def __init__(
            self,
            initial_node_name: str,
            target_node_name: Optional[str],
            path_to_propagate: PropagationPath,
            expected_status: TransitionStatus,
        ):
            self.initial_node_name = initial_node_name
            self.target_node_name = target_node_name
            self.path_to_propagate = path_to_propagate
            self.expected_status = expected_status

    INT_EDGE_PROPAGATION_CASES = [
        IntEdgePropagationTestStruct(
            initial_node_name="2 /B_0",
            target_node_name=InsertionPointGraph.get_post_hook_node_key("1 /A_0"),
            path_to_propagate=[
                (InsertionPointGraph.get_pre_hook_node_key("1 /A_0", input_port_id=0), "1 /A_0"),
                ("1 /A_0", InsertionPointGraph.get_post_hook_node_key("1 /A_0"))
            ],
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        IntEdgePropagationTestStruct(
            initial_node_name="6 /F_0",
            target_node_name=InsertionPointGraph.get_post_hook_node_key("5 /E_0"),
            path_to_propagate=[
                (InsertionPointGraph.get_pre_hook_node_key("5 /E_0", input_port_id=0), "5 /E_0"),
                ("5 /E_0", InsertionPointGraph.get_post_hook_node_key("5 /E_0"))
            ],
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        IntEdgePropagationTestStruct(
            initial_node_name="6 /F_0",
            target_node_name=InsertionPointGraph.get_post_hook_node_key("5 /E_0"),
            path_to_propagate=[
                (InsertionPointGraph.get_pre_hook_node_key("5 /E_0", input_port_id=1), "5 /E_0"),
                ("5 /E_0", InsertionPointGraph.get_post_hook_node_key("5 /E_0"))
            ],
            expected_status=TransitionStatus.SHOULD_TRANSITION
        ),

        # In this case trying to transition to a post-hook of an op that has both integer
        # and float outputs; since separate output port handling is currently not supported,
        # prohibit the transition so that the quantizers stays at the safe floating point tensor
        # edge.
        IntEdgePropagationTestStruct(
            initial_node_name="4 /D_0",
            target_node_name=None,
            path_to_propagate=[
                (InsertionPointGraph.get_post_hook_node_key("2 /B_0"),
                 InsertionPointGraph.get_pre_hook_node_key("4 /D_0", input_port_id=0)),
            ],
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        )
    ]  # fmt: skip

    @pytest.fixture(params=INT_EDGE_PROPAGATION_CASES)
    def int_prop_test_struct(self, request):
        return request.param

    def test_quantizers_are_not_propagated_through_integer_paths(
        self, ip_graph_with_int_edges: InsertionPointGraph, int_prop_test_struct: IntEdgePropagationTestStruct
    ):
        quant_prop_solver = QuantizerPropagationSolver()
        prep_data_dict = {
            int_prop_test_struct.initial_node_name: (
                QuantizationTrait.INPUTS_QUANTIZABLE,
                [QuantizerConfig()],
                int_prop_test_struct.target_node_name,
            )
        }

        prop_quantizers, quant_prop_graph = self.prepare_propagation_graph_state(
            ip_graph_with_int_edges, prep_data_dict
        )
        assert len(prop_quantizers) == 1
        pq = next(iter(prop_quantizers))
        quant_prop_graph.run_consistency_check()
        status = quant_prop_solver.check_transition_via_path(
            pq, int_prop_test_struct.path_to_propagate, quant_prop_graph
        )
        assert status == int_prop_test_struct.expected_status

    def test_quantizers_are_not_set_up_for_integer_inputs(self, ip_graph_with_int_edges):
        quant_prop_solver = QuantizerPropagationSolver()
        quant_prop_graph = QPSG(ip_graph_with_int_edges)
        for node in quant_prop_graph.nodes.values():
            node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.QUANTIZATION_AGNOSTIC

        quantizable_op_node_keys = ["1 /A_0", "4 /D_0", "5 /E_0"]
        for node_key in quantizable_op_node_keys:
            quant_prop_graph.nodes[node_key][QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.INPUTS_QUANTIZABLE
        quant_prop_graph.run_consistency_check()

        _ = quant_prop_solver.setup_initial_quantizers(quant_prop_graph)
        all_pqs = list(quant_prop_solver.get_active_propagating_quantizers_queue())

        # Set for the single-flop input op and for one of the two inputs of the double-input op, not set for the
        # single-input int op
        assert len(all_pqs) == 2
        affected_op_nodes = [pq.affected_operator_nodes for pq in all_pqs]
        assert all(len(node_key_list) == 1 for node_key_list in affected_op_nodes)
        affected_op_node_per_pq = [next(iter(node_key_list)) for node_key_list in affected_op_nodes]
        assert Counter(["4 /D_0", "5 /E_0"]) == Counter(affected_op_node_per_pq)
        double_input_pq = all_pqs[affected_op_node_per_pq.index("5 /E_0")]
        assert double_input_pq.current_location_node_key == InsertionPointGraph.get_pre_hook_node_key(
            "5 /E_0", input_port_id=1
        )


def test_metatypes_to_ignore(mocker):
    # pylint: disable=protected-access
    NOT_IGNORED_METATYHPE = "not_ignored_metatype"
    IGNORED_METATYPE = "target_metatype"

    nncf_graph = NNCFGraph()
    nodes = []
    for node_name, node_metatype in zip("ABC", [NOT_IGNORED_METATYHPE, IGNORED_METATYPE, NOT_IGNORED_METATYHPE]):
        nodes.append(nncf_graph.add_nncf_node(node_name, node_name, node_metatype=node_metatype))
    for idx in range(1, len(nodes)):
        nncf_graph.add_edge_between_nncf_nodes(
            nodes[idx - 1].node_id, nodes[idx].node_id, [1, 1, 1, 1], 0, 0, Dtype.FLOAT
        )
    ip_graph = InsertionPointGraph(nncf_graph=nncf_graph)

    solver = QuantizerPropagationSolver(
        metatypes_to_ignore=[IGNORED_METATYPE],
    )
    solver._add_node_to_ignored = mocker.MagicMock()
    solver.run_on_ip_graph(ip_graph)

    solver._add_node_to_ignored.assert_called_once()
    assert "1 B" in solver._add_node_to_ignored.call_args[0]
