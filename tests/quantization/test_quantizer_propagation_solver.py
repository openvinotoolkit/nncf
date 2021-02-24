# pylint:disable=too-many-lines
"""
 Copyright (c) 2019-2020 Intel Corporation
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
import random
from collections import namedtuple
from itertools import permutations
from typing import Dict, List, Tuple

import networkx as nx
import pytest
from nncf.quantization.quantizer_setup import MultiConfigQuantizationPoint

from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph import OperationExecutionContext, NNCFGraph, InputAgnosticOperationExecutionContext
from nncf.dynamic_graph.version_agnostic_op_names import get_version_agnostic_name
from nncf.nncf_network import InsertionPointGraph, InsertionPointGraphNodeType, InsertionPoint, \
    InsertionType
from nncf.quantization.layers import QuantizerConfig, QuantizationMode
from nncf.quantization.quantizer_propagation import QuantizerPropagationStateGraph as QPSG, \
    QuantizerPropagationStateGraphNodeType, QuantizationTrait, OPERATOR_METATYPES, DEFAULT_QUANT_TRAIT_TO_OP_DICT, \
    QuantizerPropagationSolver, TransitionStatus, PropagationStrategy, PropagatingQuantizer
from tests.quantization.test_quantizer_propagation_graph import get_edge_paths_for_propagation
from tests.test_nncf_network import get_mock_nncf_node_attrs, mark_input_ports_lexicographically_based_on_input_node_key


def get_mock_model_node_attrs_for_op_name(op_name: str, call_order=0) -> OperationExecutionContext:
    return OperationExecutionContext(op_name,
                                     Scope(),
                                     call_order,
                                     [None])


def get_randomly_connected_model_graph(op_name_keys: List[str]) -> nx.DiGraph:
    graph_len = len(op_name_keys)
    mock_graph = nx.generators.gnc_graph(graph_len, seed=0)
    shuffled_op_names = random.sample(op_name_keys, len(op_name_keys))
    for idx, (_, node) in enumerate(mock_graph.nodes.items()):
        node[NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR] = get_mock_model_node_attrs_for_op_name(shuffled_op_names[idx])
    mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
    return mock_graph


def get_sequentially_connected_model_graph(op_name_keys: List[str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    node_key_appearances = {k: 0 for k in op_name_keys}

    actual_keys = []
    for node_key in op_name_keys:
        attrs = {
            NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR:
                get_mock_model_node_attrs_for_op_name(node_key, call_order=node_key_appearances[node_key])
        }
        actual_key = node_key + '_{}'.format(node_key_appearances[node_key])
        graph.add_node(actual_key, **attrs)
        node_key_appearances[node_key] += 1
        actual_keys.append(actual_key)

    edges = [(actual_keys[i], actual_keys[i + 1]) for i in range(0, len(actual_keys) - 1)]
    for from_key, to_key in edges:
        graph.add_edge(from_key, to_key)

    mark_input_ports_lexicographically_based_on_input_node_key(graph)
    return graph


class TwoFcAfterDropout:
    DROPOUT_OPERATION_EXECUTION_CONTEXT = OperationExecutionContext('dropout',
                                                                    Scope.from_str('TwoFcAfterDropoutModel'),
                                                                    0,
                                                                    [None])
    FC_1_OPERATION_EXECUTION_CONTEXT = OperationExecutionContext('linear',
                                                                 Scope.from_str(
                                                                     'TwoFcAfterDropoutModel/NNCFLinear[branch1]'),
                                                                 0,
                                                                 [None])

    FC_2_SCOPE_STR = 'TwoFcAfterDropoutModel/NNCFLinear[branch2]'
    FC_2_OPERATION_EXECUTION_CONTEXT = OperationExecutionContext('linear',
                                                                 Scope.from_str(FC_2_SCOPE_STR),
                                                                 0,
                                                                 [None])

    @staticmethod
    def get_graph():
        graph = nx.DiGraph()
        dropout_node_attrs = {
            NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR: TwoFcAfterDropout.DROPOUT_OPERATION_EXECUTION_CONTEXT
        }

        fc_1_node_attrs = {
            NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR: TwoFcAfterDropout.FC_1_OPERATION_EXECUTION_CONTEXT
        }

        fc_2_node_attrs = {
            NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR: TwoFcAfterDropout.FC_2_OPERATION_EXECUTION_CONTEXT
        }

        graph.add_node('dropout', **dropout_node_attrs)
        graph.add_node('fc_1', **fc_1_node_attrs)
        graph.add_node('fc_2', **fc_2_node_attrs)
        graph.add_edge('dropout', 'fc_1')
        graph.add_edge('dropout', 'fc_2')

        mark_input_ports_lexicographically_based_on_input_node_key(graph)
        return graph

class TestQuantizerPropagationSolver:
    def test_quantization_traits_are_unambiguous_for_op_names(self):
        op_name_to_trait_dict = {}  # type: Dict[str, QuantizationTrait]
        for trait, arches in DEFAULT_QUANT_TRAIT_TO_OP_DICT.items():
            for op_meta in arches:
                aliases = op_meta.get_all_aliases()
                for alias in aliases:
                    if alias in op_name_to_trait_dict:
                        assert op_name_to_trait_dict[alias] == trait
                    else:
                        op_name_to_trait_dict[alias] = trait

    def test_set_quantization_traits_for_quant_prop_graph_nodes(self):
        # Test all patchable metatypes. If a patchable metatype is not registered
        # in quantization trait-to-metatype dict, the test will fail.
        tested_op_metatypes = list(OPERATOR_METATYPES.registry_dict.values()) # type: List[OperatorMetatype]
        tested_op_names = []
        for op_meta in tested_op_metatypes:
            aliases = op_meta.get_all_aliases()
            for alias in aliases:
                tested_op_names.append(get_version_agnostic_name(alias))

        # Edges should be irrelevant - using random graph
        mock_graph = get_randomly_connected_model_graph(tested_op_names)
        ip_graph = InsertionPointGraph(mock_graph)
        for node in ip_graph.nodes.values():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                op_exec_context = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR][
                    NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                op_name = op_exec_context.operator_name
                ref_meta = OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
                node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR] = ref_meta

        quant_prop_graph = QPSG(ip_graph)
        quant_prop_solver = QuantizerPropagationSolver(run_consistency_checks=True)
        quant_prop_graph = quant_prop_solver.set_allowed_quantization_types_for_operator_nodes(quant_prop_graph)
        op_quant_traits_map = quant_prop_solver.get_operator_quantization_traits_map()

        for qpg_node in quant_prop_graph.nodes().values():
            if qpg_node[QPSG.NODE_TYPE_NODE_ATTR] == QuantizerPropagationStateGraphNodeType.OPERATOR:
                quant_det_id = qpg_node[QPSG.OPERATOR_METATYPE_NODE_ATTR]
                quant_types = qpg_node[QPSG.ALLOWED_INPUT_QUANTIZATION_TYPES_NODE_ATTR]
                if op_quant_traits_map[quant_det_id] == QuantizationTrait.INPUTS_QUANTIZABLE:
                    # TODO: check for correspondence of operator type and HW config to initial
                    # quantization types
                    assert quant_types == QuantizerPropagationSolver.DEFAULT_QUANTIZATION_TYPES

    def test_setup_initial_quantizers_in_quant_prop_graph(self):
        ops_to_quantize = ['batch_norm', 'conv2d', 'matmul', 'gelu']
        ops_not_to_quantize = ['max_pool2d', 'dropout', 'min', 'softmax']
        node_keys = ['nncf_model_input'] + ops_to_quantize + ops_not_to_quantize
        mock_graph = get_sequentially_connected_model_graph(node_keys)

        ip_graph = InsertionPointGraph(mock_graph)
        for node in ip_graph.nodes.values():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                op_exec_context = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR][
                    NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                op_name = op_exec_context.operator_name
                ref_meta = OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
                node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR] = ref_meta

        qp_graph = QPSG(ip_graph)
        quant_prop_solver = QuantizerPropagationSolver(run_consistency_checks=True)
        qp_graph = quant_prop_solver.set_allowed_quantization_types_for_operator_nodes(qp_graph)
        qp_graph = quant_prop_solver.setup_initial_quantizers(qp_graph)
        qp_graph.run_consistency_check()

        for node_key in ops_to_quantize:
            actual_key = node_key + '_0'
            pred_ip_key = next(qp_graph.predecessors(actual_key))
            node = qp_graph.nodes[actual_key]
            pred_ip_node = qp_graph.nodes[pred_ip_key]
            prop_quant = pred_ip_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR]
            assert prop_quant is not None
            assert node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]

            edge = qp_graph.edges[pred_ip_key, actual_key]
            assert edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR] == [prop_quant]

        for node_key in ops_not_to_quantize:
            actual_key = node_key + '_0'
            pred_ip_key = next(qp_graph.predecessors(actual_key))
            node = qp_graph.nodes[actual_key]
            pred_ip_node = qp_graph.nodes[pred_ip_key]
            assert pred_ip_node[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR] is None

            assert not node[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]
            edge = qp_graph.edges[pred_ip_key, actual_key]
            assert not edge[QPSG.AFFECTING_PROPAGATING_QUANTIZERS_ATTR]

    MergeQConfigSolution = namedtuple('MergeQConfigSolution',
                                      ('merge_qconfig_list', 'branch_qconfig_lists_after_merge'))
    MergeQConfigTestStruct = namedtuple('MergeQConfigTestStruct',
                                        ('branch_qconfig_lists_before_merge',
                                         'strategy_vs_solution_dict'))
    QCONFIG_PRIMARY_SECONDARY_BEFORE_AND_AFTER_MERGING = [
        # Compatible configs on all branches
        # 0
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=8), ],
                [QuantizerConfig(bits=8), ],
                [QuantizerConfig(bits=8), ]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8), ],
                        [QuantizerConfig(bits=8), ],
                        [QuantizerConfig(bits=8), ]
                    ], ),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None])
            }),

        # 1
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), ]
            ],

            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), ]
                    ]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), ],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None,
                                                      None]
                ),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None,
                                                      None]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None,
                                                      None])
            }),

        # 2
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(bits=8)],
                [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(bits=8)],
                [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(bits=8)],
            ],

            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(bits=8)],
                        [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(bits=8)],
                        [QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC), QuantizerConfig(bits=8)],
                    ]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC),
                                        QuantizerConfig(bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None]
                ),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC),
                                        QuantizerConfig(bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None]),

                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC),
                                        QuantizerConfig(bits=8)],
                    branch_qconfig_lists_after_merge=[None,
                                                      None,
                                                      None])
            }),

        # Requantization necessary for merge
        # 3
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=6), ],
                [QuantizerConfig(bits=4), ],
                [QuantizerConfig(bits=5), ]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[[QuantizerConfig(bits=6), ],
                                                      [QuantizerConfig(bits=4), ],
                                                      [QuantizerConfig(bits=5), ]]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[[QuantizerConfig(bits=6), ],
                                                      [QuantizerConfig(bits=4), ],
                                                      [QuantizerConfig(bits=5), ]]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=6)],
                    branch_qconfig_lists_after_merge=[None,
                                                      [QuantizerConfig(bits=4), ],
                                                      [QuantizerConfig(bits=5), ]]),

                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[[QuantizerConfig(bits=6), ],
                                                      [QuantizerConfig(bits=4), ],
                                                      [QuantizerConfig(bits=5), ]])
            }),

        # 4
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                [QuantizerConfig(bits=4), ],
                [QuantizerConfig(bits=5), ]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(bits=4), ],
                        [QuantizerConfig(bits=5), ]]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(bits=4), ],
                        [QuantizerConfig(bits=5), ]]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                    branch_qconfig_lists_after_merge=[None,
                                                      [QuantizerConfig(bits=4), ],
                                                      [QuantizerConfig(bits=5), ]]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC), ],
                        [QuantizerConfig(bits=4), ],
                        [QuantizerConfig(bits=5), ]])
            }),

        # A branch will either have to be requantized or not for the merge, based on the
        # final choice of the branch qconfig w.r.t. the merge qconfig
        # 5
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=8), QuantizerConfig(bits=6)],
                [QuantizerConfig(bits=7), ],
                [QuantizerConfig(bits=8), ],
                [QuantizerConfig(bits=7), ]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=6)],
                        [QuantizerConfig(bits=7), ],
                        [QuantizerConfig(bits=8), ],
                        [QuantizerConfig(bits=7), ]]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=6)],
                        [QuantizerConfig(bits=7), ],
                        [QuantizerConfig(bits=8), ],
                        [QuantizerConfig(bits=7), ]]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=8), ],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=6)],
                        [QuantizerConfig(bits=7), ],
                        None,
                        [QuantizerConfig(bits=7), ]
                    ]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=6)],
                        [QuantizerConfig(bits=7), ],
                        [QuantizerConfig(bits=8), ],
                        [QuantizerConfig(bits=7), ]
                    ])
            }),

        # 6
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=7), QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)]]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)]]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)],
                        None,
                        None
                    ]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[
                        None,
                        None,
                        None
                    ])
            }),

        # 7
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=6), ],
                [QuantizerConfig(bits=8), QuantizerConfig(bits=7), QuantizerConfig(bits=6)],
                [QuantizerConfig(bits=8), QuantizerConfig(bits=5)],
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6), ],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=7), QuantizerConfig(bits=6)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5)], ]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6), ],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=7), QuantizerConfig(bits=6)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5)], ]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=8), ],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6), ],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=7), QuantizerConfig(bits=6)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5)],
                    ]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6), ],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=7), QuantizerConfig(bits=6)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5)],
                    ])
            }),

        # 8
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=6), ],
                [QuantizerConfig(bits=7), QuantizerConfig(bits=6),
                 QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(bits=8), QuantizerConfig(bits=5),
                 QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6), ],
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=6),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5),
                         QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)]]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6), ],
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=6),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5),
                         QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)]]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6), ],
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=6),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5),
                         QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)],
                    ]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=6), ],
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=6),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5),
                         QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)],
                    ])
            }),

        # 9
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=3)],
                [QuantizerConfig(bits=7), QuantizerConfig(bits=6), QuantizerConfig(bits=6,
                                                                                   mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(bits=8), QuantizerConfig(bits=5), QuantizerConfig(bits=4,
                                                                                   mode=QuantizationMode.ASYMMETRIC)]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=3)],
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=6),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5),
                         QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)]]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=3)],
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=6),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5),
                         QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)]]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=3), ],
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=6),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5),
                         QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)],
                    ]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=3), ],
                        [QuantizerConfig(bits=7), QuantizerConfig(bits=6),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5),
                         QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)],
                    ])
            }),

        # 10
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=8), QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC),
                 QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(bits=8), QuantizerConfig(bits=5)]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5)]]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5)]]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5)]
                    ]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=4, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC),
                         QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8), QuantizerConfig(bits=5)]
                    ])
            }),

        # Real-world scenarios
        # 11
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
                 QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                 QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                 QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                 QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                         QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                        [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)]]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                         QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                        [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)]]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[
                        QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                        QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False),
                        QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True), ],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True),
                         QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                        [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)]
                    ]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False)],
                    branch_qconfig_lists_after_merge=[
                        None,
                        None])
            }),

        # 12
        MergeQConfigTestStruct(
            branch_qconfig_lists_before_merge=[
                [QuantizerConfig(bits=4, mode=QuantizationMode.SYMMETRIC),
                 QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(bits=4, mode=QuantizationMode.SYMMETRIC),
                 QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC),
                 QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)]
            ],
            strategy_vs_solution_dict={
                PropagationStrategy.DO_NOT_MERGE_BRANCH_FQS: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)]]),
                PropagationStrategy.MERGE_IF_ALL_BRANCH_FQ_OPTIONS_SAME: MergeQConfigSolution(
                    merge_qconfig_list=None,
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)]]),
                PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[
                        [QuantizerConfig(bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=4, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                        [QuantizerConfig(bits=8, mode=QuantizationMode.SYMMETRIC),
                         QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)]
                    ]),
                PropagationStrategy.MERGE_WITH_SINGLE_FQ_RESULT: MergeQConfigSolution(
                    merge_qconfig_list=[QuantizerConfig(bits=8, mode=QuantizationMode.ASYMMETRIC)],
                    branch_qconfig_lists_after_merge=[
                        None,
                        None,
                        None
                    ])
            })
    ]

    @staticmethod
    @pytest.fixture(params=QCONFIG_PRIMARY_SECONDARY_BEFORE_AND_AFTER_MERGING)
    def qconfig_merge_test_struct(request):
        return request.param

    def test_get_merged_qconfigs(self, qconfig_merge_test_struct: MergeQConfigTestStruct):
        for strategy in PropagationStrategy:
            quant_prop_solver = QuantizerPropagationSolver(propagation_strategy=strategy,
                                                           run_consistency_checks=True)
            solution_for_strategy = qconfig_merge_test_struct.strategy_vs_solution_dict[strategy]
            ref_merge_qconfig_list = solution_for_strategy.merge_qconfig_list
            ref_branch_qconfig_lists_after_merge = solution_for_strategy.branch_qconfig_lists_after_merge

            merge_qconfig_list, branch_qconfig_lists_after_merge = \
                quant_prop_solver.get_merged_qconfigs_for_downward_branching_case(
                    qconfig_merge_test_struct.branch_qconfig_lists_before_merge
                )

            assert ref_merge_qconfig_list == merge_qconfig_list
            assert ref_branch_qconfig_lists_after_merge == branch_qconfig_lists_after_merge

    def test_merged_qconfig_list_is_independent_of_branch_qconfig_list_order(self,
                                                                             qconfig_merge_test_struct:
                                                                             MergeQConfigTestStruct):
        quant_prop_solver = QuantizerPropagationSolver(
            propagation_strategy=PropagationStrategy.MERGE_WITH_POTENTIAL_REQUANTIZATION)
        branch_qconfig_lists_before_merge = qconfig_merge_test_struct.branch_qconfig_lists_before_merge
        ref_merge_qconfig_list, _ = quant_prop_solver.get_merged_qconfigs_for_downward_branching_case(
            branch_qconfig_lists_before_merge)

        for permutation in permutations(branch_qconfig_lists_before_merge):
            test_merge_qconfig_list, _ = quant_prop_solver.get_merged_qconfigs_for_downward_branching_case(permutation)
            assert ref_merge_qconfig_list == test_merge_qconfig_list

    def get_branching_model_graph(self):
        mock_node_attrs = get_mock_nncf_node_attrs()
        mock_graph = nx.DiGraph()

        #     (O)  <-- treating this as an auxiliary "input" node
        #      |
        #     (A)
        #      |
        #     (B)
        #   /  |  \
        # (C) (D) (E)
        #  |       | \
        # (F)     (G) (H)
        #           \ /
        #           (I)
        #            |
        #           (J)

        node_keys = ['O', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        for node_key in node_keys:
            mock_graph.add_node(node_key, **mock_node_attrs)

        mock_graph.add_edges_from([('O', 'A'),
                                   ('A', 'B'), ('B', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F'),
                                   ('E', 'G'), ('E', 'H'), ('G', 'I'), ('H', 'I'), ('I', 'J')])

        mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
        return mock_graph

    BranchTransitionTestStruct = namedtuple('BranchTransitionTestStruct',
                                            (  # Unspecified nodes are marked as quantization agnostic
                                                'init_node_to_trait_and_configs_dict',
                                                'starting_primary_quantizer_ip_node',
                                                'target_branching_node_for_primary_quantizer',
                                                'expected_status'))

    BRANCH_TRANSITION_TEST_CASES = [
        # Downward branches are quantization-agnostic
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_TRANSITION
        ),

        # Downward branches have quantizers that are still propagating
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=7)]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=8), QuantizerConfig(bits=6)]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=8), QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4), QuantizerConfig(bits=6)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4), QuantizerConfig(bits=6)]),

                'C': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)]),

                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=8), QuantizerConfig(bits=6, mode=QuantizationMode.ASYMMETRIC)]),

                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)]),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)]),
                'H': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)]),

            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=5)]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6)]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4), QuantizerConfig(bits=6)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=7, mode=QuantizationMode.ASYMMETRIC)]),
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=8)]),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4), QuantizerConfig(bits=6)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        # A branch has a non-quantizable op
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      []),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      []),
                'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),


        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      []),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      []),
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('E'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Transition impacts a concat node that won't be quantized
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'H': (QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      []),
                'I': (QuantizationTrait.CONCAT,
                      []),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('G'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('E'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Transition impacts a concat node that is, by this point in time, quantized
        BranchTransitionTestStruct(
            init_node_to_trait_and_configs_dict=
            {
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()]),
                'H': (QuantizationTrait.QUANTIZATION_AGNOSTIC,
                      []),
                'I': (QuantizationTrait.CONCAT,
                      [QuantizerConfig(bits=6)]),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('G'),
            target_branching_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('E'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),
    ]

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
        mock_graph = self.get_branching_model_graph()
        ip_graph = InsertionPointGraph(mock_graph)
        quant_prop_graph = QPSG(ip_graph)
        for node in quant_prop_graph.nodes.values():
            node[QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = QuantizationTrait.QUANTIZATION_AGNOSTIC

        primary_prop_quant = None
        for node_key, trait_and_configs_tuple in init_node_to_trait_and_configs_dict.items():
            trait = trait_and_configs_tuple[0]
            qconfigs = trait_and_configs_tuple[1]
            quant_prop_graph.nodes[node_key][QPSG.QUANTIZATION_TRAIT_NODE_ATTR] = trait
            if trait == QuantizationTrait.INPUTS_QUANTIZABLE:
                ip_node_key = InsertionPointGraph.get_pre_hook_node_key(node_key)
                prop_quant = quant_prop_graph.add_propagating_quantizer(qconfigs,
                                                                        ip_node_key)
                if ip_node_key == starting_primary_quantizer_ip_node:
                    primary_prop_quant = prop_quant
            elif trait == QuantizationTrait.CONCAT and qconfigs:
                # Assuming two-port concat nodes are used in the test graph, adjust as necessary
                for in_port_id in [0, 1]:
                    ip_node_key = InsertionPointGraph.get_pre_hook_node_key(node_key, in_port_id=in_port_id)
                    quant_prop_graph.add_propagating_quantizer(qconfigs,
                                                               ip_node_key)

        path = get_edge_paths_for_propagation(quant_prop_graph,
                                              target_node,
                                              starting_primary_quantizer_ip_node)
        primary_prop_quant = quant_prop_graph.propagate_quantizer_via_path(primary_prop_quant,
                                                                           path[0])
        quant_prop_graph.run_consistency_check()

        # The propagating quantizers are in place, now check the transition
        solver = QuantizerPropagationSolver(run_consistency_checks=True)
        status = solver.check_branching_transition(quant_prop_graph,
                                                   primary_prop_quant,
                                                   target_node)
        assert status == expected_status

    PathTransitionTestStruct = namedtuple('PathTransitionTestStruct',
                                          ('init_node_to_trait_configs_and_target_node_dict',
                                           # Unspecified nodes are marked as quantization agnostic
                                           'starting_primary_quantizer_ip_node',
                                           'primary_quantizer_qconfigs',
                                           'target_node_for_primary_quantizer',
                                           'expected_status'))

    @staticmethod
    def prepare_propagation_graph_state(ip_graph: InsertionPointGraph,
                                        init_node_to_trait_configs_and_target_node_dict: Dict[
                                            str, Tuple]) -> Tuple[List[PropagatingQuantizer], QPSG]:
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
                prop_quant = quant_prop_graph.add_propagating_quantizer(qconfigs,
                                                                        ip_node_key)
                if target_node is not None:
                    path = get_edge_paths_for_propagation(quant_prop_graph,
                                                          target_node,
                                                          ip_node_key)
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
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('J'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('E'),
            expected_status=TransitionStatus.SHOULD_TRANSITION
        ),

        # Non-intersecting paths, no branch influence
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'D': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_post_hook_node_key('A')),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('C'),
            expected_status=TransitionStatus.SHOULD_TRANSITION
        ),

        # Non-intersecting paths, branch influence
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6)],
                      InsertionPointGraph.get_pre_hook_node_key('C')),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('E'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('A'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        # Non-intersecting paths, branch influence with downward branch config narrowing
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=8)],
                      InsertionPointGraph.get_pre_hook_node_key('C')),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=5,
                                                                mode=QuantizationMode.ASYMMETRIC)],
                      InsertionPointGraph.get_pre_hook_node_key('E'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            primary_quantizer_qconfigs=[QuantizerConfig(bits=6)],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('A'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

        # Merge cases
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('A'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('C'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_MERGE
        ),

        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('A')),
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=2)],
                      InsertionPointGraph.get_pre_hook_node_key('C'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('H'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_MERGE
        ),

        # No transition cases:

        # Path blocked by a quantizer
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'C': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('B')),
                'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=4)],
                      InsertionPointGraph.get_pre_hook_node_key('I')),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('C'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Path blocked by a non-quantizable node
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('H')),
                'C': (QuantizationTrait.NON_QUANTIZABLE,
                      [],
                      None)
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('F'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('A'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),


        # A downward branch node was marked as non-quantizable
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
                'D': (QuantizationTrait.NON_QUANTIZABLE,
                      [],
                      None)
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('C'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Incompatible upstream quantizer
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(6)],
                      InsertionPointGraph.get_post_hook_node_key('A')),
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'E': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(6)],
                      InsertionPointGraph.get_post_hook_node_key('A'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            primary_quantizer_qconfigs=[QuantizerConfig()],
            target_node_for_primary_quantizer=InsertionPointGraph.get_pre_hook_node_key('B'),
            expected_status=TransitionStatus.SHOULD_NOT_TRANSITION
        ),

        # Incompatible downstream quantizers
        PathTransitionTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=8)],
                      InsertionPointGraph.get_pre_hook_node_key('C')),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig(bits=6), QuantizerConfig(bits=5,
                                                                mode=QuantizationMode.ASYMMETRIC)],
                      InsertionPointGraph.get_pre_hook_node_key('E'))
            },
            starting_primary_quantizer_ip_node=InsertionPointGraph.get_pre_hook_node_key('D'),
            primary_quantizer_qconfigs=[QuantizerConfig(bits=4)],
            target_node_for_primary_quantizer=InsertionPointGraph.get_post_hook_node_key('A'),
            expected_status=TransitionStatus.SHOULD_WAIT_FOR_MERGE
        ),

    ]

    @staticmethod
    @pytest.fixture(params=PATH_TRANSITION_TEST_CASES)
    def path_transition_test_struct(request):
        return request.param

    def test_check_transition_via_path(self, path_transition_test_struct: PathTransitionTestStruct):
        #pylint:disable=line-too-long
        init_node_to_trait_configs_and_target_node_dict = path_transition_test_struct.init_node_to_trait_configs_and_target_node_dict
        starting_primary_quantizer_ip_node = path_transition_test_struct.starting_primary_quantizer_ip_node
        primary_quantizer_qconfigs = path_transition_test_struct.primary_quantizer_qconfigs
        target_node = path_transition_test_struct.target_node_for_primary_quantizer
        ref_status = path_transition_test_struct.expected_status

        # Graph preparation
        mock_graph = self.get_branching_model_graph()
        ip_graph = InsertionPointGraph(mock_graph)
        _, quant_prop_graph = self.prepare_propagation_graph_state(ip_graph,
                                                                   init_node_to_trait_configs_and_target_node_dict)

        primary_prop_quant = quant_prop_graph.add_propagating_quantizer(primary_quantizer_qconfigs,
                                                                        starting_primary_quantizer_ip_node)
        quant_prop_graph.run_consistency_check()
        path = get_edge_paths_for_propagation(quant_prop_graph,
                                              target_node,
                                              starting_primary_quantizer_ip_node)[0]

        solver = QuantizerPropagationSolver(run_consistency_checks=True)
        status = solver.check_transition_via_path(primary_prop_quant,
                                                  path,
                                                  quant_prop_graph)
        assert status == ref_status

    PropagationStepTestStruct = namedtuple('PropagationStepTestStruct',
                                           ('init_node_to_trait_configs_and_target_node_dict',
                                            'expected_finished_status',
                                            'current_location_node_key_for_propagated_quant',
                                            'added_quantizer_location_node_keys'))
    PROPAGATION_STEP_TEST_CASES = [
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_post_hook_node_key('O'))
            },
            expected_finished_status=True,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_post_hook_node_key('O'),
            added_quantizer_location_node_keys=[]
        ),
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict=
            {
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('C'))
            },
            expected_finished_status=False,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_pre_hook_node_key('C'),
            added_quantizer_location_node_keys=[]
        ),
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict={
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('A')),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
                'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('I'))
            },
            expected_finished_status=False,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_pre_hook_node_key('A'),
            added_quantizer_location_node_keys=[]
        ),

        # Covers the case where the quantizer should be cloned
        # (i.e. when passing through an upward branching node)
        PropagationStepTestStruct(
            init_node_to_trait_configs_and_target_node_dict={
                'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_post_hook_node_key('I')),
                'F': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('A')),
                'G': (QuantizationTrait.INPUTS_QUANTIZABLE,
                      [QuantizerConfig()],
                      InsertionPointGraph.get_pre_hook_node_key('E')),
            },
            expected_finished_status=False,
            current_location_node_key_for_propagated_quant=InsertionPointGraph.get_post_hook_node_key('I'),
            added_quantizer_location_node_keys=[InsertionPointGraph.get_pre_hook_node_key('I', in_port_id=1)]
        )
    ]

    @staticmethod
    @pytest.fixture(params=PROPAGATION_STEP_TEST_CASES)
    def propagation_step_test_struct(request):
        return request.param

    def test_propagation_step(self, propagation_step_test_struct: PropagationStepTestStruct):
        # pylint:disable=line-too-long
        init_node_to_trait_configs_and_target_node_dict = propagation_step_test_struct.init_node_to_trait_configs_and_target_node_dict
        expected_finished_status = propagation_step_test_struct.expected_finished_status
        current_location_node_key_for_propagated_quant = propagation_step_test_struct.current_location_node_key_for_propagated_quant
        # Graph preparation
        mock_graph = self.get_branching_model_graph()
        ip_graph = InsertionPointGraph(mock_graph)
        quant_prop_solver = QuantizerPropagationSolver(run_consistency_checks=True)
        # pylint:disable=line-too-long
        prop_quantizers, quant_prop_graph = self.prepare_propagation_graph_state(ip_graph,
                                                                                 init_node_to_trait_configs_and_target_node_dict)
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
            assert not pq in quant_prop_solver.get_active_propagating_quantizers_queue()
            assert not pq in quant_prop_solver.get_finished_propagating_quantizers()

        # The quantizers that were added during preparation were not registered
        # as active for the solvers; but the ones that may have appeared due to an upward
        # branching transition will be registered, and so will the propagated quantizer
        quantizers_count_after_step = quant_prop_solver.get_total_quantizer_count()
        # Should be true for non-merge cases
        assert quantizers_count_after_step == 1 + len(propagation_step_test_struct.added_quantizer_location_node_keys)

    def test_handling_upward_branching_path_with_no_transition_creates_no_extra_quantizers(self, mocker):
        # Graph preparation
        mock_graph = self.get_branching_model_graph()
        ip_graph = InsertionPointGraph(mock_graph)
        quant_prop_solver = QuantizerPropagationSolver()
        prep_data_dict = {

            'I': (QuantizationTrait.NON_QUANTIZABLE,
                  [],
                  None),
            'J': (QuantizationTrait.INPUTS_QUANTIZABLE,
                  [QuantizerConfig()],
                  InsertionPointGraph.get_post_hook_node_key('I'))
        }

        prop_quantizers, quant_prop_graph = self.prepare_propagation_graph_state(ip_graph,
                                                                                 prep_data_dict)
        assert len(prop_quantizers) == 1
        pq = prop_quantizers[0]
        mocker.spy(quant_prop_graph, "remove_propagating_quantizer")
        mocker.spy(quant_prop_graph, "clone_propagating_quantizer")
        _ = quant_prop_solver.propagation_step(pq, quant_prop_graph)
        finished_pqs = quant_prop_solver.get_finished_propagating_quantizers()

        #pylint:disable=no-member
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
            if node_attrs[QPSG.NODE_TYPE_NODE_ATTR] == QuantizerPropagationStateGraphNodeType.INSERTION_POINT:
                affecting_pq = node_attrs[QPSG.PROPAGATING_QUANTIZER_NODE_ATTR]
                assert (affecting_pq is pq) or (affecting_pq is None)


    RunOnIpGraphTestStruct = namedtuple('RunOnIpGraphTestStruct',
                                        ('base_graph',
                                         'retval_qps',
                                         'retval_unified_scale_qp_groups',
                                         'retval_shared_input_operation_set_groups',
                                         'expected_count_finished_quant',
                                         'expected_count_active_quant',
                                         'ignored_scope'))


    RUN_ON_IP_GRAPH_TEST_CASES = [
        RunOnIpGraphTestStruct(
            base_graph=get_sequentially_connected_model_graph(['conv2d', 'batch_norm']),
            retval_qps={1: MultiConfigQuantizationPoint(
                InsertionPoint(InsertionType.OPERATOR_POST_HOOK,
                               ia_op_exec_context=InputAgnosticOperationExecutionContext.from_str("/conv2d_0")),
                [QuantizerConfig()])},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}],
            expected_count_finished_quant=1,
            expected_count_active_quant=0,
            ignored_scope=None
        ),
        RunOnIpGraphTestStruct(
            base_graph=get_sequentially_connected_model_graph(['conv2d', 'gelu', 'conv2d']),
            retval_qps={1: MultiConfigQuantizationPoint(
                InsertionPoint(InsertionType.OPERATOR_POST_HOOK,
                               ia_op_exec_context=InputAgnosticOperationExecutionContext.from_str("/conv2d_0")),
                [QuantizerConfig()]),
                        2: MultiConfigQuantizationPoint(
                            InsertionPoint(InsertionType.OPERATOR_POST_HOOK,
                                           ia_op_exec_context=InputAgnosticOperationExecutionContext.from_str(
                                               "/gelu_0")),
                            [QuantizerConfig()])},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}, {2}],
            expected_count_finished_quant=2,
            expected_count_active_quant=0,
            ignored_scope=None
        ),
        RunOnIpGraphTestStruct(
            base_graph=get_sequentially_connected_model_graph(['conv2d', 'matmul', 'gelu', 'softmax']),
            retval_qps={1: MultiConfigQuantizationPoint(
                InsertionPoint(InsertionType.OPERATOR_POST_HOOK,
                               ia_op_exec_context=InputAgnosticOperationExecutionContext.from_str("/conv2d_0")),
                [QuantizerConfig()])},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}],
            expected_count_finished_quant=1,
            expected_count_active_quant=0,
            ignored_scope=['/gelu_0', '/conv2d_0']
        ),
        RunOnIpGraphTestStruct(
            base_graph=get_sequentially_connected_model_graph(['conv2d', 'matmul']),
            retval_qps={1: MultiConfigQuantizationPoint(
                InsertionPoint(InsertionType.OPERATOR_POST_HOOK,
                               ia_op_exec_context=InputAgnosticOperationExecutionContext.from_str("/conv2d_0")),
                [QuantizerConfig()])},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}],
            expected_count_finished_quant=1,
            expected_count_active_quant=0,
            ignored_scope=['/conv2d_0']
        ),
        RunOnIpGraphTestStruct(
            base_graph=get_sequentially_connected_model_graph(['conv2d', 'matmul']),
            retval_qps={},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[],
            expected_count_finished_quant=0,
            expected_count_active_quant=0,
            ignored_scope=['/conv2d_0', '/matmul_0']
        ),
        RunOnIpGraphTestStruct(
            base_graph=TwoFcAfterDropout.get_graph(),
            retval_qps={1: MultiConfigQuantizationPoint(
                InsertionPoint(InsertionType.OPERATOR_PRE_HOOK,
                               ia_op_exec_context=TwoFcAfterDropout.FC_1_OPERATION_EXECUTION_CONTEXT.input_agnostic,
                               input_port_id=0),
                [QuantizerConfig()])},
            retval_unified_scale_qp_groups=[],
            retval_shared_input_operation_set_groups=[{1}],
            expected_count_finished_quant=1,
            expected_count_active_quant=0,
            ignored_scope=[TwoFcAfterDropout.FC_2_SCOPE_STR]
        )
    ]

    @staticmethod
    @pytest.fixture(params=RUN_ON_IP_GRAPH_TEST_CASES)
    def run_on_ip_graph_test_struct(request):
        return request.param

    def test_run_on_ip_graph(self, run_on_ip_graph_test_struct: RunOnIpGraphTestStruct):
        expected_count_finished_quant = run_on_ip_graph_test_struct.expected_count_finished_quant
        expected_count_active_quant = run_on_ip_graph_test_struct.expected_count_active_quant

        # Graph preparation
        mock_graph = run_on_ip_graph_test_struct.base_graph
        ip_graph = InsertionPointGraph(mock_graph)

        for node in ip_graph.nodes.values():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                op_exec_context = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR][
                    NNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR]
                op_name = op_exec_context.operator_name
                ref_meta = OPERATOR_METATYPES.get_operator_metatype_by_op_name(op_name)
                node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR] = ref_meta

        quant_prop_solver = QuantizerPropagationSolver(ignored_scopes=run_on_ip_graph_test_struct.ignored_scope,
                                                       run_consistency_checks=True)
        retval = quant_prop_solver.run_on_ip_graph(ip_graph)

        assert retval.quantizer_setup.quantization_points == run_on_ip_graph_test_struct.retval_qps
        assert retval.quantizer_setup.unified_scale_groups == run_on_ip_graph_test_struct.retval_unified_scale_qp_groups
        assert retval.quantizer_setup.shared_input_operation_set_groups == \
               run_on_ip_graph_test_struct.retval_shared_input_operation_set_groups

        assert len(quant_prop_solver.get_active_propagating_quantizers_queue()) == expected_count_active_quant
        assert len(quant_prop_solver.get_finished_propagating_quantizers()) == expected_count_finished_quant
