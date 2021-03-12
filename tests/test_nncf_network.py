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
import itertools
from collections import Counter
from pathlib import Path
from typing import List

import networkx as nx
import pytest
import torch
from copy import deepcopy
from torch import nn

from nncf import register_module
from nncf.common.graph.transformations.commands import TargetType
from nncf.dynamic_graph.context import Scope, PreHookId
from nncf.dynamic_graph.graph import InputAgnosticOperationExecutionContext, PTNNCFGraph, OperationExecutionContext
from nncf.dynamic_graph.graph_builder import ModelInputInfo, GraphBuilder
from nncf.dynamic_graph.operator_metatypes import NoopMetatype
from nncf.dynamic_graph.input_wrapping import MODEL_INPUT_OP_NAME
from nncf.dynamic_graph.transformations.layout import PTTransformationLayout
from nncf.dynamic_graph.version_agnostic_op_names import VersionAgnosticNames
from nncf.layer_utils import _NNCFModuleMixin
from nncf.module_operations import BaseOp
from nncf.nncf_network import NNCFNetwork, InsertionPointGraph, InsertionPointGraphNodeType
from nncf.dynamic_graph.transformations.commands import TransformationPriority
from nncf.dynamic_graph.transformations.commands import PTTargetPoint
from nncf.dynamic_graph.transformations.commands import PTInsertionCommand
from nncf.nncf_network import PTInsertionPoint
from nncf.nncf_network import PTInsertionType
from nncf.nncf_network import PTModelTransformer
from tests.composite.test_sparsity_quantization import get_basic_sparsity_plus_quantization_config
from tests.conftest import TEST_ROOT
from tests.helpers import TwoConvTestModel, BasicConvTestModel, check_correct_nncf_modules_replacement, \
    create_compressed_model_and_algo_for_test
from tests.test_models.synthetic import ManyNonEvalModules


def test_disable_shape_matching():
    class MatMulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.ones([1]))

        def forward(self, inputs):
            half1, half2 = torch.chunk(inputs, 2, dim=2)
            return torch.bmm(half1, half2.transpose(1, 2))

    model = MatMulModel()

    input_shape_1 = (3, 32, 32)
    input_shape_2 = (4, 64, 64)

    qnet_no_shape = NNCFNetwork(deepcopy(model), input_infos=[ModelInputInfo(input_shape_1), ],
                                scopes_without_shape_matching=['MatMulModel'])  # type: NNCFNetwork
    _ = qnet_no_shape(torch.zeros(*input_shape_1))
    graph_1 = deepcopy(qnet_no_shape.get_graph())

    _ = qnet_no_shape(torch.zeros(*input_shape_2))
    graph_2 = deepcopy(qnet_no_shape.get_graph())

    keys_1 = list(graph_1.get_all_node_keys())
    keys_2 = list(graph_2.get_all_node_keys())
    assert len(keys_1) == 2  # 1 input node + 1 operation node
    assert keys_1 == keys_2

    qnet = NNCFNetwork(model, input_infos=[ModelInputInfo(input_shape_1), ])  # type: NNCFNetwork
    _ = qnet(torch.zeros(*input_shape_1))
    _ = qnet(torch.zeros(*input_shape_2))
    # The second forward run should have led to an increase in registered node counts
    # since disable_shape_matching was False and the network was run with a different
    # shape of input tensor
    assert qnet.get_graph().get_nodes_count() > graph_1.get_nodes_count()


def test_check_correct_modules_replacement():
    model = TwoConvTestModel()
    nncf_model = NNCFNetwork(TwoConvTestModel(), input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork

    _, nncf_modules = check_correct_nncf_modules_replacement(model, nncf_model)
    assert set(nncf_modules) == set(nncf_model.get_nncf_modules())


@register_module()
class ModuleOfUser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones([1]))

    def forward(self, input_):
        return input_ * self.weight


class TwoConvTestModelWithUserModule(TwoConvTestModel):
    def __init__(self):
        super().__init__()
        self.user_module = ModuleOfUser()

    def forward(self, x):
        x = super().forward(x)
        x = self.user_module(x)
        return x


def test_custom_module_registering():
    model = TwoConvTestModelWithUserModule()
    nncf_model = NNCFNetwork(model, input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork

    from nncf.layers import UNWRAPPED_USER_MODULES
    assert ModuleOfUser in UNWRAPPED_USER_MODULES.registry_dict.values()

    # pylint: disable=protected-access
    assert isinstance(nncf_model.user_module, ModuleOfUser)
    assert isinstance(nncf_model.user_module, _NNCFModuleMixin)
    assert type(nncf_model.user_module).__name__ == "NNCFUserModuleOfUser"

    user_module_attrs = dir(nncf_model.user_module)
    for attr in dir(_NNCFModuleMixin):
        assert attr in user_module_attrs


# pylint: disable=protected-access
def test_find_node_in_nx_graph_by_scope():
    model = TwoConvTestModel()
    nncf_model = NNCFNetwork(deepcopy(model), input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork
    nncf_graph = nncf_model.get_original_graph()

    # Valid scopes should be successfully found
    valid_nncf_modules = nncf_model.get_nncf_modules()
    nodes_list = list(nncf_graph._nx_graph.nodes)
    for module_scope, _ in valid_nncf_modules.items():
        graph_node = nncf_graph.find_node_in_nx_graph_by_scope(module_scope)
        assert graph_node is not None
        assert isinstance(graph_node, dict)
        assert graph_node['key'] in nodes_list

    fake_model = BasicConvTestModel()
    fake_nncf_model = NNCFNetwork(deepcopy(fake_model), input_infos=[ModelInputInfo([1, 1, 4, 4])])

    # Not valid scopes shouldn't be found
    fake_nncf_modules = fake_nncf_model.get_nncf_modules()
    for module_scope, _ in fake_nncf_modules.items():
        graph_node = nncf_graph.find_node_in_nx_graph_by_scope(module_scope)
        assert graph_node is None


class InsertionPointTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, 1)
        self.linear_wts = nn.Parameter(torch.FloatTensor(size=(100, 100)))
        self.conv2 = nn.Conv2d(1, 1, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, input_):
        x = self.conv1(input_)
        x = x.flatten()
        x = nn.functional.linear(x, self.linear_wts)
        x = x.reshape((1, 1, 10, 10))
        x = self.conv2(x)
        x = self.relu(x)
        return x


class TestInsertionCommands:
    @pytest.fixture()
    def setup(self):
        self.compressed_model = NNCFNetwork(InsertionPointTestModel(),
                                            [ModelInputInfo([1, 1, 10, 10])])  # type: NNCFNetwork

    conv1_module_scope = Scope.from_str('InsertionPointTestModel/NNCFConv2d[conv1]')
    point_for_conv1_weights = PTTargetPoint(target_type=TargetType.OPERATION_WITH_WEIGHTS,
                                            module_scope=conv1_module_scope)
    point_for_conv1_inputs = PTTargetPoint(target_type=TargetType.OPERATION_WITH_WEIGHTS,
                                           module_scope=conv1_module_scope)
    point_for_conv1_activations = PTTargetPoint(target_type=TargetType.POST_LAYER_OPERATION,
                                                module_scope=conv1_module_scope)

    conv2_module_scope = Scope.from_str('InsertionPointTestModel/NNCFConv2d[conv2]')
    point_for_conv2_weights = PTTargetPoint(target_type=TargetType.OPERATION_WITH_WEIGHTS,
                                            module_scope=conv2_module_scope)
    point_for_conv2_inputs = PTTargetPoint(target_type=TargetType.OPERATION_WITH_WEIGHTS,
                                           module_scope=conv2_module_scope)
    point_for_conv2_activations = PTTargetPoint(target_type=TargetType.POST_LAYER_OPERATION,
                                                module_scope=conv2_module_scope)

    linear_op_scope = Scope.from_str('InsertionPointTestModel/linear_0')
    linear_op_context = InputAgnosticOperationExecutionContext('linear',
                                                               linear_op_scope,
                                                               0)
    point_for_linear_weight_input = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK,
                                                  ia_op_exec_context=linear_op_context, input_port_id=0)
    point_for_linear_activation = PTTargetPoint(target_type=TargetType.OPERATOR_POST_HOOK,
                                                ia_op_exec_context=linear_op_context)

    relu_op_scope = Scope.from_str('InsertionPointTestModel/ReLU[relu]/relu')
    relu_op_context = InputAgnosticOperationExecutionContext('relu',
                                                             relu_op_scope,
                                                             0)
    point_for_relu_inputs = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK,
                                          ia_op_exec_context=relu_op_context, input_port_id=0)
    point_for_relu_activations = PTTargetPoint(target_type=TargetType.OPERATOR_POST_HOOK,
                                               ia_op_exec_context=relu_op_context)

    available_points = [point_for_conv1_weights,
                        point_for_conv2_weights,
                        point_for_conv1_inputs,
                        point_for_conv2_inputs,
                        point_for_conv1_activations,
                        point_for_conv2_activations,
                        point_for_linear_activation,
                        point_for_linear_weight_input,
                        point_for_relu_activations,
                        point_for_relu_inputs]

    @pytest.mark.parametrize("target_point", available_points)
    def test_single_insertions(self, setup, target_point):
        if target_point.target_type in [TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK]:
            hook = lambda x: x
        else:
            hook = BaseOp(lambda x: x)

        pt_ip = PTInsertionPoint(target_point)
        self.compressed_model.insert_at_point(pt_ip, [hook])

        # pylint:disable=protected-access
        if pt_ip.insertion_type == PTInsertionType.OPERATOR_PRE_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            pre_hook_id = PreHookId(target_point.ia_op_exec_context, input_port_id=target_point.input_port_id)
            assert ctx._pre_hooks[pre_hook_id][0] is hook
        if pt_ip.insertion_type == PTInsertionType.OPERATOR_POST_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            assert ctx._post_hooks[target_point.ia_op_exec_context][0] is hook
        if pt_ip.insertion_type == PTInsertionType.NNCF_MODULE_PRE_OP:
            module = self.compressed_model.get_module_by_scope(target_point.module_scope)
            assert module.pre_ops["0"] is hook

        if pt_ip.insertion_type == PTInsertionType.NNCF_MODULE_POST_OP:
            module = self.compressed_model.get_module_by_scope(target_point.module_scope)
            assert module.post_ops["0"] is hook

    priority_types = ["same", "different"]
    insertion_types = TargetType
    priority_test_cases = list(itertools.product(priority_types, insertion_types))

    @staticmethod
    def check_order(iterable1: List, iterable2: List, ordering: List):
        for idx, order in enumerate(ordering):
            assert iterable1[idx] is iterable2[order]

    # pylint:disable=undefined-variable
    @pytest.mark.parametrize("case", priority_test_cases, ids=[x[1].name + '-' + x[0] for x in priority_test_cases])
    def test_priority(self, case, setup):
        # pylint:disable=too-many-branches
        priority_type = case[0]
        insertion_type = case[1]
        if insertion_type in [TargetType.OPERATION_WITH_WEIGHTS, TargetType.POST_LAYER_OPERATION]:
            hook1 = BaseOp(lambda x: x)
            hook2 = BaseOp(lambda x: 2 * x)
            hook3 = BaseOp(lambda x: 3 * x)
        else:
            hook1 = lambda x: x
            hook2 = lambda x: 2 * x
            hook3 = lambda x: 3 * x

        if insertion_type == TargetType.OPERATION_WITH_WEIGHTS:
            point = self.point_for_conv2_weights
        elif insertion_type == TargetType.POST_LAYER_OPERATION:
            point = self.point_for_conv1_activations
        elif insertion_type == TargetType.OPERATOR_PRE_HOOK:
            point = self.point_for_linear_weight_input
        elif insertion_type == TargetType.OPERATOR_POST_HOOK:
            point = self.point_for_relu_activations
        else:
            pytest.skip("Insertion type {} currently unsupported in PT".format(insertion_type))

        if priority_type == "same":
            # Same-priority commands will be executed in registration order
            command1 = PTInsertionCommand(point, hook1, TransformationPriority.DEFAULT_PRIORITY)
            command2 = PTInsertionCommand(point, hook2, TransformationPriority.DEFAULT_PRIORITY)
            command3 = PTInsertionCommand(point, hook3, TransformationPriority.DEFAULT_PRIORITY)
        else:
            # Prioritized commands will be executed in ascending priority order
            command1 = PTInsertionCommand(point, hook1, TransformationPriority.SPARSIFICATION_PRIORITY)
            command2 = PTInsertionCommand(point, hook2, TransformationPriority.QUANTIZATION_PRIORITY)
            command3 = PTInsertionCommand(point, hook3, TransformationPriority.DEFAULT_PRIORITY)

        layout = PTTransformationLayout()
        layout.register(command1)
        layout.register(command2)
        layout.register(command3)
        self.compressed_model = PTModelTransformer(self.compressed_model, layout).transform()

        hook_list = [hook1, hook2, hook3]

        if priority_type == "same":
            order = [0, 1, 2]
        elif priority_type == "different":
            order = [2, 0, 1]

        # pylint:disable=protected-access
        if insertion_type == TargetType.OPERATOR_PRE_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            pre_hook_id = PreHookId(point.ia_op_exec_context, input_port_id=point.input_port_id)
            self.check_order(ctx._pre_hooks[pre_hook_id], hook_list, order)
        if insertion_type == TargetType.OPERATOR_POST_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            self.check_order(ctx._post_hooks[point.ia_op_exec_context], hook_list, order)

        if insertion_type == TargetType.OPERATION_WITH_WEIGHTS:
            module = self.compressed_model.get_module_by_scope(point.module_scope)
            # Works because Pytorch ModuleDict is ordered
            self.check_order([x.operand for x in module.pre_ops.values()], hook_list, order)

        if insertion_type == TargetType.POST_LAYER_OPERATION:
            module = self.compressed_model.get_module_by_scope(point.module_scope)
            # Works because Pytorch ModuleDict is ordered
            self.check_order(list(module.post_ops.values()), hook_list, order)


def mark_input_ports_lexicographically_based_on_input_node_key(graph: nx.DiGraph):
    for node_key in graph.nodes:
        input_edges = graph.in_edges(node_key)
        sorted_input_edges = sorted(input_edges, key=lambda x: x[0])
        for idx, edge in enumerate(sorted_input_edges):
            graph.edges[edge][PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR] = idx


def get_two_branch_mock_model_graph() -> nx.DiGraph:
    mock_node_attrs = get_mock_nncf_node_attrs()
    mock_graph = nx.DiGraph()

    #   (A)
    #    |
    #   (B)
    #  /   \
    # (C)   (D)
    # |     |
    # (E)   |
    #  \   /
    #   (F)
    #    |
    #   (G)
    #    |
    #   (H)

    node_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for node_key in node_keys:
        mock_graph.add_node(node_key, **mock_node_attrs)

    mock_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('E', 'F'),
                               ('D', 'F'), ('F', 'G'), ('G', 'H')])

    mark_input_ports_lexicographically_based_on_input_node_key(mock_graph)
    return mock_graph


MOCK_OPERATOR_NAME = "conv_transpose2d"


def get_mock_nncf_node_attrs(op_name=None):
    op_name_to_set = op_name if op_name is not None else MOCK_OPERATOR_NAME
    return {
        PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR: OperationExecutionContext(op_name_to_set,
                                                                         Scope(),
                                                                         0,
                                                                         [None])
    }


def get_mock_model_graph_with_mergeable_pattern() -> nx.DiGraph:
    mock_graph = nx.DiGraph()

    #   (A)
    #    |
    #  (conv2d)
    #    |
    # (batch_norm)
    #    |
    #  (RELU)
    #    |
    #   (B)

    node_keys = ['conv2d', 'batch_norm', VersionAgnosticNames.RELU, 'A', 'B']
    for node_key in node_keys:
        mock_graph.add_node(node_key, **get_mock_nncf_node_attrs(op_name=node_key))

    mock_graph.add_edges_from([('A', 'conv2d', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('conv2d', 'batch_norm', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('batch_norm', VersionAgnosticNames.RELU, {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               (VersionAgnosticNames.RELU, 'B', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0})])
    return mock_graph


def get_mock_model_graph_with_no_mergeable_pattern() -> nx.DiGraph:
    mock_graph = nx.DiGraph()

    #   (A)
    #    |
    #  (conv2d)
    #    |
    #   (C)
    #    |
    # (batch_norm)
    #    |
    #   (D)
    #    |
    #  (RELU)
    #    |
    #   (B)

    node_keys = ['conv2d', 'batch_norm', VersionAgnosticNames.RELU, 'A', 'B', 'C', 'D']
    for node_key in node_keys:
        mock_graph.add_node(node_key, **get_mock_nncf_node_attrs(op_name=node_key))

    mock_graph.add_edges_from([('A', 'conv2d', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('conv2d', 'C', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('C', 'batch_norm', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('batch_norm', 'D', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('D', VersionAgnosticNames.RELU, {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               (VersionAgnosticNames.RELU, 'B', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0})])
    return mock_graph


def get_mock_model_graph_with_broken_output_edge_pattern() -> nx.DiGraph:
    mock_graph = nx.DiGraph()

    #   (A)
    #    |
    #  (conv2d)----\
    #    |         |
    # (batch_norm) |
    #    |         |
    #  (RELU)      |
    #    |         |
    #   (C)--------/
    #    |
    #   (B)

    node_keys = ['conv2d', 'batch_norm', VersionAgnosticNames.RELU, 'A', 'B', 'C']
    for node_key in node_keys:
        mock_graph.add_node(node_key, **get_mock_nncf_node_attrs(op_name=node_key))

    mock_graph.add_edges_from([('A', 'conv2d', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('conv2d', 'batch_norm', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('conv2d', 'C', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 1}),
                               ('batch_norm', VersionAgnosticNames.RELU, {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               (VersionAgnosticNames.RELU, 'C', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('C', 'B', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0})])
    return mock_graph


MERGE_PATTERN_TEST_CASES = (
    [get_mock_model_graph_with_mergeable_pattern, "basic_pattern"],
    [get_mock_model_graph_with_no_mergeable_pattern, "no_pattern"],
    [get_mock_model_graph_with_broken_output_edge_pattern, "broken_output_edges_pattern"]
)


class TestInsertionPointGraph:
    def test_insertion_point_setup(self):
        # TODO: Change testing premises when module pre/post-op hooks and input/output nodes
        # are correctly handled
        mock_graph = get_two_branch_mock_model_graph()

        ip_graph = InsertionPointGraph(mock_graph)

        ref_node_len = 3 * len(mock_graph.nodes)  # 2 additional nodes per each operator node
        ref_edge_len = 3 * len(mock_graph.edges)

        assert len(ip_graph.nodes) == ref_node_len
        assert len(ip_graph.edges) == ref_edge_len

        for node_key, node in mock_graph.nodes.items():
            ip_graph_op_node = ip_graph.nodes[node_key]
            assert ip_graph_op_node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR
            preds = list(ip_graph.predecessors(node_key))
            succs = list(ip_graph.successors(node_key))
            assert len(succs) == 1
            post_hook_ip_node_key = succs[0]
            post_hook_ip_node = ip_graph.nodes[succs[0]]
            post_hook_ip_node_type = post_hook_ip_node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
            assert post_hook_ip_node_type == InsertionPointGraphNodeType.INSERTION_POINT

            pre_hook_ip_node_keys = preds
            for pre_hook_ip_node_key in pre_hook_ip_node_keys:
                pre_hook_ip_node = ip_graph.nodes[pre_hook_ip_node_key]
                pre_hook_ip_node_type = pre_hook_ip_node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
                assert pre_hook_ip_node_type == InsertionPointGraphNodeType.INSERTION_POINT

            ref_associated_ip_node_keys_set = {*pre_hook_ip_node_keys, post_hook_ip_node_key}
            assert ref_associated_ip_node_keys_set == ip_graph_op_node[
                InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR]
            original_neighbours = mock_graph.neighbors(node_key)
            for neighbour in original_neighbours:
                # IP node insertion should not disrupt the graph superstructure
                ip_graph_paths = list(nx.all_simple_paths(ip_graph, node_key, neighbour))
                for path in ip_graph_paths:
                    path = path[1:-1]
                    for path_node_key in path:
                        node = ip_graph.nodes[path_node_key]
                        node_type = node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
                        assert node_type == InsertionPointGraphNodeType.INSERTION_POINT

        for node_key, node in ip_graph.nodes.items():
            preds = list(ip_graph.predecessors(node_key))
            succs = list(ip_graph.successors(node_key))
            assert len(preds) != 0 or len(succs) != 0

        for from_node_key, to_node_key in ip_graph.edges.keys():
            assert from_node_key in ip_graph.nodes
            assert to_node_key in ip_graph.nodes

    def test_insertion_point_data_in_ip_nodes(self):
        # TODO: extend for modules
        mock_graph = nx.DiGraph()
        ref_op_exec_context = OperationExecutionContext("baz",
                                                        Scope.from_str("Test/Scope[foo]/bar"),
                                                        0,
                                                        [None])
        node_attrs = {
            PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR: ref_op_exec_context
        }

        node_key = 0
        mock_graph.add_node(node_key, **node_attrs)

        ip_graph = InsertionPointGraph(mock_graph)

        for node_key in mock_graph.nodes.keys():
            preds = list(ip_graph.predecessors(node_key))
            succs = list(ip_graph.successors(node_key))

            post_hook_ip_node = ip_graph.nodes[succs[0]]
            post_hook_ip = post_hook_ip_node[InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]
            assert post_hook_ip.target_type == TargetType.OPERATOR_POST_HOOK
            assert post_hook_ip.ia_op_exec_context == ref_op_exec_context.input_agnostic

            for pre_hook_ip_node_key in preds:
                pre_hook_ip_node = ip_graph.nodes[pre_hook_ip_node_key]
                pre_hook_ip = pre_hook_ip_node[InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]
                assert pre_hook_ip.target_type == TargetType.OPERATOR_PRE_HOOK
                assert pre_hook_ip.ia_op_exec_context == ref_op_exec_context.input_agnostic

    def test_operator_metatype_marking(self):
        from nncf.dynamic_graph.operator_metatypes import Conv2dMetatype, BatchNormMetatype, RELUMetatype, \
            MaxPool2dMetatype, \
            ConvTranspose2dMetatype, DepthwiseConv2dSubtype, AddMetatype, AvgPool2dMetatype, LinearMetatype
        ref_scope_vs_metatype_dict = {
            "/" + MODEL_INPUT_OP_NAME + "_0": NoopMetatype,
            "ModelForMetatypeTesting/NNCFConv2d[conv_regular]/conv2d_0": Conv2dMetatype,
            "ModelForMetatypeTesting/BatchNorm2d[bn]/batch_norm_0": BatchNormMetatype,
            "ModelForMetatypeTesting/RELU_0": RELUMetatype,
            "ModelForMetatypeTesting/MaxPool2d[max_pool2d]/max_pool2d_0": MaxPool2dMetatype,
            "ModelForMetatypeTesting/NNCFConvTranspose2d[conv_transpose]/conv_transpose2d_0": ConvTranspose2dMetatype,
            "ModelForMetatypeTesting/NNCFConv2d[conv_depthwise]/conv2d_0": DepthwiseConv2dSubtype,
            "ModelForMetatypeTesting/__iadd___0": AddMetatype,
            "ModelForMetatypeTesting/AdaptiveAvgPool2d[adaptive_avg_pool]/adaptive_avg_pool2d_0": AvgPool2dMetatype,
            "ModelForMetatypeTesting/NNCFLinear[linear]/linear_0": LinearMetatype
        }

        class ModelForMetatypeTesting(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_regular = torch.nn.Conv2d(in_channels=3,
                                                    out_channels=16,
                                                    kernel_size=3)
                self.bn = torch.nn.BatchNorm2d(num_features=16)
                self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2)
                self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=16,
                                                               out_channels=8,
                                                               kernel_size=3)
                self.conv_depthwise = torch.nn.Conv2d(in_channels=8, out_channels=8,
                                                      kernel_size=5, groups=8)
                self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
                self.linear = torch.nn.Linear(in_features=8, out_features=1)

            def forward(self, input_):
                x = self.conv_regular(input_)
                x = self.bn(x)
                x = torch.nn.functional.relu(x)
                x.transpose_(2, 3)
                x = self.max_pool2d(x)
                x = self.conv_transpose(x)
                x = self.conv_depthwise(x)
                x += torch.ones_like(x)
                x = self.adaptive_avg_pool(x)
                x = self.linear(x.flatten())
                return x

        model = ModelForMetatypeTesting()
        nncf_network = NNCFNetwork(model, [ModelInputInfo([1, 3, 300, 300])])
        ip_graph = nncf_network.get_insertion_point_graph()

        for node in ip_graph.nodes().values():
            if node[InsertionPointGraph.NODE_TYPE_NODE_ATTR] == InsertionPointGraphNodeType.OPERATOR:
                nncf_node_ref = node[InsertionPointGraph.REGULAR_NODE_REF_NODE_ATTR]
                scope_str = str(nncf_node_ref[PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR].input_agnostic)
                assert scope_str in ref_scope_vs_metatype_dict
                ref_metatype = ref_scope_vs_metatype_dict[scope_str]
                assert node[InsertionPointGraph.OPERATOR_METATYPE_NODE_ATTR] == ref_metatype

    @pytest.mark.parametrize(("mock_graph_factory", "dot_file_name"),
                             MERGE_PATTERN_TEST_CASES,
                             ids=[x[1] for x in MERGE_PATTERN_TEST_CASES])
    def test_get_ip_graph_with_merged_operations(self, mock_graph_factory, dot_file_name):
        mock_graph = mock_graph_factory()
        ip_graph = InsertionPointGraph(mock_graph)
        merged_ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations()

        data_dir = TEST_ROOT / 'data/reference_graphs/pattern_merging'  # type: Path

        path_to_dot_file = data_dir / '{}.dot'.format(dot_file_name)

        # validate .dot file manually!
        if not path_to_dot_file.exists():
            if not data_dir.exists():
                data_dir.mkdir(parents=True)
            nx.drawing.nx_pydot.write_dot(merged_ip_graph, str(path_to_dot_file))

        load_graph = nx.drawing.nx_pydot.read_dot(str(path_to_dot_file))

        for key in load_graph.nodes.keys():
            key.replace(r'\\n', r'\n')  # Somehow pydot mangles the \n characters while writing a .dot file

        sanitized_loaded_keys = [key.replace('\\n', '\n') for key in load_graph.nodes.keys()]
        sanitized_loaded_edges = [(u.replace('\\n', '\n'),
                                   v.replace('\\n', '\n')) for u, v in nx.DiGraph(load_graph).edges]

        assert Counter(sanitized_loaded_keys) == Counter(list(merged_ip_graph.nodes.keys()))
        assert Counter(sanitized_loaded_edges) == Counter(list(merged_ip_graph.edges))


def test_can_collect_scopes_of_train_only_modules():
    model = ManyNonEvalModules()
    graph_builder = GraphBuilder(custom_forward_fn=lambda model_: model_(torch.randn([1, 1, 1, 1])))
    actual_scopes = NNCFNetwork.collect_eval_only_ops_exec_context(model, graph_builder)
    ref_scopes = {
        'ManyNonEvalModules/AvgPool2d[avg_pool]/avg_pool2d_0',
        'ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/Dropout/dropout_0',
        'ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/Dropout/dropout_1',
        'ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/Linear[called_linear]/linear_0',
        'ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/CustomWeightModule[custom]/linear_0'
    }
    assert set(actual_scopes) == ref_scopes


def test_get_clean_shallow_copy():
    model = TwoConvTestModelWithUserModule()
    config = get_basic_sparsity_plus_quantization_config()
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    assert sparse_quantized_model.activation_quantizers
    old_nncf_modules = sparse_quantized_model.get_nncf_modules().values()
    old_nncf_module_pre_ops = [module.pre_ops for module in old_nncf_modules]
    assert any(old_nncf_module_pre_ops)
    assert sparse_quantized_model.get_graph().get_nodes_count() != \
           sparse_quantized_model.get_original_graph().get_nodes_count()

    clean_copy = sparse_quantized_model.get_clean_shallow_copy()
    assert clean_copy is not sparse_quantized_model
    assert clean_copy.get_nncf_wrapped_model() is sparse_quantized_model.get_nncf_wrapped_model()
    new_nncf_modules = clean_copy.get_nncf_modules().values()
    new_nncf_module_pre_ops = [module.pre_ops for module in new_nncf_modules]
    assert not any(new_nncf_module_pre_ops)
    assert clean_copy.get_graph().get_nodes_count() == clean_copy.get_original_graph().get_nodes_count()


def test_temporary_clean_view():
    model = TwoConvTestModelWithUserModule()
    config = get_basic_sparsity_plus_quantization_config()
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    old_sd = sparse_quantized_model.state_dict()
    old_graph = deepcopy(sparse_quantized_model.get_graph())
    with sparse_quantized_model.temporary_clean_view() as intermediate_model:
        clean_sd = intermediate_model.state_dict()
        assert len(clean_sd) < len(old_sd)
        new_nncf_modules = intermediate_model.get_nncf_modules().values()
        new_nncf_module_pre_ops = [module.pre_ops for module in new_nncf_modules]
        assert not any(new_nncf_module_pre_ops)
        assert intermediate_model.get_graph().get_nodes_count() == \
               intermediate_model.get_original_graph().get_nodes_count()
    sd_after_tmp_clean_view = sparse_quantized_model.state_dict()
    for key in old_sd.keys():
        assert key in sd_after_tmp_clean_view
        assert torch.all(torch.eq(sd_after_tmp_clean_view[key], old_sd[key]))
    sparse_quantized_model.rebuild_graph()
    graph_after_tmp_clean_view = sparse_quantized_model.get_graph()
    assert graph_after_tmp_clean_view == old_graph


class TestModelMultipleForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, 1)
        self.conv1 = nn.Conv2d(1, 1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        x1 = self.conv1(x)
        x2 = self.conv1(x)
        return x1, x2


def test_multiple_forward():
    # Check that all convolution nodes in model have op_exec_context and module_attributes
    # for case with multiple forward of one module
    model = TestModelMultipleForward()
    config = get_basic_sparsity_plus_quantization_config()
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    graph = sparse_quantized_model.get_original_graph()
    for node_key in list(graph.get_all_node_keys())[1:]:
        node = graph.get_nx_node_by_key(node_key)
        assert node.get(PTNNCFGraph.OP_EXEC_CONTEXT_NODE_ATTR)
        assert node.get(PTNNCFGraph.MODULE_ATTRIBUTES)
