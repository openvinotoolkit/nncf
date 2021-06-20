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
import os
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import List
from typing import Tuple

import networkx as nx
import pytest
import torch

from nncf.torch.graph.operator_metatypes import NoopMetatype
from nncf.torch.graph.operator_metatypes import ReshapeMetatype
from torch import nn

from nncf.torch import register_module
from nncf.common.graph import MODEL_INPUT_OP_NAME
from nncf.common.graph import MODEL_OUTPUT_OP_NAME
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.transformations.commands import TargetType
from nncf.torch.dynamic_graph.context import PreHookId
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.scope import Scope
from nncf.common.graph import NNCFGraph
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.graph.operator_metatypes import InputNoopMetatype
from nncf.torch.graph.operator_metatypes import OutputNoopMetatype
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.layer_utils import _NNCFModuleMixin
from nncf.torch.module_operations import BaseOp
from nncf.torch.nncf_network import EXTERNAL_QUANTIZERS_STORAGE_NAME
from nncf.torch.nncf_network import InsertionPointGraph
from nncf.torch.nncf_network import InsertionPointGraphNodeType
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTInsertionPoint
from nncf.torch.nncf_network import PTInsertionType
from nncf.torch.nncf_network import PTModelTransformer
from nncf.torch.quantization.node_matcher import PTOperatorMetatypeNodeMatcher
from tests.torch.composite.test_sparsity_quantization import get_basic_sparsity_plus_quantization_config
from tests.common.helpers import TEST_ROOT
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.test_models.synthetic import ManyNonEvalModules


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
    graph_1 = deepcopy(qnet_no_shape.get_dynamic_graph())

    _ = qnet_no_shape(torch.zeros(*input_shape_2))
    graph_2 = deepcopy(qnet_no_shape.get_dynamic_graph())

    assert graph_1 == graph_2

    nodes_1 = list(graph_1.get_all_nodes())
    assert len(nodes_1) == 3  # 1 input node + 1 operation node + 1 output node

    qnet = NNCFNetwork(model, input_infos=[ModelInputInfo(input_shape_1), ])  # type: NNCFNetwork
    _ = qnet(torch.zeros(*input_shape_1))
    _ = qnet(torch.zeros(*input_shape_2))
    # The second forward run should have led to an increase in registered node counts
    # since disable_shape_matching was False and the network was run with a different
    # shape of input tensor
    assert qnet.get_dynamic_graph().get_nodes_count() > graph_1.get_nodes_count()


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

    from nncf.torch.layers import UNWRAPPED_USER_MODULES
    assert ModuleOfUser in UNWRAPPED_USER_MODULES.registry_dict.values()

    # pylint: disable=protected-access
    assert isinstance(nncf_model.user_module, ModuleOfUser)
    assert isinstance(nncf_model.user_module, _NNCFModuleMixin)
    assert type(nncf_model.user_module).__name__ == "NNCFUserModuleOfUser"

    user_module_attrs = dir(nncf_model.user_module)
    for attr in dir(_NNCFModuleMixin):
        assert attr in user_module_attrs


# pylint: disable=protected-access
def test_get_op_nodes_in_scope():
    model = TwoConvTestModel()
    nncf_model = NNCFNetwork(deepcopy(model), input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork
    nncf_graph = nncf_model.get_original_graph()

    # Valid scopes should be successfully found
    valid_nncf_modules = nncf_model.get_nncf_modules()
    nodes_list = list(nncf_graph.get_all_node_ids())
    for module_scope, _ in valid_nncf_modules.items():
        matching_nncf_nodes = nncf_graph.get_op_nodes_in_scope(module_scope)
        assert len(matching_nncf_nodes) == 1
        node = matching_nncf_nodes[0]
        assert isinstance(node, NNCFNode)
        assert node.node_id in nodes_list

    fake_model = BasicConvTestModel()
    fake_nncf_model = NNCFNetwork(deepcopy(fake_model), input_infos=[ModelInputInfo([1, 1, 4, 4])])

    # Not valid scopes shouldn't be found
    fake_nncf_modules = fake_nncf_model.get_nncf_modules()
    for module_scope, _ in fake_nncf_modules.items():
        matching_nncf_nodes = nncf_graph.get_op_nodes_in_scope(module_scope)
        assert not matching_nncf_nodes


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

    conv1_node_name = 'InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0'
    point_for_conv1_weights = PTTargetPoint(target_type=TargetType.OPERATION_WITH_WEIGHTS,
                                            target_node_name=conv1_node_name)
    point_for_conv1_inputs = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK,
                                           target_node_name=conv1_node_name)
    point_for_conv1_activations = PTTargetPoint(target_type=TargetType.POST_LAYER_OPERATION,
                                                target_node_name=conv1_node_name)

    conv2_node_name = 'InsertionPointTestModel/NNCFConv2d[conv2]/conv2d_0'
    point_for_conv2_weights = PTTargetPoint(target_type=TargetType.OPERATION_WITH_WEIGHTS,
                                            target_node_name=conv2_node_name)
    point_for_conv2_inputs = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK,
                                           target_node_name=conv2_node_name)
    point_for_conv2_activations = PTTargetPoint(target_type=TargetType.POST_LAYER_OPERATION,
                                                target_node_name=conv2_node_name)

    linear_node_name = 'InsertionPointTestModel/linear_0'
    point_for_linear_weight_input = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK,
                                                  target_node_name=linear_node_name, input_port_id=0)
    point_for_linear_activation = PTTargetPoint(target_type=TargetType.OPERATOR_POST_HOOK,
                                                target_node_name=linear_node_name)

    relu_node_name = 'InsertionPointTestModel/ReLU[relu]/relu_0'
    point_for_relu_inputs = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK,
                                          target_node_name=relu_node_name, input_port_id=0)
    point_for_relu_activations = PTTargetPoint(target_type=TargetType.OPERATOR_POST_HOOK,
                                               target_node_name=relu_node_name)

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
    def test_single_insertions(self, setup, target_point: PTTargetPoint):
        insertion_point = PTInsertionPoint(target_point.target_type,
                                           OperationAddress.from_str(target_point.target_node_name),
                                           target_point.input_port_id)
        if insertion_point.insertion_type in [PTInsertionType.OPERATOR_PRE_HOOK, PTInsertionType.OPERATOR_POST_HOOK]:
            hook = lambda x: x
        else:
            hook = BaseOp(lambda x: x)

        self.compressed_model.insert_at_point(insertion_point, [hook])

        # pylint:disable=protected-access
        if insertion_point.insertion_type == PTInsertionType.OPERATOR_PRE_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            pre_hook_id = PreHookId(insertion_point.op_address, input_port_id=insertion_point.input_port_id)
            assert ctx._pre_hooks[pre_hook_id][0] is hook
        if insertion_point.insertion_type == PTInsertionType.OPERATOR_POST_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            assert ctx._post_hooks[insertion_point.op_address][0] is hook
        if insertion_point.insertion_type == PTInsertionType.NNCF_MODULE_PRE_OP:
            module = self.compressed_model.get_module_by_scope(insertion_point.module_scope)
            assert module.pre_ops["0"] is hook

        if insertion_point.insertion_type == PTInsertionType.NNCF_MODULE_POST_OP:
            module = self.compressed_model.get_module_by_scope(insertion_point.module_scope)
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
        self.compressed_model = PTModelTransformer(self.compressed_model).transform(layout)

        hook_list = [hook1, hook2, hook3]

        if priority_type == "same":
            order = [0, 1, 2]
        elif priority_type == "different":
            order = [2, 0, 1]

        # pylint:disable=protected-access
        if insertion_type == TargetType.OPERATOR_PRE_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            pre_hook_id = PreHookId(OperationAddress.from_str(point.target_node_name),
                                    input_port_id=point.input_port_id)
            self.check_order(ctx._pre_hooks[pre_hook_id], hook_list, order)
        if insertion_type == TargetType.OPERATOR_POST_HOOK:
            ctx = self.compressed_model.get_tracing_context()
            self.check_order(ctx._post_hooks[OperationAddress.from_str(point.target_node_name)],
                             hook_list, order)

        if insertion_type == TargetType.OPERATION_WITH_WEIGHTS:
            module = self.compressed_model.get_containing_module(point.target_node_name)
            # Works because Pytorch ModuleDict is ordered
            self.check_order([x.operand for x in module.pre_ops.values()], hook_list, order)

        if insertion_type == TargetType.POST_LAYER_OPERATION:
            module = self.compressed_model.get_containing_module(point.target_node_name)
            # Works because Pytorch ModuleDict is ordered
            self.check_order(list(module.post_ops.values()), hook_list, order)


def mark_input_ports_lexicographically_based_on_input_node_key(graph: nx.DiGraph):
    for node_key in graph.nodes:
        input_edges = graph.in_edges(node_key)
        sorted_input_edges = sorted(input_edges, key=lambda x: x[0])
        for idx, edge in enumerate(sorted_input_edges):
            graph.edges[edge][PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR] = idx


def get_nncf_graph_from_mock_nx_graph(nx_graph: nx.DiGraph) -> PTNNCFGraph:
    mock_graph = PTNNCFGraph()
    key_vs_id = {}
    edge_vs_output_idx_and_creator_id = {}  # type: Dict[Tuple[str, str], Tuple[int, int]]
    from networkx.algorithms.dag import lexicographical_topological_sort
    for idx, curr_node_key in enumerate(lexicographical_topological_sort(nx_graph)):
        node = nx_graph.nodes[curr_node_key]
        if NNCFGraph.NODE_NAME_ATTR in node:
            node_name = node[NNCFGraph.NODE_NAME_ATTR]
        else:
            node_name = str(OperationAddress(curr_node_key, Scope(), 0))

        if NNCFGraph.NODE_TYPE_ATTR in node:
            node_type = node[NNCFGraph.NODE_TYPE_ATTR]
        else:
            node_type = curr_node_key
        layer_attributes = node.get(NNCFGraph.LAYER_ATTRIBUTES)
        node_id = idx
        node = mock_graph.add_nncf_node(
            node_name=node_name,
            node_type=node_type,
            node_metatype=NoopMetatype,
            layer_attributes=layer_attributes,
            node_id_override=idx)
        key_vs_id[curr_node_key] = node_id

        preds = list(nx_graph.predecessors(curr_node_key))
        for pred_idx, pred in enumerate(preds):
            in_edge = (pred, curr_node_key)
            _, creator_id = edge_vs_output_idx_and_creator_id[in_edge]
            mock_graph.add_edge_between_nncf_nodes(creator_id, node_id,
                                                   [1, 1, 1, 1], pred_idx,
                                                   dtype=Dtype.FLOAT)

        for out_idx, out_edge in enumerate(nx_graph.out_edges(curr_node_key)):
            edge_vs_output_idx_and_creator_id[out_edge] = (out_idx, node.node_id)
    return mock_graph


def get_two_branch_mock_model_graph() -> PTNNCFGraph:
    mock_nx_graph = nx.DiGraph()

    #   (0 /A)
    #      |
    #   (1 /B)
    #   /     \
    # (2 /C) (3 /D)
    #  |       |
    # (4 /E)   |
    #   \     /
    #   (5 /F)
    #     |
    #   (6 /G)
    #     |
    #   (7 /H)

    node_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    for node_key in node_keys:
        mock_nx_graph.add_node(node_key)

    mock_nx_graph.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D'), ('C', 'E'), ('E', 'F'),
                                  ('D', 'F'), ('F', 'G'), ('G', 'H')])

    mark_input_ports_lexicographically_based_on_input_node_key(mock_nx_graph)
    return get_nncf_graph_from_mock_nx_graph(mock_nx_graph)


MOCK_OPERATOR_NAME = "conv_transpose2d"


def get_mock_nncf_node_attrs(op_name=None, scope_str=None):
    op_name_to_set = op_name if op_name is not None else MOCK_OPERATOR_NAME
    scope_to_set = Scope() if scope_str is None else Scope.from_str(scope_str)
    return {
        NNCFGraph.NODE_NAME_ATTR: str(OperationAddress(op_name_to_set, scope_to_set, 0)),
        NNCFGraph.NODE_TYPE_ATTR: op_name_to_set
    }


def get_mock_model_graph_with_mergeable_pattern() -> NNCFGraph:
    mock_nx_graph = nx.DiGraph()

    #   (A)
    #    |
    #  (conv2d)
    #    |
    # (batch_norm)
    #    |
    #  (RELU)
    #    |
    #   (B)

    node_keys = ['conv2d', 'batch_norm', 'relu', 'A', 'B']
    for node_key in node_keys:
        mock_nx_graph.add_node(node_key, **get_mock_nncf_node_attrs(op_name=node_key))

    mock_nx_graph.add_edges_from([('A', 'conv2d', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('conv2d', 'batch_norm', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('batch_norm', 'relu', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('relu', 'B', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0})])
    return get_nncf_graph_from_mock_nx_graph(mock_nx_graph)


def get_mock_model_graph_with_no_mergeable_pattern() -> NNCFGraph:
    mock_nx_graph = nx.DiGraph()

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

    node_keys = ['conv2d', 'batch_norm', 'relu', 'A', 'B', 'C', 'D']
    for node_key in node_keys:
        mock_nx_graph.add_node(node_key, **get_mock_nncf_node_attrs(op_name=node_key))

    mock_nx_graph.add_edges_from([('A', 'conv2d', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('conv2d', 'C', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('C', 'batch_norm', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('batch_norm', 'D', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('D', 'relu', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('relu', 'B', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0})])
    return get_nncf_graph_from_mock_nx_graph(mock_nx_graph)


def get_mock_model_graph_with_broken_output_edge_pattern() -> NNCFGraph:
    mock_nx_graph = nx.DiGraph()

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

    node_keys = ['conv2d', 'batch_norm', 'relu', 'A', 'B', 'C']
    for node_key in node_keys:
        mock_nx_graph.add_node(node_key, **get_mock_nncf_node_attrs(op_name=node_key))

    mock_nx_graph.add_edges_from([('A', 'conv2d', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('conv2d', 'batch_norm', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('conv2d', 'C', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 1}),
                               ('batch_norm', 'relu', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('relu', 'C', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0}),
                               ('C', 'B', {PTNNCFGraph.IN_PORT_NAME_EDGE_ATTR: 0})])
    return get_nncf_graph_from_mock_nx_graph(mock_nx_graph)


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

        nx_graph = mock_graph.get_nx_graph_copy()
        ref_node_len = 3 * len(nx_graph.nodes)  # 2 additional nodes per each operator node
        ref_edge_len = 3 * len(nx_graph.edges)

        assert len(ip_graph.nodes) == ref_node_len
        assert len(ip_graph.edges) == ref_edge_len

        for nncf_node_idx in mock_graph.get_all_node_ids():
            node_key = mock_graph.get_node_key_by_id(nncf_node_idx)
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
            original_neighbours = nx_graph.neighbors(node_key)
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

        mock_graph.add_node('bar')
        mock_graph.add_node('baz')
        mock_graph.add_edge('bar', 'baz')
        nncf_graph = get_nncf_graph_from_mock_nx_graph(mock_graph)

        ip_graph = InsertionPointGraph(nncf_graph)

        for nncf_node in nncf_graph.get_all_nodes():
            node_id = nncf_node.node_id
            node_key = nncf_graph.get_node_key_by_id(node_id)
            preds = list(ip_graph.predecessors(node_key))
            succs = list(ip_graph.successors(node_key))

            post_hook_ip_node = ip_graph.nodes[succs[0]]
            post_hook_ip = post_hook_ip_node[InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]
            assert post_hook_ip.target_type == TargetType.OPERATOR_POST_HOOK
            assert post_hook_ip.target_node_name == nncf_node.node_name

            for pre_hook_ip_node_key in preds:
                pre_hook_ip_node = ip_graph.nodes[pre_hook_ip_node_key]
                pre_hook_ip = pre_hook_ip_node[InsertionPointGraph.INSERTION_POINT_DATA_NODE_ATTR]
                assert pre_hook_ip.target_type == TargetType.OPERATOR_PRE_HOOK
                assert pre_hook_ip.target_node_name == nncf_node.node_name

    def test_operator_metatype_marking(self):
        from nncf.torch.graph.operator_metatypes import Conv2dMetatype, BatchNormMetatype, RELUMetatype, \
            MaxPool2dMetatype, \
            ConvTranspose2dMetatype, DepthwiseConv2dSubtype, AddMetatype, AvgPool2dMetatype, LinearMetatype
        ref_scope_vs_metatype_dict = {
            "/" + MODEL_INPUT_OP_NAME + "_0": InputNoopMetatype,
            "ModelForMetatypeTesting/NNCFConv2d[conv_regular]/conv2d_0": Conv2dMetatype,
            "ModelForMetatypeTesting/BatchNorm2d[bn]/batch_norm_0": BatchNormMetatype,
            "ModelForMetatypeTesting/relu_0": RELUMetatype,
            "ModelForMetatypeTesting/MaxPool2d[max_pool2d]/max_pool2d_0": MaxPool2dMetatype,
            "ModelForMetatypeTesting/NNCFConvTranspose2d[conv_transpose]/conv_transpose2d_0": ConvTranspose2dMetatype,
            "ModelForMetatypeTesting/NNCFConv2d[conv_depthwise]/conv2d_0": DepthwiseConv2dSubtype,
            "ModelForMetatypeTesting/__iadd___0": AddMetatype,
            "ModelForMetatypeTesting/AdaptiveAvgPool2d[adaptive_avg_pool]/adaptive_avg_pool2d_0": AvgPool2dMetatype,
            "ModelForMetatypeTesting/NNCFLinear[linear]/linear_0": LinearMetatype,
            'ModelForMetatypeTesting/flatten_0': ReshapeMetatype,
            "/" + MODEL_OUTPUT_OP_NAME + "_0": OutputNoopMetatype,
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
        nncf_graph = nncf_network.get_original_graph()

        for nncf_node in nncf_graph.get_all_nodes():  # type: NNCFNode
            assert nncf_node.node_name in ref_scope_vs_metatype_dict
            ref_metatype = ref_scope_vs_metatype_dict[nncf_node.node_name]
            assert PTOperatorMetatypeNodeMatcher.match(nncf_node) == ref_metatype

    @pytest.mark.parametrize(("mock_graph_factory", "dot_file_name"),
                             MERGE_PATTERN_TEST_CASES,
                             ids=[x[1] for x in MERGE_PATTERN_TEST_CASES])
    def test_get_ip_graph_with_merged_operations(self, mock_graph_factory, dot_file_name):
        mock_graph = mock_graph_factory()
        ip_graph = InsertionPointGraph(mock_graph)
        merged_ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations()

        data_dir = TEST_ROOT / 'torch/data/reference_graphs/pattern_merging'  # type: Path

        path_to_dot_file = data_dir / '{}.dot'.format(dot_file_name)

        if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
            if not os.path.exists(str(data_dir)):
                os.makedirs(str(data_dir))
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
    graph = graph_builder.build_graph(model, as_eval=True)
    actual_scopes = [n.node_name for n in graph.get_all_nodes()]
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
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    external_quantizers = getattr(sparse_quantized_model, EXTERNAL_QUANTIZERS_STORAGE_NAME)
    assert external_quantizers
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
    register_bn_adaptation_init_args(config)
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
    # Check that all convolution nodes in model have op_address and layer_attributes
    # for case with multiple forward of one module
    model = TestModelMultipleForward()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    graph = sparse_quantized_model.get_original_graph()
    for node in list(graph.get_all_nodes())[1:-2]:
        assert node.layer_attributes is not None
