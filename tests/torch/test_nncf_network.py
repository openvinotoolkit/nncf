"""
 Copyright (c) 2019-2023 Intel Corporation
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
import inspect
import itertools
import os
from abc import ABCMeta
from abc import abstractmethod
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import List

import networkx as nx
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm

from nncf import nncf_logger
from nncf.common.graph import NNCFNode
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.operator_metatypes import UnknownMetatype
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.patterns.manager import TargetDevice
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.insertion_point_graph import InsertionPointGraphNodeType
from nncf.common.insertion_point_graph import PostHookInsertionPoint
from nncf.common.insertion_point_graph import PreHookInsertionPoint
from nncf.common.logging.logger import NNCFDeprecationWarning
from nncf.common.utils.backend import BackendType
from nncf.common.utils.dot_file_rw import get_graph_without_data
from nncf.common.utils.dot_file_rw import read_dot_graph
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.torch import register_module
from nncf.torch.dynamic_graph.context import PreHookId
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.scope import Scope
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph_builder import GraphBuilder
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTInputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTOutputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.layer_utils import _NNCFModuleMixin
from nncf.torch.layers import NNCFConv2d
from nncf.torch.module_operations import BaseOp
from nncf.torch.nncf_network import EXTERNAL_QUANTIZERS_STORAGE_NAME
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTInsertionPoint
from nncf.torch.nncf_network import PTInsertionType
from nncf.torch.nncf_network import PTModelTransformer
from tests.common.quantization.mock_graphs import get_ip_graph_for_test
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_broken_output_edge_pattern
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_mergeable_pattern
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_no_mergeable_pattern
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph
from tests.common.quantization.mock_graphs import get_two_branch_mock_model_graph
from tests.shared.paths import TEST_ROOT
from tests.torch.composite.test_sparsity_quantization import get_basic_sparsity_plus_quantization_config
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import check_correct_nncf_modules_replacement
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.test_models.synthetic import ManyNonEvalModules

# pylint:disable=too-many-lines


@pytest.fixture()
def _nncf_caplog(caplog):
    nncf_logger.propagate = True
    yield caplog
    nncf_logger.propagate = False


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

    qnet_no_shape = NNCFNetwork(
        deepcopy(model),
        input_infos=[
            ModelInputInfo(input_shape_1),
        ],
        scopes_without_shape_matching=["MatMulModel"],
    )  # type: NNCFNetwork

    context = qnet_no_shape.nncf.get_tracing_context()
    context.enable_trace_dynamic_graph()
    _ = qnet_no_shape(torch.zeros(*input_shape_1))
    graph_1 = deepcopy(qnet_no_shape.nncf.get_dynamic_graph())

    _ = qnet_no_shape(torch.zeros(*input_shape_2))
    graph_2 = deepcopy(qnet_no_shape.nncf.get_dynamic_graph())

    assert graph_1 == graph_2

    nodes_1 = list(graph_1.get_all_nodes())
    assert len(nodes_1) == 5  # 1 input node + 1 chunk + 1 transpose + 1 matmul + 1 output node

    qnet = NNCFNetwork(
        model,
        input_infos=[
            ModelInputInfo(input_shape_1),
        ],
    )  # type: NNCFNetwork
    context = qnet.nncf.get_tracing_context()
    context.enable_trace_dynamic_graph()
    _ = qnet(torch.zeros(*input_shape_1))
    _ = qnet(torch.zeros(*input_shape_2))
    # The second forward run should have led to an increase in registered node counts
    # since disable_shape_matching was False and the network was run with a different
    # shape of input tensor
    assert qnet.nncf.get_dynamic_graph().get_nodes_count() > graph_1.get_nodes_count()


def test_check_correct_modules_replacement():
    model = TwoConvTestModel()
    nncf_model = NNCFNetwork(TwoConvTestModel(), input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork

    _, detected_nncf_modules = check_correct_nncf_modules_replacement(model, nncf_model)
    replaced_modules_reported_by_nncf_network = {
        scope: module for module, scope in nncf_model.nncf.get_nncf_modules().items()
    }
    assert set(detected_nncf_modules) == set(replaced_modules_reported_by_nncf_network)


class WeightNormedConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = weight_norm(torch.nn.Conv1d(1, 1, 1))

    def forward(self, x):
        return self.conv(x)


def test_weight_normed_modules_are_replaced_correctly():
    nncf_model = NNCFNetwork(WeightNormedConvModel(), input_infos=[ModelInputInfo([1, 1, 10])])

    wrapped_conv = nncf_model.conv
    assert hasattr(wrapped_conv, "weight_g")
    assert hasattr(wrapped_conv, "weight_v")
    assert hasattr(wrapped_conv, "weight")

    assert isinstance(wrapped_conv.weight_g, torch.nn.Parameter)
    assert isinstance(wrapped_conv.weight_v, torch.nn.Parameter)
    assert not isinstance(wrapped_conv.weight, torch.nn.Parameter)

    # pylint:disable=protected-access
    assert len(wrapped_conv._forward_pre_hooks) == 1


class UnregisteredUserModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones([1]))
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, input_):
        x = input_ * self.weight
        x += torch.rand_like(x)
        x = F.conv2d(x, self.conv.weight)
        x = self.conv(x)
        return x


@register_module()
class RegisteredUserModule(UnregisteredUserModule):
    pass


class TwoConvTestModelWithUserModule(TwoConvTestModel):
    def __init__(self):
        super().__init__()
        self.unregistered_user_module = UnregisteredUserModule()
        self.registered_user_module = RegisteredUserModule()

    def forward(self, x):
        x = super().forward(x)
        x = self.unregistered_user_module(x)
        x = self.registered_user_module(x)
        return x


def test_custom_module_registering():
    model = TwoConvTestModelWithUserModule()
    nncf_model = NNCFNetwork(model, input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork

    from nncf.torch.layers import UNWRAPPED_USER_MODULES

    assert RegisteredUserModule in UNWRAPPED_USER_MODULES.registry_dict.values()
    assert UnregisteredUserModule not in UNWRAPPED_USER_MODULES.registry_dict.values()

    # pylint: disable=protected-access
    modules = [nncf_model.registered_user_module, nncf_model.unregistered_user_module.conv]
    base_modules = [RegisteredUserModule, torch.nn.Conv2d]
    names = ["NNCFUserRegisteredUserModule", "NNCFConv2d"]
    for module, base_module, name in zip(modules, base_modules, names):
        assert isinstance(module, base_module)
        assert isinstance(module, _NNCFModuleMixin)
        assert type(module).__name__ == name

        module_attrs = dir(module)
        for attr in dir(_NNCFModuleMixin):
            assert attr in module_attrs

    # Check user ops metatypes
    graph = nncf_model.nncf.get_original_graph()
    nodes_dict = {
        "TwoConvTestModelWithUserModule/UnregisteredUserModule[unregistered_user_module]/rand_like_0": UnknownMetatype,
        "TwoConvTestModelWithUserModule/UnregisteredUserModule[unregistered_user_module]/conv2d_0": PTConv2dMetatype,
        "TwoConvTestModelWithUserModule/UnregisteredUserModule[unregistered_user_module]/NNCFConv2d[conv]/conv2d_0": PTModuleConv2dMetatype,
        "TwoConvTestModelWithUserModule/NNCFUserRegisteredUserModule[registered_user_module]/rand_like_0": UnknownMetatype,
        "TwoConvTestModelWithUserModule/NNCFUserRegisteredUserModule[registered_user_module]/conv2d_0": PTModuleConv2dMetatype,
        "TwoConvTestModelWithUserModule/NNCFUserRegisteredUserModule[registered_user_module]/Conv2d[conv]/conv2d_0": PTConv2dMetatype,
    }
    for node_name, ref_metatype in nodes_dict.items():
        assert graph.get_node_by_name(node_name).metatype is ref_metatype


def test_get_weighted_original_graph_nodes():
    model = TwoConvTestModelWithUserModule()
    nncf_model = NNCFNetwork(model, input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork
    weighted_nodes = nncf_model.nncf.get_weighted_original_graph_nodes()
    ref_node_names = [
        "TwoConvTestModelWithUserModule/Sequential[features]/Sequential[0]/NNCFConv2d[0]/conv2d_0",
        "TwoConvTestModelWithUserModule/Sequential[features]/Sequential[1]/NNCFConv2d[0]/conv2d_0",
        "TwoConvTestModelWithUserModule/UnregisteredUserModule[unregistered_user_module]/NNCFConv2d[conv]/conv2d_0",
        # The next one matched because it's a free op with metatypes corresponding to weighted ops
        # and is within a registered user module
        "TwoConvTestModelWithUserModule/NNCFUserRegisteredUserModule[registered_user_module]/conv2d_0",
    ]
    ref_weighted_nodes = [nncf_model.nncf.get_original_graph().get_node_by_name(name) for name in ref_node_names]
    assert set(weighted_nodes) == set(ref_weighted_nodes)


# pylint: disable=protected-access
def test_get_op_nodes_in_scope():
    model = TwoConvTestModel()
    nncf_model = NNCFNetwork(deepcopy(model), input_infos=[ModelInputInfo([1, 1, 4, 4])])  # type: NNCFNetwork
    nncf_graph = nncf_model.nncf.get_original_graph()

    # Valid scopes should be successfully found
    valid_nncf_modules = nncf_model.nncf.get_nncf_modules()
    nodes_list = list(nncf_graph.get_all_node_ids())
    for module_scope in valid_nncf_modules.values():
        matching_nncf_nodes = nncf_graph.get_op_nodes_in_scope(module_scope)
        assert len(matching_nncf_nodes) == 1
        node = matching_nncf_nodes[0]
        assert isinstance(node, NNCFNode)
        assert node.node_id in nodes_list

    fake_model = BasicConvTestModel()
    fake_nncf_model = NNCFNetwork(deepcopy(fake_model), input_infos=[ModelInputInfo([1, 1, 4, 4])])

    # Not valid scopes shouldn't be found
    fake_nncf_modules = fake_nncf_model.nncf.get_nncf_modules()
    for module_scope in fake_nncf_modules.values():
        matching_nncf_nodes = nncf_graph.get_op_nodes_in_scope(module_scope)
        assert not matching_nncf_nodes


def test_nncf_node_attrs_are_consistent():
    # Check that node returned from `add_nncf_node`
    # refer to the save `data` dict as node returned by
    # `get_node_by_id` and `get_op_nodes_in_scope`
    nncf_graph = PTNNCFGraph()
    new_node = nncf_graph.add_nncf_node(
        node_name="dummy", node_type="dummy", layer_name="dummy", node_metatype=UnknownMetatype
    )
    new_node_saved = nncf_graph.get_node_by_id(new_node.node_id)
    assert new_node.data is new_node_saved.data
    nodes_in_scope = nncf_graph.get_op_nodes_in_scope(nncf_graph.get_scope_by_node_name("dummy"))
    assert new_node.data is nodes_in_scope[0].data


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
        self.compressed_model = NNCFNetwork(
            InsertionPointTestModel(), [ModelInputInfo([1, 1, 10, 10])]
        )  # type: NNCFNetwork

    conv1_node_name = "InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0"
    point_for_conv1_weights = PTTargetPoint(
        target_type=TargetType.OPERATION_WITH_WEIGHTS, target_node_name=conv1_node_name
    )
    point_for_conv1_inputs = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK, target_node_name=conv1_node_name)
    point_for_conv1_activations = PTTargetPoint(
        target_type=TargetType.POST_LAYER_OPERATION, target_node_name=conv1_node_name
    )

    conv2_node_name = "InsertionPointTestModel/NNCFConv2d[conv2]/conv2d_0"
    point_for_conv2_weights = PTTargetPoint(
        target_type=TargetType.OPERATION_WITH_WEIGHTS, target_node_name=conv2_node_name
    )
    point_for_conv2_inputs = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK, target_node_name=conv2_node_name)
    point_for_conv2_activations = PTTargetPoint(
        target_type=TargetType.POST_LAYER_OPERATION, target_node_name=conv2_node_name
    )

    linear_node_name = "InsertionPointTestModel/linear_0"
    point_for_linear_weight_input = PTTargetPoint(
        target_type=TargetType.OPERATOR_PRE_HOOK, target_node_name=linear_node_name, input_port_id=0
    )
    point_for_linear_activation = PTTargetPoint(
        target_type=TargetType.OPERATOR_POST_HOOK, target_node_name=linear_node_name
    )

    relu_node_name = "InsertionPointTestModel/ReLU[relu]/relu_0"
    point_for_relu_inputs = PTTargetPoint(
        target_type=TargetType.OPERATOR_PRE_HOOK, target_node_name=relu_node_name, input_port_id=0
    )
    point_for_relu_activations = PTTargetPoint(
        target_type=TargetType.OPERATOR_POST_HOOK, target_node_name=relu_node_name
    )

    available_points = [
        point_for_conv1_weights,
        point_for_conv2_weights,
        point_for_conv1_inputs,
        point_for_conv2_inputs,
        point_for_conv1_activations,
        point_for_conv2_activations,
        point_for_linear_activation,
        point_for_linear_weight_input,
        point_for_relu_activations,
        point_for_relu_inputs,
    ]

    @pytest.mark.parametrize("target_point", available_points)
    def test_single_insertions(self, setup, target_point: PTTargetPoint):
        insertion_point = PTInsertionPoint(
            target_point.target_type,
            OperationAddress.from_str(target_point.target_node_name),
            target_point.input_port_id,
        )
        if insertion_point.insertion_type in [PTInsertionType.OPERATOR_PRE_HOOK, PTInsertionType.OPERATOR_POST_HOOK]:
            hook = lambda x: x
        else:
            hook = BaseOp(lambda x: x)

        self.compressed_model.nncf.insert_at_point(insertion_point, [hook])

        # pylint:disable=protected-access
        if insertion_point.insertion_type == PTInsertionType.OPERATOR_PRE_HOOK:
            ctx = self.compressed_model.nncf.get_tracing_context()
            pre_hook_id = PreHookId(insertion_point.op_address, input_port_id=insertion_point.input_port_id)
            assert ctx._pre_hooks[pre_hook_id][0] is hook
        if insertion_point.insertion_type == PTInsertionType.OPERATOR_POST_HOOK:
            ctx = self.compressed_model.nncf.get_tracing_context()
            assert ctx._post_hooks[insertion_point.op_address][0] is hook
        if insertion_point.insertion_type == PTInsertionType.NNCF_MODULE_PRE_OP:
            module = self.compressed_model.nncf.get_module_by_scope(insertion_point.module_scope)
            assert module.pre_ops["0"] is hook

        if insertion_point.insertion_type == PTInsertionType.NNCF_MODULE_POST_OP:
            module = self.compressed_model.nncf.get_module_by_scope(insertion_point.module_scope)
            assert module.post_ops["0"] is hook

    priority_types = ["same", "different"]
    insertion_types = TargetType
    priority_test_cases = list(itertools.product(priority_types, insertion_types))

    @staticmethod
    def check_order(iterable1: List, iterable2: List, ordering: List):
        for idx, order in enumerate(ordering):
            assert iterable1[idx] is iterable2[order]

    # pylint:disable=undefined-variable
    @pytest.mark.parametrize("case", priority_test_cases, ids=[x[1].name + "-" + x[0] for x in priority_test_cases])
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
            ctx = self.compressed_model.nncf.get_tracing_context()
            pre_hook_id = PreHookId(
                OperationAddress.from_str(point.target_node_name), input_port_id=point.input_port_id
            )
            self.check_order(ctx._pre_hooks[pre_hook_id], hook_list, order)
        if insertion_type == TargetType.OPERATOR_POST_HOOK:
            ctx = self.compressed_model.nncf.get_tracing_context()
            self.check_order(ctx._post_hooks[OperationAddress.from_str(point.target_node_name)], hook_list, order)

        if insertion_type == TargetType.OPERATION_WITH_WEIGHTS:
            module = self.compressed_model.nncf.get_containing_module(point.target_node_name)
            # Works because Pytorch ModuleDict is ordered
            self.check_order([x.operand for x in module.pre_ops.values()], hook_list, order)

        if insertion_type == TargetType.POST_LAYER_OPERATION:
            module = self.compressed_model.nncf.get_containing_module(point.target_node_name)
            # Works because Pytorch ModuleDict is ordered
            self.check_order(list(module.post_ops.values()), hook_list, order)


MERGE_PATTERN_TEST_CASES = (
    [get_mock_model_graph_with_mergeable_pattern, "basic_pattern"],
    [get_mock_model_graph_with_no_mergeable_pattern, "no_pattern"],
    [get_mock_model_graph_with_broken_output_edge_pattern, "broken_output_edges_pattern"],
)


class TestInsertionPointGraph:
    def test_insertion_point_setup(self):
        # TODO: Change testing premises when module pre/post-op hooks and input/output nodes
        # are correctly handled
        mock_graph = get_two_branch_mock_model_graph()

        ip_graph = get_ip_graph_for_test(mock_graph)

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
            assert post_hook_ip_node_type == InsertionPointGraphNodeType.POST_HOOK

            pre_hook_ip_node_keys = preds
            for pre_hook_ip_node_key in pre_hook_ip_node_keys:
                pre_hook_ip_node = ip_graph.nodes[pre_hook_ip_node_key]
                pre_hook_ip_node_type = pre_hook_ip_node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
                assert pre_hook_ip_node_type == InsertionPointGraphNodeType.PRE_HOOK

            ref_associated_ip_node_keys_set = {*pre_hook_ip_node_keys, post_hook_ip_node_key}
            assert (
                ref_associated_ip_node_keys_set
                == ip_graph_op_node[InsertionPointGraph.ASSOCIATED_IP_NODE_KEYS_NODE_ATTR]
            )
            original_neighbours = nx_graph.neighbors(node_key)
            for neighbour in original_neighbours:
                # IP node insertion should not disrupt the graph superstructure
                ip_graph_paths = list(nx.all_simple_paths(ip_graph, node_key, neighbour))
                for path in ip_graph_paths:
                    path = path[1:-1]
                    for path_node_key in path:
                        node = ip_graph.nodes[path_node_key]
                        node_type = node[InsertionPointGraph.NODE_TYPE_NODE_ATTR]
                        assert node_type in [
                            InsertionPointGraphNodeType.PRE_HOOK,
                            InsertionPointGraphNodeType.POST_HOOK,
                        ]

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

        mock_graph.add_node("bar")
        mock_graph.add_node("baz")
        mock_graph.add_edge("bar", "baz")
        nncf_graph = get_nncf_graph_from_mock_nx_graph(mock_graph)

        ip_graph = get_ip_graph_for_test(nncf_graph)

        for nncf_node in nncf_graph.get_all_nodes():
            node_id = nncf_node.node_id
            node_key = nncf_graph.get_node_key_by_id(node_id)
            preds = list(ip_graph.predecessors(node_key))
            succs = list(ip_graph.successors(node_key))

            post_hook_ip_node = ip_graph.nodes[succs[0]]
            post_hook_ip = post_hook_ip_node[InsertionPointGraph.INSERTION_POINT_NODE_ATTR]
            assert isinstance(post_hook_ip, PostHookInsertionPoint)
            assert post_hook_ip.target_node_name == nncf_node.node_name

            for pre_hook_ip_node_key in preds:
                pre_hook_ip_node = ip_graph.nodes[pre_hook_ip_node_key]
                pre_hook_ip = pre_hook_ip_node[InsertionPointGraph.INSERTION_POINT_NODE_ATTR]
                assert isinstance(pre_hook_ip, PreHookInsertionPoint)
                assert pre_hook_ip.target_node_name == nncf_node.node_name

    def test_operator_metatype_marking(self):
        from nncf.torch.graph.operator_metatypes import PTAddMetatype
        from nncf.torch.graph.operator_metatypes import PTAvgPool2dMetatype
        from nncf.torch.graph.operator_metatypes import PTBatchNormMetatype
        from nncf.torch.graph.operator_metatypes import PTConvTranspose2dMetatype
        from nncf.torch.graph.operator_metatypes import PTDepthwiseConv2dSubtype
        from nncf.torch.graph.operator_metatypes import PTLinearMetatype
        from nncf.torch.graph.operator_metatypes import PTMaxPool2dMetatype
        from nncf.torch.graph.operator_metatypes import PTModuleBatchNormMetatype
        from nncf.torch.graph.operator_metatypes import PTModuleConvTranspose2dMetatype
        from nncf.torch.graph.operator_metatypes import PTModuleLinearMetatype
        from nncf.torch.graph.operator_metatypes import PTRELUMetatype
        from nncf.torch.graph.operator_metatypes import PTTransposeMetatype

        ref_scope_vs_metatype_dict = {
            "/" + MODEL_INPUT_OP_NAME + "_0": PTInputNoopMetatype,
            "ModelForMetatypeTesting/NNCFConv2d[conv_regular]/conv2d_0": PTModuleConv2dMetatype,
            "ModelForMetatypeTesting/NNCFBatchNorm2d[bn]/batch_norm_0": PTModuleBatchNormMetatype,
            "ModelForMetatypeTesting/batch_norm_0": PTBatchNormMetatype,
            "ModelForMetatypeTesting/relu_0": PTRELUMetatype,
            "ModelForMetatypeTesting/transpose__0": PTTransposeMetatype,
            "ModelForMetatypeTesting/MaxPool2d[max_pool2d]/max_pool2d_0": PTMaxPool2dMetatype,
            "ModelForMetatypeTesting/NNCFConvTranspose2d[conv_transpose]/conv_transpose2d_0": PTModuleConvTranspose2dMetatype,
            "ModelForMetatypeTesting/conv_transpose2d_0": PTConvTranspose2dMetatype,
            "ModelForMetatypeTesting/__add___0": PTAddMetatype,
            "ModelForMetatypeTesting/NNCFConv2d[conv_depthwise]/conv2d_0": PTDepthwiseConv2dSubtype,
            "ModelForMetatypeTesting/conv2d_0": PTConv2dMetatype,
            "ModelForMetatypeTesting/__iadd___0": PTAddMetatype,
            "ModelForMetatypeTesting/AdaptiveAvgPool2d[adaptive_avg_pool]/adaptive_avg_pool2d_0": PTAvgPool2dMetatype,
            "ModelForMetatypeTesting/flatten_0": PTReshapeMetatype,
            "ModelForMetatypeTesting/NNCFLinear[linear]/linear_0": PTModuleLinearMetatype,
            "ModelForMetatypeTesting/linear_0": PTLinearMetatype,
            "/" + MODEL_OUTPUT_OP_NAME + "_0": PTOutputNoopMetatype,
        }

        class ModelForMetatypeTesting(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_regular = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
                self.bn = torch.nn.BatchNorm2d(num_features=16)
                self.max_pool2d = torch.nn.MaxPool2d(kernel_size=2)
                self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3)
                self.conv_depthwise = torch.nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, groups=8)
                self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
                self.linear = torch.nn.Linear(in_features=8, out_features=8)

            def forward(self, input_):
                x = self.conv_regular(input_)
                x = self.bn(x)
                x = F.batch_norm(x, self.bn.running_mean, self.bn.running_var)
                x = F.relu(x)
                x.transpose_(2, 3)
                x = self.max_pool2d(x)
                y = self.conv_transpose(x)
                z = F.conv_transpose2d(x, self.conv_transpose.weight)
                x = y + z
                x = self.conv_depthwise(x)
                x = F.conv2d(x, self.conv_depthwise.weight, groups=self.conv_depthwise.groups)
                x += torch.ones_like(x)
                x = self.adaptive_avg_pool(x)
                x = self.linear(x.flatten())
                x = F.linear(x, self.linear.weight)
                return x

        model = ModelForMetatypeTesting()
        nncf_network = NNCFNetwork(model, [ModelInputInfo([1, 3, 300, 300])])
        nncf_graph = nncf_network.nncf.get_original_graph()

        for nncf_node in nncf_graph.get_all_nodes():  # type: NNCFNode
            assert nncf_node.node_name in ref_scope_vs_metatype_dict
            ref_metatype = ref_scope_vs_metatype_dict[nncf_node.node_name]
            assert nncf_node.metatype == ref_metatype

    @pytest.mark.parametrize(
        ("mock_graph_factory", "dot_file_name"), MERGE_PATTERN_TEST_CASES, ids=[x[1] for x in MERGE_PATTERN_TEST_CASES]
    )
    def test_get_ip_graph_with_merged_operations(self, mock_graph_factory, dot_file_name):
        mock_graph = mock_graph_factory()
        ip_graph = get_ip_graph_for_test(mock_graph)
        pattern = PatternsManager.get_full_pattern_graph(BackendType.TORCH, TargetDevice.ANY)
        merged_ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        data_dir = TEST_ROOT / "torch/data/reference_graphs/pattern_merging"  # type: Path

        path_to_dot_file = data_dir / "{}.dot".format(dot_file_name)

        if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
            if not os.path.exists(str(data_dir)):
                os.makedirs(str(data_dir))
            graph_without_data = get_graph_without_data(merged_ip_graph)
            write_dot_graph(graph_without_data, str(path_to_dot_file))

        load_graph = read_dot_graph(str(path_to_dot_file))

        for key in load_graph.nodes.keys():
            key.replace(r"\\n", r"\n")  # Somehow pydot mangles the \n characters while writing a .dot file

        sanitized_loaded_keys = [key.replace("\\n", "\n") for key in load_graph.nodes.keys()]
        sanitized_loaded_edges = [
            (u.replace("\\n", "\n"), v.replace("\\n", "\n")) for u, v in nx.DiGraph(load_graph).edges
        ]

        assert Counter(sanitized_loaded_keys) == Counter(list(merged_ip_graph.nodes.keys()))
        assert Counter(sanitized_loaded_edges) == Counter(list(merged_ip_graph.edges))


def test_can_collect_scopes_of_train_only_modules():
    model = ManyNonEvalModules()
    graph_builder = GraphBuilder(custom_forward_fn=lambda model_: model_(torch.randn([1, 1, 1, 1])))
    graph = graph_builder.build_graph(model, as_eval=True)
    actual_scopes = [n.node_name for n in graph.get_all_nodes()]
    ref_scopes = {
        "ManyNonEvalModules/AvgPool2d[avg_pool]/avg_pool2d_0",
        "ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/Dropout/dropout_0",
        "ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/Dropout/dropout_1",
        "ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/Linear[called_linear]/linear_0",
        "ManyNonEvalModules/ModuleWithMixedModules[mixed_modules]/CustomWeightModule[custom]/linear_0",
    }
    assert set(actual_scopes) == ref_scopes


def test_get_clean_shallow_copy():
    model = TwoConvTestModelWithUserModule()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    external_quantizers = getattr(sparse_quantized_model.nncf, EXTERNAL_QUANTIZERS_STORAGE_NAME)
    assert external_quantizers
    old_nncf_modules = sparse_quantized_model.nncf.get_nncf_modules()
    old_nncf_module_pre_ops = [module.pre_ops for module in old_nncf_modules]
    assert any(old_nncf_module_pre_ops)
    assert (
        sparse_quantized_model.nncf.get_graph().get_nodes_count()
        != sparse_quantized_model.nncf.get_original_graph().get_nodes_count()
    )

    old_interface = sparse_quantized_model.nncf
    clean_copy = sparse_quantized_model.nncf.get_clean_shallow_copy()
    assert clean_copy is sparse_quantized_model
    new_interface = clean_copy.nncf
    assert old_interface is not new_interface
    new_nncf_modules = clean_copy.nncf.get_nncf_modules()
    new_nncf_module_pre_ops = [module.pre_ops for module in new_nncf_modules]
    assert not any(new_nncf_module_pre_ops)
    assert clean_copy.nncf.get_graph().get_nodes_count() == clean_copy.nncf.get_original_graph().get_nodes_count()


class TwoConvTestModelWithUniqueFunction(TwoConvTestModel):
    def __init__(self):
        super().__init__()
        self.unique_attr = "unique_attr"
        self.non_unique_attr = "model_non_unique_attr"

    def train_step(self):
        pass

    @staticmethod
    def static_func():
        pass


def test_get_attr():
    model = TwoConvTestModelWithUniqueFunction()
    nncf_network = NNCFNetwork(model, [ModelInputInfo([1, 1, 4, 4])])

    assert hasattr(nncf_network, "unique_attr")
    assert hasattr(nncf_network, "non_unique_attr")
    assert inspect.ismethod(nncf_network.train_step)
    assert inspect.isfunction(nncf_network.static_func)


class ModelWithAttr(torch.nn.Module):
    CLASS_ATTR = 0

    def __init__(self):
        super().__init__()
        self.instance_attr = 0
        self._input_infos = 0  # deliberately set to coincide with an attr in NNCFNetwork

    def forward(self, x):
        return x

    def other_forward(self, x):
        return x * 2


def test_setting_attrs():
    model = ModelWithAttr()
    assert model.CLASS_ATTR == 0
    assert model.instance_attr == 0
    # pylint:disable=protected-access
    assert model._input_infos == 0
    nncf_network = NNCFNetwork(model, input_infos=[ModelInputInfo([1])])
    # pylint:disable=protected-access
    assert nncf_network._input_infos == 0

    nncf_network.instance_attr = 1
    assert nncf_network.instance_attr == 1

    nncf_network.non_original_model_attr = 1
    assert nncf_network.non_original_model_attr == 1
    assert hasattr(nncf_network, "non_original_model_attr")
    assert nncf_network.non_original_model_attr == 1


def mock_forward(*args, **kwargs):
    mock_forward.called = True


def mock_forward_with_self(self, *args, **kwargs):
    mock_forward_with_self.called = True


def mock_forward_with_matching_signature(self, x):
    mock_forward_with_matching_signature.called = True


@pytest.mark.parametrize("new_forward", [mock_forward, mock_forward_with_self, mock_forward_with_matching_signature])
def test_replacing_forward_with_free_functions(new_forward, _nncf_caplog):
    model = ModelWithAttr()
    nncf_network = NNCFNetwork(model, input_infos=[ModelInputInfo([1])])

    nncf_network.forward = new_forward
    assert "set_original_unbound_forward" in _nncf_caplog.text


def test_replacing_forward_with_another_own_method(_nncf_caplog):
    model = ModelWithAttr()
    nncf_network = NNCFNetwork(model, input_infos=[ModelInputInfo([1])])

    nncf_network.forward = nncf_network.other_forward
    assert "set_original_unbound_forward" in _nncf_caplog.text


def test_temporary_clean_view():
    model = TwoConvTestModelWithUserModule()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    old_sd = sparse_quantized_model.state_dict()
    old_graph = deepcopy(sparse_quantized_model.nncf.get_graph())
    with sparse_quantized_model.nncf.temporary_clean_view() as intermediate_model:
        clean_sd = intermediate_model.state_dict()
        assert len(clean_sd) < len(old_sd)
        new_nncf_modules = intermediate_model.nncf.get_nncf_modules()
        new_nncf_module_pre_ops = [module.pre_ops for module in new_nncf_modules]
        assert not any(new_nncf_module_pre_ops)
        assert (
            intermediate_model.nncf.get_graph().get_nodes_count()
            == intermediate_model.nncf.get_original_graph().get_nodes_count()
        )
    sd_after_tmp_clean_view = sparse_quantized_model.state_dict()
    for key in old_sd.keys():
        assert key in sd_after_tmp_clean_view  # pylint: disable=E1135
        assert torch.all(torch.eq(sd_after_tmp_clean_view[key], old_sd[key]))  # pylint: disable=E1136
    sparse_quantized_model.nncf.rebuild_graph()
    graph_after_tmp_clean_view = sparse_quantized_model.nncf.get_graph()
    assert graph_after_tmp_clean_view == old_graph


class MultipleForwardModel(nn.Module):
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
    model = MultipleForwardModel()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    graph = sparse_quantized_model.nncf.get_original_graph()
    for node in list(graph.get_all_nodes())[1:-2]:
        assert node.layer_attributes is not None


def test_deepcopy_nncf_network():
    model = TwoConvTestModelWithUserModule()
    config = get_basic_sparsity_plus_quantization_config()
    register_bn_adaptation_init_args(config)
    sparse_quantized_model, _ = create_compressed_model_and_algo_for_test(model, config)
    _ = deepcopy(sparse_quantized_model)


def test_insertion_point_target_point_translation():
    op_address = OperationAddress("dummy", Scope(), 0)
    for target_type in [PTInsertionType.NNCF_MODULE_POST_OP, TargetType.AFTER_LAYER]:
        with pytest.raises(RuntimeError):
            PTInsertionPoint(target_type, op_address)
    target_type = TargetType.POST_LAYER_OPERATION
    assert PTInsertionPoint(target_type, op_address).insertion_type == PTInsertionType.NNCF_MODULE_POST_OP


class IndirectModuleCaller(nn.Module):
    def __init__(self, module_for_indirection: torch.nn.Module):
        super().__init__()

        self.module_for_indirection = module_for_indirection
        self.conv_immediate = nn.Conv2d(32, 32, 3, 1, 1, bias=False)

    def forward(self, img):
        enc_features = self.module_for_indirection.forward(img)
        out = self.conv_immediate(enc_features)

        return out


class Backbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels

        # This is another source of indirection - a module whose reference then
        # is being passed to an owning container (Sequential). NNCFNetwork creation algo
        # must replace the module object in both places with the same object.
        self.conv_indirect = nn.Conv2d(self.in_channels, 32, 3, 2, 1, bias=False)
        self.features = nn.Sequential(self.conv_indirect, nn.BatchNorm2d(32), nn.ReLU6(inplace=True))

    def forward(self, img):
        enc_features = self.features(img)

        return enc_features


class ModelWithIndirectModuleCallBranch(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.in_channels = in_channels

        self.backbone = Backbone(self.in_channels)
        self.testBranch = IndirectModuleCaller(self.backbone)

    def forward(self, img):
        out = self.testBranch(img)
        return out


def test_wrapping_of_indirect_module_operations():
    model = NNCFNetwork(ModelWithIndirectModuleCallBranch(), [ModelInputInfo([1, 3, 32, 32])])
    assert isinstance(model.backbone.conv_indirect, NNCFConv2d)
    assert model.backbone.features[0] is model.backbone.conv_indirect
    assert model.testBranch.module_for_indirection.features[0] is model.backbone.features[0]
    assert model.testBranch.module_for_indirection.conv_indirect is model.backbone.conv_indirect
    assert model.testBranch.module_for_indirection.features[0] is model.testBranch.module_for_indirection.conv_indirect
    assert isinstance(model.testBranch.conv_immediate, NNCFConv2d)


def test_can_work_with_sequential_models():
    sequential = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Conv2d(1, 1, 1))
    model = NNCFNetwork(sequential, [ModelInputInfo([1, 1, 32, 32])])
    assert model.nncf not in model  # Sequential, even wrapped, should only iterate over its real modules
    model(torch.ones([1, 1, 1, 1]))
    _ = model.nncf.get_clean_shallow_copy()


class SimplestModel(torch.nn.Module):
    INPUT_SIZE = [1, 1, 32, 32]

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x)


@pytest.fixture(name="simple_net")
def simple_net_():
    model = NNCFNetwork(SimplestModel(), [ModelInputInfo(SimplestModel.INPUT_SIZE)])
    return model


def test_works_when_wrapped_with_dataparallel(simple_net):
    if not torch.cuda.is_available():
        pytest.xfail("The executing host must have > 1 CUDA GPU in order for this test to be relevant.")
    simple_net.cuda()
    dp_model = torch.nn.DataParallel(simple_net)
    dp_model(torch.zeros([10, *simple_net.INPUT_SIZE[1:]], device="cuda"))


def test_works_with_old_wrapping_and_deprecation_warning(simple_net):
    with pytest.warns(NNCFDeprecationWarning):
        simple_net.get_graph()


def test_class_has_same_name_and_module_as_original(simple_net):
    assert simple_net.__class__.__name__ == SimplestModel.__name__
    assert simple_net.__class__.__module__ == SimplestModel.__module__


def test_class_hashes_as_original(simple_net):
    assert hash(simple_net.__class__) == hash(SimplestModel)


def test_class_compares_as_original(simple_net):
    assert simple_net.__class__ == SimplestModel
    assert SimplestModel == simple_net.__class__
    assert simple_net.__class__ == simple_net.__class__
    assert not simple_net.__class__ != simple_net.__class__
    assert simple_net.__class__ != ModelWithAttr
    assert ModelWithAttr != simple_net.__class__


class MultiInputModel(torch.nn.Module):
    def forward(self, x, y):
        return x, y


def test_forward_signature_is_same_as_for_original_model(simple_net):
    original_obj = SimplestModel()
    original_signature_inst = inspect.signature(original_obj.forward)
    net_signature_inst = inspect.signature(simple_net.forward)
    assert original_signature_inst == net_signature_inst

    original_signature_cls = inspect.signature(SimplestModel.forward)
    net_signature_cls = inspect.signature(simple_net.__class__.forward)
    assert original_signature_cls == net_signature_cls

    # Verify that if we create 2 NNCFNetworks, then each will have its own signature
    another_original_obj = MultiInputModel()
    another_nncf_net = NNCFNetwork(
        MultiInputModel(), input_infos=[ModelInputInfo([1, 1, 1, 1]), ModelInputInfo([1, 1, 1, 1])]
    )
    assert inspect.signature(another_nncf_net.forward) == inspect.signature(another_original_obj.forward)
    assert inspect.signature(simple_net.forward) == inspect.signature(original_obj.forward)


class MetaModel(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def method(self):
        pass


class ConcreteMetaModel(MetaModel):
    def method(self):
        pass

    def forward(self, x):
        return x


def test_can_wrap_models_with_metaclass():
    _ = NNCFNetwork(ConcreteMetaModel(), [ModelInputInfo([1, 1, 1, 1])])
