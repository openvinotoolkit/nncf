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
import os
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List

import networkx as nx
import pytest
import torch
import torch.nn.functional as F
from torch import nn

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.definitions import MODEL_INPUT_OP_NAME
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.patterns.manager import TargetDevice
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.insertion_point_graph import InsertionPointGraph
from nncf.common.insertion_point_graph import InsertionPointGraphNodeType
from nncf.common.insertion_point_graph import PostHookInsertionPoint
from nncf.common.insertion_point_graph import PreHookInsertionPoint
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.utils.backend import BackendType
from nncf.common.utils.dot_file_rw import get_graph_without_data
from nncf.common.utils.dot_file_rw import read_dot_graph
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.torch import wrap_model
from nncf.torch.dynamic_graph.context import PreHookId
from nncf.torch.dynamic_graph.io_handling import FillerInputElement
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.dynamic_graph.patch_pytorch import register_operator
from nncf.torch.external_hook import ExternalOpCallHook
from nncf.torch.graph.operator_metatypes import PTInputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTOutputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.transformations.command_creation import create_quantizer_insertion_command
from nncf.torch.graph.transformations.command_creation import create_shared_quantizer_insertion_command
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTBiasCorrectionCommand
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTWeightUpdateCommand
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.layers import register_module
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.module_operations import BaseOp
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTInsertionPoint
from nncf.torch.nncf_network import PTInsertionType
from nncf.torch.nncf_network import compression_module_type_to_attr_name
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.utils import get_model_device
from tests.common.quantization.mock_graphs import get_ip_graph_for_test
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_broken_output_edge_pattern
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_mergeable_pattern
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_no_mergeable_pattern
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph
from tests.common.quantization.mock_graphs import get_two_branch_mock_model_graph
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.helpers import HookChecker


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
    conv1_node_name = "InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0"
    point_for_conv1_weights = PTTargetPoint(
        target_type=TargetType.OPERATION_WITH_WEIGHTS, target_node_name=conv1_node_name, input_port_id=1
    )
    point_for_conv1_inputs = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK, target_node_name=conv1_node_name)
    point_for_conv1_activations = PTTargetPoint(
        target_type=TargetType.PRE_LAYER_OPERATION, target_node_name=conv1_node_name, input_port_id=0
    )
    point_for_conv1_post = PTTargetPoint(target_type=TargetType.POST_LAYER_OPERATION, target_node_name=conv1_node_name)

    conv2_node_name = "InsertionPointTestModel/NNCFConv2d[conv2]/conv2d_0"
    point_for_conv2_weights = PTTargetPoint(
        target_type=TargetType.OPERATION_WITH_WEIGHTS, target_node_name=conv2_node_name, input_port_id=1
    )
    point_for_conv2_inputs = PTTargetPoint(target_type=TargetType.OPERATOR_PRE_HOOK, target_node_name=conv2_node_name)
    point_for_conv2_activations = PTTargetPoint(
        target_type=TargetType.PRE_LAYER_OPERATION, target_node_name=conv2_node_name, input_port_id=0
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
        point_for_conv1_post,
        point_for_linear_activation,
        point_for_linear_weight_input,
        point_for_relu_activations,
        point_for_relu_inputs,
    ]

    @pytest.mark.parametrize("target_point", available_points)
    @pytest.mark.parametrize("trace_parameters", (True, False))
    def test_single_insertions(self, trace_parameters, target_point: PTTargetPoint):
        model = wrap_model(InsertionPointTestModel(), torch.ones([1, 1, 10, 10]), trace_parameters)
        insertion_point = PTInsertionPoint(
            target_point.target_type,
            OperationAddress.from_str(target_point.target_node_name),
            target_point.input_port_id,
            not trace_parameters,
        )
        if insertion_point.insertion_type in [PTInsertionType.OPERATOR_PRE_HOOK, PTInsertionType.OPERATOR_POST_HOOK]:

            def hook(x):
                return x

        else:
            hook = BaseOp(lambda x: x)

        test_hook_group = "test_hook_group"
        model.nncf.insert_at_point(insertion_point, hook, hooks_group_name=test_hook_group)

        if insertion_point.insertion_type == PTInsertionType.OPERATOR_PRE_HOOK:
            ctx = model.nncf.get_tracing_context()
            pre_hook_id = PreHookId(insertion_point.op_address, input_port_id=insertion_point.input_port_id)
            assert ctx._pre_hooks[pre_hook_id]["0"] is hook
        elif insertion_point.insertion_type == PTInsertionType.OPERATOR_POST_HOOK:
            ctx = model.nncf.get_tracing_context()
            assert ctx._post_hooks[insertion_point.op_address]["0"] is hook
        elif insertion_point.insertion_type == PTInsertionType.NNCF_MODULE_PRE_OP:
            module = model.nncf.get_module_by_scope(insertion_point.module_scope)
            assert module.pre_ops["0"] is hook
        elif insertion_point.insertion_type == PTInsertionType.NNCF_MODULE_POST_OP:
            assert not trace_parameters
            module = model.nncf.get_module_by_scope(insertion_point.module_scope)
            assert module.post_ops["0"] is hook
        else:
            raise Exception(f"Not check order for {insertion_point.insertion_type}")

        assert len(model.nncf._groups_vs_hooks_handlers[test_hook_group]) == 1

    class BaseOpWithParam(BaseOp):
        def __init__(self, op):
            super().__init__(op)
            self.param1 = torch.nn.Parameter(torch.zeros((1,)))
            self.param2 = torch.nn.Parameter(torch.zeros((1,)))
            self.to_device = None

        def to(self, device):
            super().to(device)
            self.to_device = device

    @pytest.mark.parametrize("target_point", available_points)
    @pytest.mark.parametrize("multidevice", (False, pytest.param(True, marks=pytest.mark.cuda)))
    @pytest.mark.parametrize("hook", (lambda x: x, BaseOpWithParam(lambda x: x).cpu()))
    def test_pt_insertion_command(self, target_point: PTTargetPoint, multidevice: bool, hook):
        model = wrap_model(InsertionPointTestModel(), torch.ones([1, 1, 10, 10]))

        if multidevice:
            if not torch.cuda.is_available():
                pytest.skip("Cuda is not available, could not run multidevice test case")
            model.conv2.to("cuda")

        test_hook_group = "test_hook_group"
        insertion_command = PTInsertionCommand(target_point, hook, hooks_group_name=test_hook_group)
        layout = PTTransformationLayout()
        layout.register(insertion_command)
        transformer = PTModelTransformer(model)

        if target_point.target_type in [
            TargetType.PRE_LAYER_OPERATION,
            TargetType.POST_LAYER_OPERATION,
        ] and not isinstance(hook, nn.Module):
            with pytest.raises(TypeError):
                transformer.transform(layout)
            return
        transformer.transform(layout)

        insertion_point = PTInsertionPoint(
            target_point.target_type,
            model.nncf.get_node_to_op_address_mapping()[target_point.target_node_name],
            target_point.input_port_id,
        )

        if target_point.target_type == TargetType.OPERATOR_PRE_HOOK:
            ctx = model.nncf.get_tracing_context()
            pre_hook_id = PreHookId(insertion_point.op_address, input_port_id=insertion_point.input_port_id)
            assert ctx._pre_hooks[pre_hook_id]["0"] is hook
        elif target_point.target_type == TargetType.OPERATOR_POST_HOOK:
            ctx = model.nncf.get_tracing_context()
            assert ctx._post_hooks[insertion_point.op_address]["0"] is hook
        elif target_point.target_type == TargetType.OPERATION_WITH_WEIGHTS:
            module = model.nncf.get_module_by_scope(insertion_point.module_scope)
            w_hook = module.pre_ops["0"]
            assert isinstance(w_hook, UpdateWeight)
            assert w_hook.op is hook
        elif target_point.target_type == TargetType.PRE_LAYER_OPERATION:
            module = model.nncf.get_module_by_scope(insertion_point.module_scope)
            assert module.pre_ops["0"] is hook
        elif target_point.target_type == TargetType.POST_LAYER_OPERATION:
            module = model.nncf.get_module_by_scope(insertion_point.module_scope)
            assert module.post_ops["0"] is hook
        else:
            raise Exception(f"Not check order for {insertion_point.insertion_type}")

        if isinstance(hook, nn.Module) and not multidevice:
            assert hook.to_device == get_model_device(model)

        assert len(model.nncf._groups_vs_hooks_handlers[test_hook_group]) == 1

    @staticmethod
    def check_order(iterable1: List, iterable2: List, ordering: List):
        for idx, order in enumerate(ordering):
            assert iterable1[idx] is iterable2[order]

    @pytest.mark.parametrize("priority_type", ("same", "different"))
    @pytest.mark.parametrize(
        "trace_parameters, target_type",
        (
            (False, TargetType.POST_LAYER_OPERATION),
            (False, TargetType.OPERATION_WITH_WEIGHTS),
            (False, TargetType.OPERATOR_PRE_HOOK),
            (False, TargetType.OPERATOR_POST_HOOK),
            (True, TargetType.OPERATION_WITH_WEIGHTS),
            (True, TargetType.OPERATOR_PRE_HOOK),
            (True, TargetType.OPERATOR_POST_HOOK),
        ),
    )
    def test_priority(self, target_type, trace_parameters, priority_type):
        model = wrap_model(InsertionPointTestModel(), torch.ones([1, 1, 10, 10]), trace_parameters=trace_parameters)

        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            hook1 = BaseOp(lambda x: x)
            hook2 = BaseOp(lambda x: 2 * x)
            hook3 = BaseOp(lambda x: 3 * x)
        elif target_type in [TargetType.POST_LAYER_OPERATION, TargetType.POST_LAYER_OPERATION]:
            hook1 = BaseOp(lambda m, x: x)
            hook2 = BaseOp(lambda m, x: 2 * x)
            hook3 = BaseOp(lambda m, x: 3 * x)
        else:
            hook1 = lambda x: x
            hook2 = lambda x: 2 * x
            hook3 = lambda x: 3 * x

        if target_type == TargetType.OPERATION_WITH_WEIGHTS:
            point = self.point_for_conv1_weights
        elif target_type == TargetType.PRE_LAYER_OPERATION:
            point = self.point_for_conv1_activations
        elif target_type == TargetType.POST_LAYER_OPERATION:
            point = self.point_for_conv1_post
        elif target_type == TargetType.OPERATOR_PRE_HOOK:
            point = self.point_for_linear_weight_input
        elif target_type == TargetType.OPERATOR_POST_HOOK:
            point = self.point_for_relu_activations
        point = deepcopy(point)

        if trace_parameters:
            point.target_node_name = point.target_node_name.replace("NNCFConv2d", "Conv2d")

        if priority_type == "same":
            # Same-priority commands will be executed in registration order
            command1 = PTInsertionCommand(point, hook1, TransformationPriority.DEFAULT_PRIORITY)
            command2 = PTInsertionCommand(point, hook2, TransformationPriority.DEFAULT_PRIORITY)
            command3 = PTInsertionCommand(point, hook3, TransformationPriority.DEFAULT_PRIORITY)
        else:
            # Prioritized commands will be executed in ascending priority order
            command1 = PTInsertionCommand(point, hook1, TransformationPriority.SPARSIFICATION_PRIORITY)
            command2 = PTInsertionCommand(point, hook2, TransformationPriority.QUANTIZATION_PRIORITY)
            command3 = PTInsertionCommand(
                point,
                hook3,
                TransformationPriority.DEFAULT_PRIORITY,
            )

        layout = PTTransformationLayout()
        layout.register(command1)
        layout.register(command2)
        layout.register(command3)
        model = PTModelTransformer(model).transform(layout)

        hook_list = [hook1, hook2, hook3]

        if priority_type == "same":
            order = [0, 1, 2]
        elif priority_type == "different":
            order = [2, 0, 1]

        ctx = model.nncf.get_tracing_context()
        if target_type == TargetType.OPERATOR_PRE_HOOK:
            pre_hook_id = PreHookId(
                OperationAddress.from_str(point.target_node_name), input_port_id=point.input_port_id
            )
            actual_pre_hooks = list(ctx._pre_hooks[pre_hook_id].values())
            self.check_order(actual_pre_hooks, hook_list, order)
        elif target_type == TargetType.OPERATOR_POST_HOOK:
            actual_hooks = list(ctx._post_hooks[OperationAddress.from_str(point.target_node_name)].values())
            self.check_order(actual_hooks, hook_list, order)
        elif target_type == TargetType.OPERATION_WITH_WEIGHTS:
            if trace_parameters:
                pre_hook_id = PreHookId(
                    OperationAddress.from_str(point.target_node_name), input_port_id=point.input_port_id
                )
                actual_hooks = list(ctx._pre_hooks[pre_hook_id].values())
                self.check_order(actual_hooks, hook_list, order)
            else:
                module = model.nncf.get_containing_module(point.target_node_name)
                self.check_order([x.operand for x in module.pre_ops.values()], hook_list, order)

        elif target_type == TargetType.POST_LAYER_OPERATION:
            if trace_parameters:
                op_address = OperationAddress.from_str(point.target_node_name)
                actual_hooks = list(ctx._post_hooks[op_address].values())
                self.check_order(actual_hooks, hook_list, order)
            else:
                module = model.nncf.get_containing_module(point.target_node_name)
                self.check_order(list(module.post_ops.values()), hook_list, order)
        else:
            raise Exception(f"Not check order for {target_type}")


MERGE_PATTERN_TEST_CASES = (
    [
        partial(
            get_mock_model_graph_with_mergeable_pattern,
            conv2d_metatype=om.PTConv2dMetatype,
            batchnorm_metatype=om.PTBatchNormMetatype,
            relu_metatype=om.PTRELUMetatype,
        ),
        "basic_pattern",
    ],
    [
        partial(
            get_mock_model_graph_with_no_mergeable_pattern,
            conv2d_metatype=om.PTConv2dMetatype,
            batchnorm_metatype=om.PTBatchNormMetatype,
            relu_metatype=om.PTRELUMetatype,
        ),
        "no_pattern",
    ],
    [
        partial(
            get_mock_model_graph_with_broken_output_edge_pattern,
            conv2d_metatype=om.PTConv2dMetatype,
            batchnorm_metatype=om.PTBatchNormMetatype,
            relu_metatype=om.PTRELUMetatype,
        ),
        "broken_output_edges_pattern",
    ],
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

        for from_node_key, to_node_key in ip_graph.edges:
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
        from nncf.torch.graph.operator_metatypes import PTModuleDepthwiseConv2dSubtype
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
            "ModelForMetatypeTesting/NNCFConvTranspose2d[conv_transpose]/conv_transpose2d_0": (
                PTModuleConvTranspose2dMetatype
            ),
            "ModelForMetatypeTesting/conv_transpose2d_0": PTConvTranspose2dMetatype,
            "ModelForMetatypeTesting/__add___0": PTAddMetatype,
            "ModelForMetatypeTesting/NNCFConv2d[conv_depthwise]/conv2d_0": PTModuleDepthwiseConv2dSubtype,
            "ModelForMetatypeTesting/conv2d_0": PTDepthwiseConv2dSubtype,
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
        nncf_network = NNCFNetwork(model, FillerInputInfo([FillerInputElement([1, 3, 300, 300])]))
        nncf_graph = nncf_network.nncf.get_original_graph()

        for nncf_node in nncf_graph.get_all_nodes():
            assert nncf_node.node_name in ref_scope_vs_metatype_dict
            ref_metatype = ref_scope_vs_metatype_dict[nncf_node.node_name]
            assert nncf_node.metatype == ref_metatype

    @pytest.mark.parametrize(
        ("mock_graph_factory", "dot_file_name"), MERGE_PATTERN_TEST_CASES, ids=[x[1] for x in MERGE_PATTERN_TEST_CASES]
    )
    def test_get_ip_graph_with_merged_operations(self, mock_graph_factory, dot_file_name):
        mock_graph = mock_graph_factory()
        ip_graph = get_ip_graph_for_test(mock_graph)
        pattern = PatternsManager.get_full_hw_pattern_graph(backend=BackendType.TORCH, device=TargetDevice.ANY)
        merged_ip_graph = ip_graph.get_ip_graph_with_merged_hw_optimized_operations(pattern)

        data_dir: Path = TEST_ROOT / "torch/data/reference_graphs/pattern_merging"

        path_to_dot_file = data_dir / "{}.dot".format(dot_file_name)

        if os.getenv("NNCF_TEST_REGEN_DOT") is not None:
            if not os.path.exists(str(data_dir)):
                os.makedirs(str(data_dir))
            graph_without_data = get_graph_without_data(merged_ip_graph)
            write_dot_graph(graph_without_data, str(path_to_dot_file))

        load_graph = read_dot_graph(str(path_to_dot_file))

        for key in load_graph.nodes:
            key.replace(r"\\n", r"\n")  # Somehow pydot mangles the \n characters while writing a .dot file

        sanitized_loaded_keys = [key.replace("\\n", "\n") for key in load_graph.nodes]
        sanitized_loaded_edges = [
            (u.replace("\\n", "\n"), v.replace("\\n", "\n")) for u, v in nx.DiGraph(load_graph).edges
        ]

        assert Counter(sanitized_loaded_keys) == Counter(list(merged_ip_graph.nodes))
        assert Counter(sanitized_loaded_edges) == Counter(list(merged_ip_graph.edges))


def test_extraction_model_transformations():
    model = wrap_model(InsertionPointTestModel(), torch.ones([1, 1, 10, 10]), trace_parameters=True)
    model_transformer = PTModelTransformer(model)

    command = PTModelExtractionCommand(
        ["InsertionPointTestModel/Conv2d[conv1]/conv2d_0"], ["InsertionPointTestModel/Conv2d[conv1]/conv2d_0"]
    )
    transformation_layout = PTTransformationLayout()
    transformation_layout.register(command)
    model_transformer.transform(transformation_layout)


@pytest.mark.parametrize(
    "command_cls,attr_name,new_value",
    [(PTBiasCorrectionCommand, "bias", torch.tensor([42.0])), (PTWeightUpdateCommand, "weight", torch.tensor([42.0]))],
)
def test_correction_transformations(command_cls, attr_name, new_value):
    model = wrap_model(InsertionPointTestModel(), torch.ones([1, 1, 10, 10]), trace_parameters=True)
    model_transformer = PTModelTransformer(model)

    target_point = PTTargetPoint(TargetType.LAYER, "InsertionPointTestModel/Conv2d[conv1]/conv2d_0")
    command = command_cls(target_point, new_value)

    transformation_layout = PTTransformationLayout()
    transformation_layout.register(command)
    updated_model = model_transformer.transform(transformation_layout)
    param = getattr(updated_model.conv1, attr_name)
    assert param.data == new_value


def test_rebuild_graph_after_insert_transformation():
    model = NNCFNetwork(InsertionPointTestModel(), FillerInputInfo([FillerInputElement([1, 1, 10, 10])]))

    graph = model.nncf.get_graph()

    command = PTInsertionCommand(
        PTTargetPoint(
            TargetType.OPERATION_WITH_WEIGHTS, target_node_name="InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0"
        ),
        AsymmetricQuantizer(PTQuantizerSpec(8, QuantizationMode.ASYMMETRIC, None, False, False, (1, 1, 1, 1), False)),
        TransformationPriority.QUANTIZATION_PRIORITY,
    )
    transformation_layout = PTTransformationLayout()
    transformation_layout.register(command)

    model_transformer = PTModelTransformer(model)
    transformed_model = model_transformer.transform(transformation_layout=transformation_layout)
    new_graph = transformed_model.nncf.get_graph()
    assert len(new_graph.get_all_nodes()) == len(graph.get_all_nodes()) + 1


class Hook(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._param = torch.nn.Parameter(torch.zeros((1,)))
        self.to_device = None

    def forward(self, x):
        return x + self._param

    def to(self, device):
        super().to(device)
        self.to_device = device


SHARED_FN_TARGET_POINTS = (
    PTTargetPoint(
        TargetType.OPERATOR_POST_HOOK,
        "/nncf_model_input_0",
    ),
    PTTargetPoint(
        TargetType.OPERATOR_PRE_HOOK,
        "InsertionPointTestModel/linear_0",
        input_port_id=0,
    ),
    PTTargetPoint(
        TargetType.OPERATION_WITH_WEIGHTS,
        "InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0",
    ),
)


@pytest.mark.parametrize("target_point", SHARED_FN_TARGET_POINTS)
def test_create_quantizer_insertion_command(target_point):
    hook = Hook()
    command = create_quantizer_insertion_command(target_point, hook)
    assert command.fn is hook
    quantizer_id = NonWeightQuantizerId(target_point.target_node_name, target_point.input_port_id)
    assert isinstance(command, PTSharedFnInsertionCommand)
    assert command.target_points == [target_point]
    assert command.fn is hook
    storage_key = str(quantizer_id)
    assert command.op_name == storage_key
    assert command.compression_module_type is ExtraCompressionModuleType.EXTERNAL_QUANTIZER


def test_create_shared_quantizer_insertion_command():
    ref_storage_key = (
        "/nncf_model_input_0|OUTPUT;"
        "InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0|OUTPUT;"
        "InsertionPointTestModel/linear_0|INPUT0"
    )
    hook = Hook()

    command = create_shared_quantizer_insertion_command(list(SHARED_FN_TARGET_POINTS), hook)
    assert command.fn is hook
    assert isinstance(command, PTSharedFnInsertionCommand)
    assert command.target_points == list(SHARED_FN_TARGET_POINTS)
    assert command.fn is hook
    assert command.op_name == ref_storage_key
    assert command.compression_module_type is ExtraCompressionModuleType.EXTERNAL_QUANTIZER


@pytest.mark.parametrize("compression_module_type", ExtraCompressionModuleType)
@pytest.mark.parametrize(
    "priority", [TransformationPriority.FP32_TENSOR_STATISTICS_OBSERVATION, TransformationPriority.DEFAULT_PRIORITY]
)
@pytest.mark.parametrize("compression_module_registered", [False, True])
@pytest.mark.parametrize("multidevice_model", (False, pytest.param(True, marks=pytest.mark.cuda)))
def test_shared_fn_insertion_point(
    priority, compression_module_registered, compression_module_type, multidevice_model, mocker
):
    if not torch.cuda.is_available() and multidevice_model:
        pytest.skip("Could not test multidevice case without cuda")

    tps = SHARED_FN_TARGET_POINTS
    OP_UNIQUE_NAME = "UNIQUE_NAME"
    HOOK_GROUP_NAME = "shared_commands_hooks_group"
    STORAGE_NAME = compression_module_type_to_attr_name(compression_module_type)
    hook_instance = Hook()

    def _insert_external_op_mocked():
        model = NNCFNetwork(InsertionPointTestModel(), FillerInputInfo([FillerInputElement([1, 1, 10, 10])]))
        model = model.cpu()
        if multidevice_model:
            model.conv1.to(torch.device("cpu"))
            model.conv2.to(torch.device("cuda"))

        if compression_module_registered:
            model.nncf.register_compression_module_type(compression_module_type)
        unique_name = f"{OP_UNIQUE_NAME}[{';'.join([tp.target_node_name for tp in tps])}]"
        command = PTSharedFnInsertionCommand(
            target_points=tps,
            fn=hook_instance,
            op_unique_name=unique_name,
            compression_module_type=compression_module_type,
            priority=priority,
            hooks_group_name=HOOK_GROUP_NAME,
        )
        transformation_layout = PTTransformationLayout()
        transformation_layout.register(command)

        mocker.MagicMock()
        mocker.patch(
            "nncf.torch.model_transformer.PTModelTransformer._apply_insertion_transformations",
            return_value=mocker.MagicMock(),
        )
        model_transformer = PTModelTransformer(model)
        model_transformer.transform(transformation_layout=transformation_layout)
        return model

    transformed_model = _insert_external_op_mocked()

    assert transformed_model.nncf.is_compression_module_registered(compression_module_type)

    REF_STORAGE_KEY = (
        "UNIQUE_NAME[/nncf_model_input_0;InsertionPointTestModel/linear_0;"
        "InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0]"
    )

    storage = getattr(transformed_model.nncf, STORAGE_NAME)
    assert storage[REF_STORAGE_KEY] is hook_instance
    assert hook_instance in transformed_model.modules()

    mock = PTModelTransformer._apply_insertion_transformations
    mock.assert_called_once()

    _, commands, device = mock.call_args.args
    assert len(commands) == len(tps)
    for command in commands:
        assert command.target_point in tps
        assert command.hooks_group_name == HOOK_GROUP_NAME
        assert command.priority == priority
        fn = command.fn
        assert isinstance(fn, ExternalOpCallHook)
        assert fn._storage_name == STORAGE_NAME
        assert fn._storage_key == REF_STORAGE_KEY

    if multidevice_model:
        assert hook_instance.to_device is None
        assert device is None
    else:
        actual_model_device = get_model_device(transformed_model)
        assert hook_instance.to_device == actual_model_device
        assert device == actual_model_device

    # Check torch can correctly save and load model state dict with an external quantizer
    state_dict = transformed_model.state_dict()
    assert f"_nncf.{STORAGE_NAME}.{REF_STORAGE_KEY}._param" in state_dict
    del transformed_model
    transformed_model = _insert_external_op_mocked()
    transformed_model.load_state_dict(state_dict)


@pytest.mark.parametrize(
    "priority", [TransformationPriority.FP32_TENSOR_STATISTICS_OBSERVATION, TransformationPriority.DEFAULT_PRIORITY]
)
@pytest.mark.parametrize("compression_module_registered", [False, True])
@pytest.mark.parametrize("multidevice_model", (False, pytest.param(True, marks=pytest.mark.cuda)))
def test_shared_fn_insertion_command_several_module_types(
    priority, compression_module_registered, multidevice_model, mocker
):
    if not torch.cuda.is_available() and multidevice_model:
        pytest.skip("Could not test multidevice case without cuda")

    tps = SHARED_FN_TARGET_POINTS
    OP_UNIQUE_NAME = "UNIQUE_NAME"
    HOOK_GROUP_NAME = "shared_commands_hooks_group"
    MODULE_TYPES = [t for t in ExtraCompressionModuleType]
    hook_instance = Hook()

    def _insert_external_op_mocked():
        model = NNCFNetwork(InsertionPointTestModel(), FillerInputInfo([FillerInputElement([1, 1, 10, 10])]))
        model = model.cpu()
        if multidevice_model:
            model.conv1.to(torch.device("cpu"))
            model.conv2.to(torch.device("cuda"))

        transformation_layout = PTTransformationLayout()
        for compression_module_type in MODULE_TYPES:
            if compression_module_registered:
                model.nncf.register_compression_module_type(compression_module_type)
            unique_name = f"{OP_UNIQUE_NAME}[{';'.join([tp.target_node_name for tp in tps])}]"
            command = PTSharedFnInsertionCommand(
                target_points=tps,
                fn=hook_instance,
                op_unique_name=unique_name,
                compression_module_type=compression_module_type,
                priority=priority,
                hooks_group_name=HOOK_GROUP_NAME,
            )
            transformation_layout.register(command)

        mocker.MagicMock()
        mocker.patch(
            "nncf.torch.model_transformer.PTModelTransformer._apply_shared_node_insertion_with_compression_type",
            return_value=mocker.MagicMock(),
        )
        model_transformer = PTModelTransformer(model)
        model_transformer.transform(transformation_layout=transformation_layout)
        return model

    transformed_model = _insert_external_op_mocked()

    mock = PTModelTransformer._apply_shared_node_insertion_with_compression_type
    assert len(mock.call_args_list) == len(MODULE_TYPES)

    REF_STORAGE_KEY = (
        "UNIQUE_NAME[/nncf_model_input_0;InsertionPointTestModel/linear_0;"
        "InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0]"
    )

    module_types_set = set(MODULE_TYPES)
    for (_, commands, device, compression_module_type), _ in mock.call_args_list:
        module_types_set -= set((compression_module_type,))
        assert len(commands) == 1
        command = commands[0]
        assert isinstance(command, PTSharedFnInsertionCommand)
        assert command.fn is hook_instance
        assert command.target_points is tps
        assert command.compression_module_type == compression_module_type
        assert command.op_name == REF_STORAGE_KEY
        assert command.priority == priority
        assert command.hooks_group_name == HOOK_GROUP_NAME

        if multidevice_model:
            assert device is None
        else:
            assert device == get_model_device(transformed_model)

    assert not module_types_set


INSERTION_POINT_TEST_MODEL_TARGET_POINTS = (
    (
        TargetType.OPERATOR_POST_HOOK,
        "/nncf_model_input_0",
        None,
    ),
    (
        TargetType.OPERATOR_PRE_HOOK,
        "InsertionPointTestModel/linear_0",
        0,
    ),
    (
        TargetType.OPERATION_WITH_WEIGHTS,
        "InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0",
        None,
    ),
)


@pytest.mark.parametrize("target_type, node_name, input_port_id", INSERTION_POINT_TEST_MODEL_TARGET_POINTS)
def test_successive_insertion_transformation(target_type, node_name, input_port_id):
    model = NNCFNetwork(InsertionPointTestModel(), FillerInputInfo([FillerInputElement([1, 1, 10, 10])]))

    target_point = PTTargetPoint(target_type, node_name, input_port_id=input_port_id)
    transformed_model = model
    ops = [BaseOp(lambda x: x), BaseOp(lambda x: x)]
    for op in ops:
        command = PTInsertionCommand(target_point, op)

        model_transformer = PTModelTransformer(transformed_model)
        transformation_layout = PTTransformationLayout()
        transformation_layout.register(command)
        transformed_model = model_transformer.transform(transformation_layout)
        transformed_model.nncf.rebuild_graph()

    checker = HookChecker(transformed_model, "conv1")
    checker.add_ref(
        ref_hooks=ops,
        target_type=target_type,
        target_node_name=node_name,
        input_port_id=input_port_id,
    )
    checker.check_with_reference()


GLOBAL_LIST = []


def get_dummy_op(op_id):
    @register_operator()
    def dummy_op(x):
        GLOBAL_LIST.append(op_id)
        return x

    return dummy_op


@register_module()
class DummyModule(torch.nn.Module):
    def __init__(self, module_id):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((1,)))
        self._op = get_dummy_op(module_id)

    def forward(self, x):
        return self._op(x)


def get_model_to_test_nested_modules():
    class TestModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.op1 = get_dummy_op("op1")
            self.m = DummyModule("DummyModule")
            self.op2 = get_dummy_op("op2")

        def forward(self, x):
            x = self.op1(x)
            x = self.m(x)
            x = self.op2(x)

    return TestModel()


@dataclass
class NestedHooksTestCase:
    target_type: TargetType
    target_node_name: str
    input_port_id: int


NESTED_HOOKS_TEST_CASES = [
    NestedHooksTestCase(
        TargetType.OPERATOR_POST_HOOK,
        "/nncf_model_input_0",
        None,
    ),
    NestedHooksTestCase(
        TargetType.OPERATOR_PRE_HOOK,
        "TestModel/dummy_op_1",
        0,
    ),
    NestedHooksTestCase(
        TargetType.OPERATION_WITH_WEIGHTS,
        "TestModel/NNCFUserDummyModule[m]/dummy_op_0",
        None,
    ),
]

REFS_DEPTH_0 = ["op1", "DummyModule", "op2"]


REFS_DEPTH_1 = [
    ["pre_hook_1", "op1", "DummyModule", "op2"],
    ["op1", "DummyModule", "pre_hook_1", "op2"],
    ["op1", "pre_hook_1", "DummyModule", "op2"],
]


REFS_DEPTH_2 = [
    ["pre_hook_0", "pre_hook_1", "pre_hook_2", "op1", "DummyModule", "op2"],
    ["op1", "DummyModule", "pre_hook_0", "pre_hook_1", "pre_hook_2", "op2"],
    ["op1", "pre_hook_0", "pre_hook_1", "pre_hook_2", "DummyModule", "op2"],
]


def _add_and_test_hook_depth_one(test_params: NestedHooksTestCase, pre_hook_op, ref):
    model = NNCFNetwork(get_model_to_test_nested_modules(), FillerInputInfo([FillerInputElement([10])]))

    # Check test model is working as expected
    GLOBAL_LIST.clear()
    model.nncf.rebuild_graph()
    assert GLOBAL_LIST == REFS_DEPTH_0

    target_point = PTTargetPoint(
        test_params.target_type, test_params.target_node_name, input_port_id=test_params.input_port_id
    )
    transformed_model = model

    pre_hook_1 = pre_hook_op("pre_hook_1")
    command = PTInsertionCommand(target_point, pre_hook_1)

    model_transformer = PTModelTransformer(transformed_model)
    transformation_layout = PTTransformationLayout()
    transformation_layout.register(command)
    transformed_model = model_transformer.transform(transformation_layout)

    # Check order of calls
    GLOBAL_LIST.clear()
    transformed_model.nncf.rebuild_graph()
    assert ref == GLOBAL_LIST

    # Check hooks are kept correctly
    checker = HookChecker(transformed_model, "m")
    checker.add_ref(
        ref_hooks=[pre_hook_1],
        target_type=test_params.target_type,
        target_node_name=test_params.target_node_name,
        input_port_id=test_params.input_port_id,
    )
    checker.check_with_reference()

    graph = transformed_model.nncf.get_graph()
    target_node = graph.get_node_by_name(test_params.target_node_name)
    if test_params.target_type == TargetType.OPERATOR_POST_HOOK:
        target_node = graph.get_next_nodes(target_node)[0]
    elif test_params.target_type == TargetType.OPERATOR_PRE_HOOK:
        target_node = graph.get_previous_nodes(target_node)[0]
    else:
        target_node = graph.get_node_by_id(2)

    return transformed_model, target_node, pre_hook_1


@pytest.mark.parametrize(
    "test_params, ref_depth_one, ref_depth_two", zip(NESTED_HOOKS_TEST_CASES, REFS_DEPTH_1, REFS_DEPTH_2)
)
@pytest.mark.parametrize("pre_hook_op", [get_dummy_op, DummyModule], ids=["op_hook", "module_hook"])
def test_nested_pre_post_hooks(test_params: NestedHooksTestCase, ref_depth_one, ref_depth_two, pre_hook_op):
    transformed_model, target_node, pre_hook_1 = _add_and_test_hook_depth_one(test_params, pre_hook_op, ref_depth_one)

    transformation_layout = PTTransformationLayout()
    nested_pre_hooks = []
    for i, target_type_ in enumerate([TargetType.OPERATOR_PRE_HOOK, TargetType.OPERATOR_POST_HOOK]):
        target_point_on_hook = PTTargetPoint(target_type_, target_node.node_name, input_port_id=0)
        nested_pre_hooks.append(pre_hook_op(f"pre_hook_{i * 2}"))
        transformation_layout.register(PTInsertionCommand(target_point_on_hook, nested_pre_hooks[-1]))
    model_transformer = PTModelTransformer(transformed_model)
    model_with_nested_hooks = model_transformer.transform(transformation_layout)

    # Check order of calls
    GLOBAL_LIST.clear()
    model_with_nested_hooks.nncf.rebuild_graph()
    assert ref_depth_two == GLOBAL_LIST

    # Check hooks are kept correctly
    checker = HookChecker(model_with_nested_hooks, "m")
    checker.add_ref(
        ref_hooks=[pre_hook_1],
        target_type=test_params.target_type,
        target_node_name=test_params.target_node_name,
        input_port_id=test_params.input_port_id,
    )
    checker.add_ref(
        ref_hooks=[nested_pre_hooks[0]],
        target_type=TargetType.OPERATOR_PRE_HOOK,
        target_node_name=target_node.node_name,
        input_port_id=0,
    )
    checker.add_ref(
        ref_hooks=[nested_pre_hooks[1]],
        target_type=TargetType.OPERATOR_POST_HOOK,
        target_node_name=target_node.node_name,
        input_port_id=0,
    )
    checker.check_with_reference()


@pytest.mark.skip("Nested weight hooks are not supported yet.")
@pytest.mark.parametrize("test_params, ref_depth_one", zip(NESTED_HOOKS_TEST_CASES, REFS_DEPTH_1))
def test_nested_weight_hooks(test_params: NestedHooksTestCase, ref_depth_one):
    transformed_model, target_node, _ = _add_and_test_hook_depth_one(test_params, DummyModule, ref_depth_one)

    transformation_layout = PTTransformationLayout()
    target_point_on_hook = PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, target_node.node_name, input_port_id=0)
    transformation_layout.register(PTInsertionCommand(target_point_on_hook, DummyModule("pre_hook_3")))
    model_transformer = PTModelTransformer(transformed_model)
    model_with_nested_hooks = model_transformer.transform(transformation_layout)
    GLOBAL_LIST.clear()
    model_with_nested_hooks.nncf.rebuild_graph()
    # TODO: Add refeference check when nested weight hooks will be supported
    # assert GLOBAL_LIST == ref_depth_two
