# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import os
from collections import Counter
from pathlib import Path
from typing import List

import networkx as nx
import pytest
import torch
import torch.nn.functional as F
from torch import nn

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
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.utils.backend import BackendType
from nncf.common.utils.dot_file_rw import get_graph_without_data
from nncf.common.utils.dot_file_rw import read_dot_graph
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.torch.dynamic_graph.context import PreHookId
from nncf.torch.dynamic_graph.graph_tracer import ModelInputInfo
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTInputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTOutputNoopMetatype
from nncf.torch.graph.operator_metatypes import PTReshapeMetatype
from nncf.torch.graph.transformations.commands import PTBiasCorrectionCommand
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTModelExtractionWithFusedBiasCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.layout import PTTransformationLayout
from nncf.torch.layers import NNCFConv2d
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.module_operations import BaseOp
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.nncf_network import PTInsertionPoint
from nncf.torch.nncf_network import PTInsertionType
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from tests.common.quantization.mock_graphs import get_ip_graph_for_test
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_broken_output_edge_pattern
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_mergeable_pattern
from tests.common.quantization.mock_graphs import get_mock_model_graph_with_no_mergeable_pattern
from tests.common.quantization.mock_graphs import get_nncf_graph_from_mock_nx_graph
from tests.common.quantization.mock_graphs import get_two_branch_mock_model_graph
from tests.shared.paths import TEST_ROOT


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

        if insertion_type == TargetType.OPERATION_WITH_WEIGHTS:
            hook1 = BaseOp(lambda x: x)
            hook2 = BaseOp(lambda x: 2 * x)
            hook3 = BaseOp(lambda x: 3 * x)
        elif insertion_type == TargetType.POST_LAYER_OPERATION:
            hook1 = BaseOp(lambda m, x: x)
            hook2 = BaseOp(lambda m, x: 2 * x)
            hook3 = BaseOp(lambda m, x: 3 * x)
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
            "ModelForMetatypeTesting/NNCFConvTranspose2d[conv_transpose]/conv_transpose2d_0": (
                PTModuleConvTranspose2dMetatype
            ),
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
        pattern = PatternsManager.get_full_pattern_graph(backend=BackendType.TORCH, device=TargetDevice.ANY)
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


def test_extraction_with_fused_bias_transformations():
    model = NNCFNetwork(InsertionPointTestModel(), [ModelInputInfo([1, 1, 10, 10])])
    model_transformer = PTModelTransformer(model)

    command = PTModelExtractionWithFusedBiasCommand("InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0")
    transformation_layout = PTTransformationLayout()
    transformation_layout.register(command)
    extracted_model = model_transformer.transform(transformation_layout)

    assert isinstance(extracted_model, nn.Sequential)
    assert len(extracted_model) == 1
    assert isinstance(extracted_model[0], NNCFConv2d)


def test_bias_correction_transformations():
    model = NNCFNetwork(InsertionPointTestModel(), [ModelInputInfo([1, 1, 10, 10])])
    model_transformer = PTModelTransformer(model)

    new_bias = torch.Tensor([42])

    target_point = PTTargetPoint(TargetType.LAYER, "InsertionPointTestModel/NNCFConv2d[conv1]/conv2d_0")
    command = PTBiasCorrectionCommand(target_point, new_bias)

    transformation_layout = PTTransformationLayout()
    transformation_layout.register(command)
    updated_model = model_transformer.transform(transformation_layout)
    assert updated_model.conv1.bias.data == new_bias


def test_rebuild_graph_after_insert_transformation():
    model = NNCFNetwork(InsertionPointTestModel(), [ModelInputInfo([1, 1, 10, 10])])

    graph = model.get_graph()

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
    new_graph = transformed_model.get_graph()
    assert len(new_graph.get_all_nodes()) == len(graph.get_all_nodes()) + 1
