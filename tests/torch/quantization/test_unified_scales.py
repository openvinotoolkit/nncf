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

import itertools
from collections import Counter
from functools import partial
from typing import Dict, List

import onnx
import pytest
import torch
import torch.nn

import nncf
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfigType
from nncf.common.quantization.quantizer_propagation.solver import QuantizerPropagationSolver
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.quantization.layers import AsymmetricQuantizer
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import get_nodes_by_type
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.helpers import resolve_constant_node_inputs_to_values
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init
from tests.torch.quantization.test_onnx_export import get_successors


def make_op_address_for_coalescing_test(scope_str: str) -> OperationAddress:
    op_address = OperationAddress.from_str(scope_str)
    return op_address


def make_insertion_point_for_coalescing_test(node_name: NNCFNodeName, input_port_id: int = None) -> PTTargetPoint:
    retval = PTTargetPoint(TargetType.OPERATOR_POST_HOOK, target_node_name=node_name, input_port_id=input_port_id)
    return retval


@pytest.mark.parametrize(
    "input_insertion_points, linked_scopes_groups_list, ref_coalesced_ip_lists",
    # ref_coalesced_ip_lists == None means that the coalescing should raise an exception
    [
        # 0 - Empty linked scopes list
        (
            [
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=1),
            ],
            [],
            # Each coalesced list has one entry
            [
                [
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=1),
                ],
            ],
        ),
        # 1 - Linked scope only affects 1 operation
        (
            [
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0", input_port_id=0),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0"),
            ],
            [["Foo/Baz[bar]/conv2d_0"]],
            # Each coalesced list has one entry
            [
                [
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0", input_port_id=0),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0"),
                ],
            ],
        ),
        # 2 - Same as 1 but with multiple groups
        (
            [
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0", input_port_id=0),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=1),
            ],
            [["Foo/Baz[bar]/conv2d_0"], ["Foo/Xyz[leet]/__add___0"]],
            # Each coalesced list has one entry again
            [
                [
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0", input_port_id=0),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=1),
                ],
            ],
        ),
        # 3 - Single group affecting some of the scopes
        (
            [
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0", input_port_id=0),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=1),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/matmul_0", input_port_id=1),
            ],
            [["Foo/Xyz[leet]/matmul_0", "Foo/Xyz[leet]/__add___0", "Foo/Baz[bar]/linear_0"]],
            [
                [
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/matmul_0", input_port_id=1),
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0", input_port_id=0),
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=1),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                ],
            ],
        ),
        # 4 - Multiple groups, each affecting one operation
        (
            [
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0", input_port_id=0),
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0"),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=0),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/matmul_0", input_port_id=0),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_0"),
            ],
            [["Foo/Baz[bar]/linear_0"], ["Foo/Asdf[jkl]/softmax_0"]],
            [
                # Each coalesced list has one entry again
                [
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0"),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_0"),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0", input_port_id=0),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=0),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/matmul_0", input_port_id=0),
                ],
            ],
        ),
        # 5 - Multiple groups affecting multiple operations without overlapping
        (
            [
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0", input_port_id=0),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=1),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/matmul_0"),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_0"),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_1", input_port_id=0),
            ],
            [
                ["Foo/Baz[bar]/conv2d_0", "Foo/Baz[bar]/linear_0"],
                ["Foo/Asdf[jkl]/softmax_1", "Foo/Xyz[leet]/__add___0"],
            ],
            [
                [
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0", input_port_id=0),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_1", input_port_id=0),
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0", input_port_id=1),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/matmul_0"),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_0"),
                ],
            ],
        ),
        # 6 - A variation of 5
        (
            [
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0"),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0"),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/matmul_0"),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_0"),
                make_insertion_point_for_coalescing_test(
                    "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0",
                    input_port_id=0,
                ),
            ],
            [
                ["Foo/Baz[bar]/conv2d_0", "Foo/Baz[bar]/linear_0", "Foo/Xyz[leet]/matmul_0"],
                ["Foo/Asdf[jkl]/softmax_0", "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"],
            ],
            [
                [
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                    make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0"),
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/matmul_0"),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_0"),
                    make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0", input_port_id=0),
                ],
                [
                    make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0"),
                ],
            ],
        ),
        # 7 - Overlapping groups
        (
            [
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0"),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0"),
                make_insertion_point_for_coalescing_test(
                    "Foo/Xyz[leet]/matmul_0",
                    input_port_id=1,
                ),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_0"),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"),
            ],
            [
                ["Foo/Baz[bar]/conv2d_0", "Foo/Baz[bar]/linear_0", "Foo/Xyz[leet]/matmul_0"],
                ["Foo/Xyz[leet]/matmul_0", "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"],
            ],
            None,
        ),
        # 8 - More than 1 match for the operation specified in the group
        (
            [
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/conv2d_0"),
                make_insertion_point_for_coalescing_test(
                    "Foo/Baz[bar]/conv2d_0",
                    input_port_id=0,
                ),
                make_insertion_point_for_coalescing_test(
                    "Foo/Baz[bar]/linear_0",
                ),
                make_insertion_point_for_coalescing_test(
                    "Foo/Xyz[leet]/__add___0",
                ),
                make_insertion_point_for_coalescing_test(
                    "Foo/Xyz[leet]/matmul_0",
                    input_port_id=1,
                ),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_0"),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"),
            ],
            [
                ["Foo/Baz[bar]/conv2d_0", "Foo/Xyz[leet]/matmul_0"],
                ["Foo/Xyz[leet]/matmul_0", "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"],
            ],
            None,
        ),
        # 9 - No match for an operation specified in the group
        (
            [
                make_insertion_point_for_coalescing_test(
                    "Foo/Baz[bar]/conv2d_0",
                    input_port_id=0,
                ),
                make_insertion_point_for_coalescing_test("Foo/Baz[bar]/linear_0"),
                make_insertion_point_for_coalescing_test("Foo/Xyz[leet]/__add___0"),
                make_insertion_point_for_coalescing_test(
                    "Foo/Xyz[leet]/matmul_0",
                    input_port_id=1,
                ),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/softmax_0"),
                make_insertion_point_for_coalescing_test("Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"),
            ],
            [
                ["Foo/Baz[bar]/conv2d_0", "Foo/Xyz[leet]/matmul_1"],
                ["Foo/Xyz[leet]/matmul_0", "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"],
            ],
            None,
        ),
    ],
)
def test_insertion_point_coalescing(
    input_insertion_points: List[PTTargetPoint],
    linked_scopes_groups_list: List[List[str]],
    ref_coalesced_ip_lists: List[List[PTTargetPoint]],
):
    if ref_coalesced_ip_lists is None:
        with pytest.raises((nncf.InternalError, nncf.ValidationError)):
            _ = QuantizerPropagationSolver.coalesce_insertion_points(input_insertion_points, linked_scopes_groups_list)
    else:
        test_coalesced_ip_lists = QuantizerPropagationSolver.coalesce_insertion_points(
            input_insertion_points, linked_scopes_groups_list
        )
        assert len(test_coalesced_ip_lists) == len(ref_coalesced_ip_lists)
        for idx, test_list in enumerate(test_coalesced_ip_lists):
            assert Counter(test_list) == Counter(ref_coalesced_ip_lists[idx])


class EltwiseQuantizerLinkingTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Path(torch.nn.Module):
            def forward(self, input_1, input_2):
                retval0 = input_1 + input_2
                retval1 = retval0 * input_2
                retval2 = retval0 + retval1
                # __add___0, __mul___0, __add___1 results respectively
                return retval0, retval1, retval2

        self.path1 = Path()
        self.path2 = Path()

    def forward(self, input_1, input_2):
        path1_results = self.path1(input_1, input_2)
        path2_results = self.path2(input_1, input_2)
        return tuple(x + y for x, y in zip(path1_results, path2_results))


def test_quantizer_scale_linking(mocker):
    nncf_config = get_quantization_config_without_range_init(model_size=1)
    nncf_config["input_info"] = [
        {
            "sample_size": [1, 1, 1, 1],
        },
        {
            "sample_size": [1, 1, 1, 1],
        },
    ]
    nncf_config["compression"]["activations"] = {
        "unified_scale_ops": [
            [
                # Note: Assuming that quantizers are attached as a post-op to the specified operation
                "EltwiseQuantizerLinkingTestModel/Path[path2]/__mul___0",
                "EltwiseQuantizerLinkingTestModel/Path[path2]/__add___0",
            ]
        ],
        "ignored_scopes": [
            # Ignore path output averaging operations
            "EltwiseQuantizerLinkingTestModel/__add___0",
            "EltwiseQuantizerLinkingTestModel/__add___1",
            "EltwiseQuantizerLinkingTestModel/__add___2",
        ],
    }
    register_bn_adaptation_init_args(nncf_config)

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
        EltwiseQuantizerLinkingTestModel(), nncf_config
    )

    # 18 inputs to quantize (14 regular + 4 linked),
    # 8 quantization points left after propagation, out of these 3 are linked
    assert len(compression_ctrl.non_weight_quantizers) == 6

    shared_quantizer_id = NonWeightQuantizerId(target_node_name="/nncf_model_input_0")

    non_shared_spies = []
    for aq_id, aq_info in compression_ctrl.non_weight_quantizers.items():
        quantizer = aq_info.quantizer_module_ref
        spy = mocker.spy(quantizer, "forward")
        if aq_id == shared_quantizer_id:
            shared_spy = spy
        else:
            non_shared_spies.append(spy)

    test_input1 = torch.ones([1, 1, 1, 1])
    test_input2 = 2 * test_input1
    compressed_model(test_input1, test_input2)

    assert shared_spy.call_count == 3
    for non_shared_spy in non_shared_spies:
        assert non_shared_spy.call_count == 1


def test_eltwise_unified_scales_for_npu():
    nncf_config = get_quantization_config_without_range_init(model_size=1)
    nncf_config["input_info"] = [
        {
            "sample_size": [1, 1, 1, 1],
        },
        {
            "sample_size": [1, 1, 1, 1],
        },
    ]
    nncf_config["target_device"] = "NPU"
    register_bn_adaptation_init_args(nncf_config)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(EltwiseQuantizerLinkingTestModel(), nncf_config)

    assert len(compression_ctrl.non_weight_quantizers) == 2

    total_quantizations = sum(len(info.affected_insertions) for info in compression_ctrl.non_weight_quantizers.values())
    assert total_quantizations == 8


class SingleCatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 1)

    def forward(self, x, y):
        x = x * x
        y = y * y
        z = torch.cat([x, y])
        v = self.conv(z)
        return v


class DoubleCatModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 1)

    def forward(self, x, y):
        x = x * x
        y = y * y
        z = torch.cat([x, y])
        v = torch.cat([x, z])
        w = self.conv(v)
        return w


class UNetLikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(4, 8, 1)
        self.conv_2 = torch.nn.Conv2d(8, 16, 1)
        self.conv_3 = torch.nn.Conv2d(16, 32, 1)
        self.conv_t_3 = torch.nn.ConvTranspose2d(32, 16, 1)
        self.conv_t_2 = torch.nn.ConvTranspose2d(16, 8, 1)
        self.conv_t_1 = torch.nn.ConvTranspose2d(8, 4, 1)

    def forward(self, x, y):
        y1 = self.conv_1(x)
        y2 = self.conv_2(y1)
        y3 = self.conv_3(y2)
        z3 = self.conv_t_3(y3)
        z3 = torch.cat([z3, y2])
        z2 = self.conv_t_2(z3)
        z2 = torch.cat([z2, y1])
        z1 = self.conv_t_1(z2)
        return z1


CAT_UNIFIED_SCALE_TEST_STRUCTS = [(SingleCatModel, 3, 4), (DoubleCatModel, 3, 4), (UNetLikeModel, 4, 6)]


@pytest.mark.parametrize(
    "target_device, model_creator, ref_aq_module_count, ref_quantizations",
    [
        (t_dev,) + rest
        for t_dev, rest in itertools.product([x.value for x in HWConfigType], CAT_UNIFIED_SCALE_TEST_STRUCTS)
    ],
)
def test_unified_scales_with_concat(target_device, model_creator, ref_aq_module_count, ref_quantizations):
    nncf_config = get_quantization_config_without_range_init(model_size=1)
    nncf_config["input_info"] = [
        {
            "sample_size": [1, 4, 1, 1],
        },
        {
            "sample_size": [1, 4, 1, 1],
        },
    ]

    nncf_config["target_device"] = target_device
    register_bn_adaptation_init_args(nncf_config)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(model_creator(), nncf_config)

    assert len(compression_ctrl.non_weight_quantizers) == ref_aq_module_count

    total_quantizations = sum(len(info.affected_insertions) for info in compression_ctrl.non_weight_quantizers.values())
    assert total_quantizations == ref_quantizations


class SimplerModelForUnifiedScalesTesting(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2d_2 = torch.nn.Conv2d(1, 1, 1)
        self.conv2d_3 = torch.nn.Conv2d(1, 1, 1)
        self.conv2d_4 = torch.nn.Conv2d(1, 1, 1)
        self.conv2d_5 = torch.nn.Conv2d(1, 1, 1)
        self.conv2d_6 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        in_1, in_2 = x.chunk(dim=-1, chunks=2)
        in_1 = self.conv2d_1(in_1)
        in_2 = self.conv2d_2(in_2)
        x = in_1 + in_2
        x = torch.stack([x, x], dim=-1)
        x = x.squeeze(dim=0)
        in1, in2 = x.chunk(dim=-1, chunks=2)
        in1 = self.conv2d_3(in1)
        in2 = self.conv2d_3(in2)
        x = torch.cat([in1, in2], dim=-1)
        in_1, in_2 = x.chunk(dim=-1, chunks=2)
        in_1 = self.conv2d_5(in_1)
        in_2 = self.conv2d_6(in_2)
        x = in_1 * in_2
        return x


class TwoEmbeddingAddModel(torch.nn.Module):
    EMBEDDING_IO_SHAPE = [10, 10]

    def __init__(self):
        super().__init__()
        self.embedding1 = torch.nn.Embedding(*self.EMBEDDING_IO_SHAPE)
        self.embedding2 = torch.nn.Embedding(*self.EMBEDDING_IO_SHAPE)

    def forward(self, x):
        y1 = self.embedding1(x)
        y2 = self.embedding2(x)
        return y1 + y2


class TestsWithONNXInspection:
    @staticmethod
    def get_fq_nodes(onnx_model: onnx.ModelProto) -> List[onnx.NodeProto]:
        return get_nodes_by_type(onnx_model, "FakeQuantize")

    @staticmethod
    def immediately_dominates_add_or_mul(node: onnx.NodeProto, graph: onnx.GraphProto) -> bool:
        if len(node.output) != 1:
            return False
        output_tensor_id = node.output[0]
        matches = [x for x in graph.node if output_tensor_id in x.input]
        for match in matches:
            if match.op_type in ["Add", "Mul"]:
                return True
        return False

    @staticmethod
    def immediately_dominates_cat(node: onnx.NodeProto, graph: onnx.GraphProto) -> bool:
        if len(node.output) != 1:
            return False
        output_tensor_id = node.output[0]
        matches = [x for x in graph.node if output_tensor_id in x.input]
        for match in matches:
            if match.op_type in ["Concat"]:
                return True
        return False

    @staticmethod
    def immediately_dominates_embedding(node: onnx.NodeProto, graph: onnx.GraphProto) -> bool:
        if len(node.output) != 1:
            return False
        output_tensor_id = node.output[0]
        matches = [x for x in graph.node if output_tensor_id in x.input]
        for match in matches:
            if match.op_type in ["Gather"]:
                return True
        return False

    @staticmethod
    def group_nodes_by_output_target(nodes: List[onnx.NodeProto], graph: onnx.GraphProto) -> List[List[onnx.NodeProto]]:
        output_nodes: Dict[str, List[onnx.NodeProto]] = {}
        for node in nodes:
            succs = get_successors(node, graph)
            assert len(succs) == 1
            target_node_name = next(iter(succs)).name
            if target_node_name not in output_nodes:
                output_nodes[target_node_name] = []
            output_nodes[target_node_name].append(node)
        return list(output_nodes.values())

    def test_unified_scales_are_identical_in_onnx(self, tmp_path):
        nncf_config = get_quantization_config_without_range_init(model_size=1)
        nncf_config["compression"]["quantize_outputs"] = True
        nncf_config["input_info"] = [
            {
                "sample_size": [1, 1, 1, 2],
            },
        ]
        nncf_config["target_device"] = "NPU"
        register_bn_adaptation_init_args(nncf_config)

        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
            SimplerModelForUnifiedScalesTesting(), nncf_config
        )

        with torch.no_grad():
            for quant_info in compression_ctrl.non_weight_quantizers.values():
                if isinstance(quant_info.quantizer_module_ref, AsymmetricQuantizer):
                    quant_info.quantizer_module_ref.input_range *= torch.abs(
                        torch.rand_like(quant_info.quantizer_module_ref.input_range)
                    )
                else:
                    quant_info.quantizer_module_ref.scale *= torch.abs(
                        torch.rand_like(quant_info.quantizer_module_ref.scale)
                    )

        test_input1 = torch.ones([1, 1, 1, 2])
        compressed_model.forward(test_input1)

        onnx_path = str(tmp_path / "model.onnx")
        # Exporting the operator ::chunk to ONNX opset version 9 is not supported.
        # Support for this operator was added in version 11
        compression_ctrl.export_model(onnx_path, save_format="onnx_11")

        onnx_model = onnx.load(onnx_path)

        fq_nodes = TestsWithONNXInspection.get_fq_nodes(onnx_model)
        eltwise_dominator_predicate = partial(
            TestsWithONNXInspection.immediately_dominates_add_or_mul, graph=onnx_model.graph
        )
        eltwise_fq_nodes = list(filter(eltwise_dominator_predicate, fq_nodes))

        cat_dominator_predicate = partial(TestsWithONNXInspection.immediately_dominates_cat, graph=onnx_model.graph)
        cat_fq_nodes = list(filter(cat_dominator_predicate, fq_nodes))

        fq_nodes_grouped_by_output = TestsWithONNXInspection.group_nodes_by_output_target(
            eltwise_fq_nodes + cat_fq_nodes, onnx_model.graph
        )

        for unified_scale_group in fq_nodes_grouped_by_output:
            inputs = [
                resolve_constant_node_inputs_to_values(fq_node, onnx_model.graph) for fq_node in unified_scale_group
            ]
            for inputs_dict in inputs[1:]:
                curr_values = list(inputs_dict.values())
                ref_values = list(inputs[0].values())
                assert curr_values == ref_values  # All inputs for unified scale quantizers must be equal

    def test_weight_and_act_quantizer_scale_unification(self, tmp_path):
        nncf_config = get_quantization_config_without_range_init(model_size=1)
        nncf_config["input_info"] = [
            {"sample_size": [1, 5], "type": "long", "filler": "zeros"},
        ]
        nncf_config["target_device"] = "NPU"
        register_bn_adaptation_init_args(nncf_config)

        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
            TwoEmbeddingAddModel(), nncf_config
        )

        with torch.no_grad():
            for quant_module in compression_ctrl.all_quantizations.values():
                if isinstance(quant_module, AsymmetricQuantizer):
                    quant_module.input_range *= torch.abs(torch.rand_like(quant_module.input_range))
                else:
                    quant_module.scale *= torch.abs(torch.rand_like(quant_module.scale))

        test_input1 = torch.ones([1, 5], dtype=torch.long)
        compressed_model.forward(test_input1)
        onnx_path = str(tmp_path / "model.onnx")
        compression_ctrl.export_model(onnx_path)

        onnx_model = onnx.load(onnx_path)

        fq_nodes = TestsWithONNXInspection.get_fq_nodes(onnx_model)
        eltwise_dominator_predicate = partial(
            TestsWithONNXInspection.immediately_dominates_add_or_mul, graph=onnx_model.graph
        )
        embedding_dominator_predicate = partial(
            TestsWithONNXInspection.immediately_dominates_embedding, graph=onnx_model.graph
        )
        eltwise_fq_nodes = list(filter(eltwise_dominator_predicate, fq_nodes))
        embedding_weight_fq_nodes = list(filter(embedding_dominator_predicate, fq_nodes))

        fq_nodes_with_expected_unified_scales = embedding_weight_fq_nodes + eltwise_fq_nodes

        unified_fq_node_inputs = [
            resolve_constant_node_inputs_to_values(fq_node, onnx_model.graph)
            for fq_node in fq_nodes_with_expected_unified_scales
        ]

        # delete weights from input dict
        for inputs_for_single_fq in unified_fq_node_inputs:
            weight_input_names = []
            for input_name, input_tensor in inputs_for_single_fq.items():
                if list(input_tensor.shape) == TwoEmbeddingAddModel.EMBEDDING_IO_SHAPE:
                    weight_input_names.append(input_name)
            for weight_input_name in weight_input_names:
                inputs_for_single_fq.pop(weight_input_name)

        ref_values = list(unified_fq_node_inputs[0].values())
        for inputs_dict in unified_fq_node_inputs[1:]:
            curr_values = list(inputs_dict.values())
            assert curr_values == ref_values  # All inputs for unified scale quantizers must be equal


class SharedEmbeddingAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_embedding = torch.nn.Embedding(10, 10)

    def forward(self, x):
        y1 = self.shared_embedding(x)
        y2 = self.shared_embedding(x)
        return y1 + y2


def test_unified_scales_with_shared_nodes():
    nncf_config = get_quantization_config_without_range_init(model_size=1)
    nncf_config["input_info"] = [
        {"sample_size": [1, 5], "type": "long", "filler": "zeros"},
    ]
    nncf_config["target_device"] = "NPU"
    register_bn_adaptation_init_args(nncf_config)

    _, compression_ctrl = create_compressed_model_and_algo_for_test(
        SharedEmbeddingAddModel(), nncf_config
    )  # type: NNCFNetwork, QuantizationController

    assert len(compression_ctrl.weight_quantizers) == 1  # The two embedding nodes point to a single shared layer
    assert len(compression_ctrl.non_weight_quantizers) == 0  # The "add" operation has its inputs already quantized
