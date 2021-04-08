"""
 Copyright (c) Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from collections import Counter
from functools import partial
from typing import Dict
from typing import List

import onnx
import pytest
import torch
import torch.nn
from nncf.common.graph.transformations.commands import TargetType
from nncf.graph.graph import InputAgnosticOperationExecutionContext
from nncf.graph.transformations.commands import PTTargetPoint
from nncf.quantization.layers import AsymmetricQuantizer
from nncf.quantization.quantizer_id import NonWeightQuantizerId
from nncf.quantization.quantizer_propagation import QuantizerPropagationSolver
from onnx import numpy_helper
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init
from tests.helpers import get_nodes_by_type

def make_ia_op_exec_context_for_coalescing_test(scope_str: str) -> InputAgnosticOperationExecutionContext:
    ia_op_exec_context = InputAgnosticOperationExecutionContext.from_str(scope_str)
    return ia_op_exec_context


def make_insertion_point_for_coalescing_test(scope_str: str,
                                             input_port_id: int = None)\
        -> PTTargetPoint:
    ia_op_exec_context = make_ia_op_exec_context_for_coalescing_test(scope_str)
    retval = PTTargetPoint(TargetType.OPERATOR_POST_HOOK,
                           ia_op_exec_context=ia_op_exec_context,
                           input_port_id=input_port_id)
    return retval


@pytest.mark.parametrize("input_insertion_points, linked_scopes_groups_list, ref_coalesced_ip_lists",
                         # ref_coalesced_ip_lists == None means that the coalescing should raise an exception
                         [
                             # 0 - Empty linked scopes list
                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0",
                                         input_port_id=1
                                     ),
                                 ],
                                 [],
                                 # Each coalesced list has one entry
                                 [
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ), ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0",
                                         input_port_id=1
                                     ), ]
                                 ],
                             ),
                             # 1 - Linked scope only affects 1 operation
                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         input_port_id=0
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0"]],
                                 # Each coalesced list has one entry
                                 [
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         input_port_id=0
                                     ), ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ), ]
                                 ]
                             ),
                             # 2 - Same as 1 but with multiple groups
                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         input_port_id=0
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0",
                                         input_port_id=1
                                     )
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0"], ["Foo/Xyz[leet]/__add___0"]],
                                 # Each coalesced list has one entry again
                                 [
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         input_port_id=0
                                     ), ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0",
                                         input_port_id=1
                                     ), ]
                                 ]
                             ),
                             # 3 - Single group affecting some of the scopes
                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0",
                                         input_port_id=0
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0",
                                         input_port_id=1
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0",
                                         input_port_id=1
                                     )
                                 ],
                                 [["Foo/Xyz[leet]/matmul_0", "Foo/Xyz[leet]/__add___0", "Foo/Baz[bar]/linear_0"]],
                                 [
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0",
                                         input_port_id=1),
                                         make_insertion_point_for_coalescing_test(
                                             "Foo/Baz[bar]/linear_0",
                                             input_port_id=0),
                                         make_insertion_point_for_coalescing_test(
                                             "Foo/Xyz[leet]/__add___0",
                                             input_port_id=1),
                                     ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ), ]
                                 ]
                             ),

                             # 4 - Multiple groups, each affecting one operation
                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         input_port_id=0
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0",
                                         input_port_id=0
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0",
                                         input_port_id=0
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/linear_0"], ["Foo/Asdf[jkl]/softmax_0"]],
                                 [
                                     # Each coalesced list has one entry again
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"), ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"), ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         input_port_id=0), ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0",
                                         input_port_id=0), ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0",
                                         input_port_id=0), ],
                                 ]
                             ),

                             # 5 - Multiple groups affecting multiple operations without overlapping
                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0",
                                         input_port_id=0
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0",
                                         input_port_id=1
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_1",
                                         input_port_id=0
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0",
                                   "Foo/Baz[bar]/linear_0"],
                                  ["Foo/Asdf[jkl]/softmax_1", "Foo/Xyz[leet]/__add___0"]],
                                 [
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"),
                                         make_insertion_point_for_coalescing_test(
                                             "Foo/Baz[bar]/linear_0",
                                             input_port_id=0),
                                     ],
                                     [make_insertion_point_for_coalescing_test(
                                                 "Foo/Asdf[jkl]/softmax_1",
                                                 input_port_id=0),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0",
                                         input_port_id=1), ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"), ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"), ]
                                 ]
                             ),

                             # 6 - A variation of 5
                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0",
                                         input_port_id=0,
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Baz[bar]/linear_0", "Foo/Xyz[leet]/matmul_0"],
                                  ["Foo/Asdf[jkl]/softmax_0", "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 [
                                     [
                                         make_insertion_point_for_coalescing_test(
                                             "Foo/Baz[bar]/conv2d_0"),
                                         make_insertion_point_for_coalescing_test(
                                             "Foo/Baz[bar]/linear_0"
                                         ),
                                         make_insertion_point_for_coalescing_test(
                                             "Foo/Xyz[leet]/matmul_0"
                                         )
                                     ],
                                     [
                                         make_insertion_point_for_coalescing_test(
                                             "Foo/Asdf[jkl]/softmax_0"),
                                         make_insertion_point_for_coalescing_test(
                                             "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0",
                                             input_port_id=0)
                                     ],
                                     [make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"),]
                                 ]
                             ),

                             # 7 - Overlapping groups
                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0",
                                         input_port_id=1,
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Baz[bar]/linear_0", "Foo/Xyz[leet]/matmul_0"],
                                  ["Foo/Xyz[leet]/matmul_0",
                                   "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 None
                             ),

                             # 8 - More than 1 match for the operation specified in the group

                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
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
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Xyz[leet]/matmul_0"],
                                  ["Foo/Xyz[leet]/matmul_0",
                                   "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 None
                             ),

                             # 9 - No match for an operation specified in the group
                             (
                                 [
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         input_port_id=0,
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0",
                                         input_port_id=1,
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_point_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Xyz[leet]/matmul_1"],
                                  ["Foo/Xyz[leet]/matmul_0",
                                   "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 None
                             ),
                         ])
def test_insertion_point_coalescing(input_insertion_points: List[PTTargetPoint],
                                    linked_scopes_groups_list: List[List[str]],
                                    ref_coalesced_ip_lists: List[List[PTTargetPoint]]):
    if ref_coalesced_ip_lists is None:
        with pytest.raises(RuntimeError):
            _ = QuantizerPropagationSolver.coalesce_insertion_points(input_insertion_points,
                                                                     linked_scopes_groups_list)
    else:
        test_coalesced_ip_lists = QuantizerPropagationSolver.coalesce_insertion_points(
            input_insertion_points,
            linked_scopes_groups_list)
        assert len(test_coalesced_ip_lists) == len(ref_coalesced_ip_lists)
        for idx, test_list in enumerate(test_coalesced_ip_lists):
            assert Counter(test_list) == Counter(ref_coalesced_ip_lists[idx])


class QuantizerLinkingTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_trainable_param = torch.nn.Parameter(torch.ones([1]))

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
        }
    ]
    nncf_config["compression"]["activations"] = {
        "unified_scale_ops": [
            [
                # Note: Assuming that quantizers are attached as a post-op to the specified operation
                "QuantizerLinkingTestModel/Path[path2]/__mul___0",
                "QuantizerLinkingTestModel/Path[path2]/__add___0",
            ]
        ],
        "ignored_scopes": [
            # Ignore path output averaging operations
            "QuantizerLinkingTestModel/__add___0",
            "QuantizerLinkingTestModel/__add___1",
            "QuantizerLinkingTestModel/__add___2",
        ]
    }

    compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(QuantizerLinkingTestModel(),
                                                                                   nncf_config)

    # 18 inputs to quantize (14 regular + 4 linked),
    # 8 quantization points left after propagation, out of these 3 are linked
    assert len(compression_ctrl.non_weight_quantizers) == 6

    shared_quantizer_id = NonWeightQuantizerId(
        InputAgnosticOperationExecutionContext.from_str("/nncf_model_input_0"))

    non_shared_spies = []
    for aq_id, aq_info in compression_ctrl.non_weight_quantizers.items():
        quantizer = aq_info.quantizer_module_ref
        spy = mocker.spy(quantizer, 'forward')
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


def test_unified_scales_for_vpu():
    nncf_config = get_quantization_config_without_range_init(model_size=1)
    nncf_config["input_info"] = [
        {
            "sample_size": [1, 1, 1, 1],
        },
        {
            "sample_size": [1, 1, 1, 1],
        }
    ]
    nncf_config["target_device"] = "VPU"

    _, compression_ctrl = create_compressed_model_and_algo_for_test(QuantizerLinkingTestModel(),
                                                                    nncf_config)

    assert len(compression_ctrl.non_weight_quantizers) == 2

    total_quantizations = sum(
        [len(info.affected_insertions) for info in compression_ctrl.non_weight_quantizers.values()])
    assert total_quantizations == 8


class SimplerModelForUnifiedScalesTesting(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2d_2 = torch.nn.Conv2d(1, 1, 1)
        self.conv2d_3 = torch.nn.Conv2d(1, 1, 1)
        self.conv2d_4 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        in_1, in_2 = x.chunk(dim=-1, chunks=2)
        in_1 = self.conv2d_1(in_1)
        in_2 = self.conv2d_2(in_2)
        x = in_1 + in_2
        x = torch.cat([x, x], dim=-1)
        in_1, in_2 = x.chunk(dim=-1, chunks=2)
        in_1 = self.conv2d_3(in_1)
        in_2 = self.conv2d_4(in_2)
        x = in_1 * in_2
        return x


class TwoEmbeddingAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding1 = torch.nn.Embedding(10, 10)
        self.embedding2 = torch.nn.Embedding(10, 10)

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
    def get_successor(node: onnx.NodeProto, graph: onnx.GraphProto) -> onnx.NodeProto:
        assert len(node.output) == 1  # Only single-output nodes are supported in this func
        for target_node in graph.node:
            if node.output[0] in target_node.input:
                return target_node
        return None

    @staticmethod
    def group_nodes_by_output_target(nodes: List[onnx.NodeProto], graph: onnx.GraphProto) -> List[
        List[onnx.NodeProto]]:
        output_nodes = {}  # type: Dict[str, List[onnx.NodeProto]]
        for node in nodes:
            target_node_name = TestsWithONNXInspection.get_successor(node, graph).name
            if target_node_name not in output_nodes:
                output_nodes[target_node_name] = []
            output_nodes[target_node_name].append(node)
        return list(output_nodes.values())

    @staticmethod
    def resolve_constant_node_inputs_to_values(node: onnx.NodeProto, graph: onnx.GraphProto) -> \
            Dict[str, onnx.AttributeProto]:
        retval = {}
        for input_ in node.input:
            constant_input_nodes = [x for x in graph.node if input_ in x.output and x.op_type == "Constant"]
            for constant_input_node in constant_input_nodes:
                assert len(constant_input_node.attribute) == 1
                val = constant_input_node.attribute[0]
                retval[input_] = numpy_helper.to_array(val.t)
        return retval

    def test_unified_scales_are_identical_in_onnx(self, tmp_path):
        # pylint:disable=no-member
        nncf_config = get_quantization_config_without_range_init(model_size=1)
        nncf_config["compression"]["quantize_outputs"] = True
        nncf_config["input_info"] = [
            {
                "sample_size": [1, 1, 1, 2],
            },
        ]
        nncf_config["target_device"] = "VPU"

        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
            SimplerModelForUnifiedScalesTesting(),
            nncf_config)

        with torch.no_grad():
            for quant_info in compression_ctrl.non_weight_quantizers.values():
                if isinstance(quant_info.quantizer_module_ref, AsymmetricQuantizer):
                    quant_info.quantizer_module_ref.input_range *= torch.abs(
                        torch.rand_like(quant_info.quantizer_module_ref.input_range))
                else:
                    quant_info.quantizer_module_ref.scale *= torch.abs(
                        torch.rand_like(quant_info.quantizer_module_ref.scale))

        test_input1 = torch.ones([1, 1, 1, 2])
        compressed_model.forward(test_input1)

        onnx_path = str(tmp_path / "model.onnx")
        compression_ctrl.export_model(onnx_path)

        onnx_model = onnx.load(onnx_path)

        fq_nodes = TestsWithONNXInspection.get_fq_nodes(onnx_model)
        eltwise_dominator_predicate = partial(TestsWithONNXInspection.immediately_dominates_add_or_mul,
                                    graph=onnx_model.graph)
        eltwise_fq_nodes = list(filter(eltwise_dominator_predicate, fq_nodes))
        fq_nodes_grouped_by_output = TestsWithONNXInspection.group_nodes_by_output_target(eltwise_fq_nodes,
                                                                                          onnx_model.graph)

        for unified_scale_group in fq_nodes_grouped_by_output:
            inputs = [TestsWithONNXInspection.resolve_constant_node_inputs_to_values(fq_node,
                                                                                     onnx_model.graph)
                      for fq_node in unified_scale_group]
            for inputs_dict in inputs[1:]:
                curr_values = list(inputs_dict.values())
                ref_values = list(inputs[0].values())
                assert curr_values == ref_values  # All inputs for unified scale quantizers must be equal

    def test_weight_and_act_quantizer_scale_unification(self, tmp_path):
        # pylint:disable=no-member
        nncf_config = get_quantization_config_without_range_init(model_size=1)
        nncf_config["input_info"] = [
            {
                "sample_size": [1, 5],
                "type": "long",
                "filler": "zeros"
            },
        ]
        nncf_config["target_device"] = "VPU"

        compressed_model, compression_ctrl = create_compressed_model_and_algo_for_test(
            TwoEmbeddingAddModel(),
            nncf_config)

        with torch.no_grad():
            for quant_module in compression_ctrl.all_quantizations.values():
                if isinstance(quant_module, AsymmetricQuantizer):
                    quant_module.input_range *= torch.abs(
                        torch.rand_like(quant_module.input_range))
                else:
                    quant_module.scale *= torch.abs(
                        torch.rand_like(quant_module.scale))

        test_input1 = torch.ones([1, 5], dtype=torch.long)
        compressed_model.forward(test_input1)

        onnx_path = str(tmp_path / "model.onnx")
        compression_ctrl.export_model(onnx_path)

        onnx_model = onnx.load(onnx_path)

        fq_nodes = TestsWithONNXInspection.get_fq_nodes(onnx_model)
        eltwise_dominator_predicate = partial(TestsWithONNXInspection.immediately_dominates_add_or_mul,
                                              graph=onnx_model.graph)
        embedding_dominator_predicate = partial(TestsWithONNXInspection.immediately_dominates_embedding,
                                                graph=onnx_model.graph)
        eltwise_fq_nodes = list(filter(eltwise_dominator_predicate, fq_nodes))
        embedding_weight_fq_nodes = list(filter(embedding_dominator_predicate, fq_nodes))

        fq_nodes_with_expected_unified_scales = embedding_weight_fq_nodes + eltwise_fq_nodes

        unified_fq_node_inputs = [TestsWithONNXInspection.resolve_constant_node_inputs_to_values(fq_node,
                                                                                                 onnx_model.graph)
                                  for fq_node in fq_nodes_with_expected_unified_scales]
        for inputs_dict in unified_fq_node_inputs[1:]:
            curr_values = list(inputs_dict.values())
            ref_values = list(unified_fq_node_inputs[0].values())
            assert curr_values == ref_values  # All inputs for unified scale quantizers must be equal
