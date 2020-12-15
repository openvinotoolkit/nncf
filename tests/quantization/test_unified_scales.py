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
from typing import List, Dict

import onnx
import pytest
import torch
import torch.nn
from onnx import numpy_helper

from nncf.dynamic_graph.graph import OperationExecutionContext, InputAgnosticOperationExecutionContext
from nncf.dynamic_graph.trace_tensor import TensorMeta
from nncf.nncf_network import InsertionInfo
from nncf.quantization.algo import QuantizationBuilder
from nncf.quantization.layers import AsymmetricQuantizer
from nncf.quantization.quantizer_id import NonWeightQuantizerId
from tests.helpers import create_compressed_model_and_algo_for_test
from tests.quantization.test_quantization_helpers import get_quantization_config_without_range_init


def make_op_exec_context_for_coalescing_test(scope_str: str) -> OperationExecutionContext:
    ia_op_exec_context = InputAgnosticOperationExecutionContext.from_str(scope_str)
    op_exec_context = OperationExecutionContext(ia_op_exec_context.operator_name,
                                                ia_op_exec_context.scope_in_model,
                                                ia_op_exec_context.call_order,
                                                [TensorMeta(0, 0, [1])])
    return op_exec_context


def make_insertion_info_for_coalescing_test(scope_str: str,
                                            linked_op_exec_contexts: List[OperationExecutionContext] = None):
    op_exec_context = make_op_exec_context_for_coalescing_test(scope_str)
    retval = InsertionInfo(op_exec_context)
    if linked_op_exec_contexts is not None:
        retval.linked_op_exec_contexts = linked_op_exec_contexts
    return retval


@pytest.mark.parametrize("input_insertion_infos, linked_scopes_groups_list, ref_coalesced_insertion_infos",
                         # ref_coalesced_insertion_infos == None means that the coalescing should raise an exception
                         [
                             # 0 - Empty linked scopes list
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ],
                                 [],
                                 # Same as input
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ],
                             ),
                             # 1 - Linked scope only affects 1 operation
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0"]],
                                 # Same as input
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ]
                             ),
                             # 2 - Same as 1 but with multiple groups
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0"], ["Foo/Xyz[leet]/__add___0"]],
                                 # Same as input again
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     )
                                 ]
                             ),
                             # 3 - Single group affecting some of the scopes
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     )
                                 ],
                                 [["Foo/Xyz[leet]/matmul_0", "Foo/Xyz[leet]/__add___0", "Foo/Baz[bar]/linear_0"]],
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Baz[bar]/linear_0"
                                             ),
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Xyz[leet]/__add___0"
                                             ),
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     )
                                 ]
                             ),

                             # 4 - Multiple groups, each affecting one operation
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/linear_0"], ["Foo/Asdf[jkl]/softmax_0"]],
                                 [
                                     # Same as input
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                 ]
                             ),

                             # 5 - Multiple groups affecting multiple operations without overlapping
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_1"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0",
                                   "Foo/Baz[bar]/linear_0"],
                                  ["Foo/Asdf[jkl]/softmax_1", "Foo/Xyz[leet]/__add___0"]],
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Baz[bar]/linear_0"
                                             ),
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_1",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Xyz[leet]/__add___0"
                                             ),
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                 ]
                             ),

                             # 6 - A variation of 5
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Baz[bar]/linear_0", "Foo/Xyz[leet]/matmul_0"],
                                  ["Foo/Asdf[jkl]/softmax_0", "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Baz[bar]/linear_0"
                                             ),
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Xyz[leet]/matmul_0"
                                             )
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0",
                                         linked_op_exec_contexts=[
                                             make_op_exec_context_for_coalescing_test(
                                                 "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                             ),
                                         ]
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                 ]
                             ),

                             # 7 - Overlapping groups
                             (
                                 [
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
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
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
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
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/conv2d_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Baz[bar]/linear_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/__add___0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Xyz[leet]/matmul_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/softmax_0"
                                     ),
                                     make_insertion_info_for_coalescing_test(
                                         "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"
                                     ),
                                 ],
                                 [["Foo/Baz[bar]/conv2d_0", "Foo/Xyz[leet]/matmul_1"],
                                  ["Foo/Xyz[leet]/matmul_0",
                                   "Foo/Asdf[jkl]/Qwer[tyu]/conv2d_0"]],
                                 None
                             ),
                         ])
def test_insertion_info_coalescing(input_insertion_infos: List[InsertionInfo],
                                   linked_scopes_groups_list: List[List[str]],
                                   ref_coalesced_insertion_infos: List[InsertionInfo]):
    if ref_coalesced_insertion_infos is None:
        with pytest.raises(RuntimeError):
            _ = QuantizationBuilder.coalesce_insertion_infos(input_insertion_infos,
                                                             linked_scopes_groups_list)
    else:
        test_coalesced_insertion_infos = QuantizationBuilder.coalesce_insertion_infos(input_insertion_infos,
                                                                                      linked_scopes_groups_list)
        assert Counter(test_coalesced_insertion_infos) == Counter(ref_coalesced_insertion_infos)


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


def test_quantizer_scale_linking():
    nncf_config = get_quantization_config_without_range_init(model_size=1)
    nncf_config['quantizer_setup_type'] = 'pattern_based'
    nncf_config["compression"]["quantize_outputs"] = True
    nncf_config["input_info"] = [
        {
            "sample_size": [1, 1, 1, 1],
        },
        {
            "sample_size": [1, 1, 1, 1],
        }
    ]
    nncf_config["compression"]["activations"] = {
        "linked_quantizer_scopes": [
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

    # 2 paths x 3 quantizers - 1 because two are shared in one path
    assert len(compression_ctrl.non_weight_quantizers) == 5

    test_input1 = torch.ones([1, 1, 1, 1])
    test_input2 = 2 * test_input1

    non_shared_mul_quantizer_id = NonWeightQuantizerId(
        InputAgnosticOperationExecutionContext.from_str("QuantizerLinkingTestModel/Path[path1]/__mul___0"))

    non_shared_add_quantizer_id = NonWeightQuantizerId(
        InputAgnosticOperationExecutionContext.from_str("QuantizerLinkingTestModel/Path[path1]/__add___0"))

    shared_quantizer_id = NonWeightQuantizerId(
        InputAgnosticOperationExecutionContext.from_str("QuantizerLinkingTestModel/Path[path2]/__mul___0"))

    non_shared_mul_quantizer = compression_ctrl.non_weight_quantizers[non_shared_mul_quantizer_id].quantizer_module_ref
    non_shared_add_quantizer = compression_ctrl.non_weight_quantizers[non_shared_add_quantizer_id].quantizer_module_ref
    shared_quantizer = compression_ctrl.non_weight_quantizers[shared_quantizer_id].quantizer_module_ref

    old_scale = 765.0  # so that the quantum is equal to 3
    with torch.no_grad():
        for quantizer in compression_ctrl.all_quantizations.values():
            quantizer.scale.fill_(old_scale)

    # Expected outputs without compression - 6, 12, 8. Scale deliberately set to preserve the values
    uncompressed_expected_outputs = (6.0 * torch.ones([1]), 12.0 * torch.ones([1]), 18.0 * torch.ones([1]))
    outputs_with_shared_scale_1 = compressed_model(test_input1, test_input2)

    for uncomp_out, comp_out_1 in zip(uncompressed_expected_outputs, outputs_with_shared_scale_1):
        assert torch.allclose(uncomp_out, comp_out_1)

    # Specifically clip the shared quantizer's outputs by setting scale to 1.0
    new_shared_scale = 1.0
    with torch.no_grad():
        shared_quantizer.scale.fill_(new_shared_scale)
    outputs_with_shared_scale_2 = compressed_model(test_input1, test_input2)

    # __add___0 outputs
    assert torch.allclose(outputs_with_shared_scale_2[0], 4.0 * torch.ones([1]))
    # __mul___0 outputs
    assert torch.allclose(outputs_with_shared_scale_2[1], 7.0 * torch.ones([1]))
    # __add___1 outputs
    assert torch.allclose(outputs_with_shared_scale_2[2], 12.0 * torch.ones([1]))

    # Clipping the non-shared quantizers at the same position in the path as the two shared ones
    # in the same manner is required to simulate the same grad input for both the shared quantizers
    # and the unshared ones
    with torch.no_grad():
        non_shared_mul_quantizer.scale.fill_(new_shared_scale)
        non_shared_add_quantizer.scale.fill_(new_shared_scale)
    final_output = compressed_model(test_input1, test_input2)[2]
    final_output.backward()

    assert torch.allclose(shared_quantizer.scale.grad,
                          non_shared_mul_quantizer.scale.grad + non_shared_add_quantizer.scale.grad)


def test_unified_scales_for_vpu():
    nncf_config = get_quantization_config_without_range_init(model_size=1)
    nncf_config["compression"]["quantize_outputs"] = True
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
        [len(info.affected_ia_op_exec_contexts) for info in compression_ctrl.non_weight_quantizers.values()])
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


def test_unified_scales_are_identical_in_onnx(tmp_path):
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

    onnx_path = tmp_path / "model.onnx"
    compression_ctrl.export_model(onnx_path)

    onnx_model = onnx.load(onnx_path)

    def get_fq_nodes(onnx_model: onnx.ModelProto) -> List[onnx.NodeProto]:
        retval = []
        for node in onnx_model.graph.node:
            if str(node.op_type) == "FakeQuantize":
                retval.append(node)
        return retval

    def immediately_dominates_add_or_mul(node: onnx.NodeProto, graph: onnx.GraphProto) -> bool:
        if len(node.output) != 1:
            return False
        output_tensor_id = node.output[0]
        matches = [x for x in graph.node if output_tensor_id in x.input]
        for match in matches:
            if match.op_type in ["Add", "Mul"]:
                return True
        return False

    def get_successor(node: onnx.NodeProto, graph: onnx.GraphProto) -> onnx.NodeProto:
        assert len(node.output) == 1  # Only single-output nodes are supported in this func
        for target_node in graph.node:
            if node.output[0] in target_node.input:
                return target_node
        return None

    def group_nodes_by_output_target(nodes: List[onnx.NodeProto], graph: onnx.GraphProto) -> List[List[onnx.NodeProto]]:
        output_nodes = {}  # type: Dict[str, List[onnx.NodeProto]]
        for node in nodes:
            target_node_name = get_successor(node, graph).name
            if target_node_name not in output_nodes:
                output_nodes[target_node_name] = []
            output_nodes[target_node_name].append(node)
        return list(output_nodes.values())

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

    fq_nodes = get_fq_nodes(onnx_model)
    eltwise_predicate = partial(immediately_dominates_add_or_mul, graph=onnx_model.graph)
    eltwise_fq_nodes = list(filter(eltwise_predicate, fq_nodes))
    fq_nodes_grouped_by_output = group_nodes_by_output_target(eltwise_fq_nodes, onnx_model.graph)

    for unified_scale_group in fq_nodes_grouped_by_output:
        inputs = [resolve_constant_node_inputs_to_values(fq_node, onnx_model.graph) for fq_node in unified_scale_group]
        for inputs_dict in inputs[1:]:
            curr_values = list(inputs_dict.values())
            ref_values = list(inputs[0].values())
            assert curr_values == ref_values  # All inputs for unified scale quantizers must be equal
