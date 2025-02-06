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

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import pytest
import torch
import torch.ao.quantization
import torch.fx
from torch.ao.quantization.fx.utils import create_getattr_from_value
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.quantization.fake_quantize import FakeQuantize

import nncf
import nncf.common
import nncf.common.factory
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.experimental.torch.fx.constant_folding import constant_fold
from nncf.experimental.torch.fx.model_transformer import FXModelTransformer
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.experimental.torch.fx.transformations import _get_node_by_input_port_id
from nncf.experimental.torch.fx.transformations import _set_new_node_meta
from nncf.experimental.torch.fx.transformations import compress_post_quantize_transformation
from nncf.experimental.torch.fx.transformations import constant_update_transformation_builder
from nncf.experimental.torch.fx.transformations import fold_constant_except_qdq
from nncf.experimental.torch.fx.transformations import leaf_module_insertion_transformation_builder
from nncf.experimental.torch.fx.transformations import module_insertion_transformation_builder
from nncf.experimental.torch.fx.transformations import node_removal_transformation_builder
from nncf.experimental.torch.fx.transformations import output_insertion_transformation_builder
from nncf.experimental.torch.fx.transformations import qdq_insertion_transformation_builder
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from tests.torch.fx.helpers import get_torch_fx_model
from tests.torch.test_compressed_graph import check_graph
from tests.torch.test_models.synthetic import ConstantFoldingTestModel
from tests.torch.test_models.synthetic import ConvolutionWithAllConstantInputsModel
from tests.torch.test_models.synthetic import ConvolutionWithNotTensorBiasModel
from tests.torch.test_models.synthetic import ConvolutionWithSeveralOutputs
from tests.torch.test_models.synthetic import MultiBranchesConnectedModel
from tests.torch.test_models.synthetic import MultiBranchesConnectedModelWithConcat


@dataclass
class ModelExtractionTestCase:
    model: torch.nn.Module
    input_shape: Tuple[int, ...]
    command: PTModelExtractionCommand


EXTRACTED_GRAPHS_DIR_NAME = str(Path("fx") / "extracted")
TRANSFORMED_GRAPH_DIR_NAME = str(Path("fx") / "transformed")

MODEL_EXTRACTION_CASES = (
    ModelExtractionTestCase(
        ConvolutionWithNotTensorBiasModel, (1, 1, 3, 3), PTModelExtractionCommand(["conv2d"], ["conv2d"])
    ),
    ModelExtractionTestCase(
        ConvolutionWithAllConstantInputsModel, (1, 1, 3, 3), PTModelExtractionCommand(["conv2d"], ["conv2d"])
    ),
    ModelExtractionTestCase(
        MultiBranchesConnectedModel, (1, 3, 3, 3), PTModelExtractionCommand(["conv2d_1"], ["add__1"])
    ),
    ModelExtractionTestCase(
        MultiBranchesConnectedModel, (1, 3, 3, 3), PTModelExtractionCommand(["conv2d"], ["add_", "add"])
    ),
    ModelExtractionTestCase(
        MultiBranchesConnectedModel, (1, 3, 3, 3), PTModelExtractionCommand(["conv2d", "conv2d_1"], ["add_", "add__1"])
    ),
    ModelExtractionTestCase(
        MultiBranchesConnectedModel, (1, 3, 3, 3), PTModelExtractionCommand(["conv2d"], ["conv2d_2"])
    ),
    ModelExtractionTestCase(
        MultiBranchesConnectedModel, (1, 3, 3, 3), PTModelExtractionCommand(["conv2d"], ["add__1"])
    ),
)


def get_test_id(test_case: ModelExtractionTestCase):
    return test_case.model.__name__ + "_".join(test_case.command.input_node_names + test_case.command.output_node_names)


def idfn(value: Any):
    if isinstance(value, ModelExtractionTestCase):
        return get_test_id(value)
    return None


def _target_point_to_str(target_point: PTTargetPoint) -> str:
    return "_".join(
        map(str, (target_point.target_node_name, target_point.target_type.value, target_point.input_port_id))
    )


@pytest.mark.parametrize("test_case", MODEL_EXTRACTION_CASES, ids=idfn)
def test_model_extraction(test_case: ModelExtractionTestCase):
    captured_model = get_torch_fx_model(test_case.model(), torch.ones(test_case.input_shape))
    _, nncf_graph = _extract_model(test_case, captured_model)
    check_graph(nncf_graph, f"{get_test_id(test_case)}.dot", EXTRACTED_GRAPHS_DIR_NAME, extended=True)


@pytest.mark.parametrize(
    "test_case,tuple_output,ref_output",
    (
        (
            ModelExtractionTestCase(
                ConvolutionWithNotTensorBiasModel, (1, 1, 3, 3), PTModelExtractionCommand(["conv2d"], ["output_1"])
            ),
            False,
            "(conv2d,)",
        ),
        (
            ModelExtractionTestCase(
                ConvolutionWithNotTensorBiasModel,
                (1, 1, 3, 3),
                PTModelExtractionCommand(["conv2d"], ["conv2d", "output_1", "conv2d"]),
            ),
            False,
            "(conv2d, conv2d, conv2d)",
        ),
        (
            ModelExtractionTestCase(
                ConvolutionWithSeveralOutputs, (1, 1, 3, 3), PTModelExtractionCommand(["conv2d"], ["output_1"])
            ),
            False,
            "([conv2d, add],)",
        ),
        (
            ModelExtractionTestCase(
                ConvolutionWithSeveralOutputs,
                (1, 1, 3, 3),
                PTModelExtractionCommand(["conv2d"], ["conv2d", "output_1", "conv2d"]),
            ),
            False,
            "(conv2d, [conv2d, add], conv2d)",
        ),
        (
            ModelExtractionTestCase(
                ConvolutionWithNotTensorBiasModel, (1, 1, 3, 3), PTModelExtractionCommand(["conv2d"], ["output_1"])
            ),
            True,
            "(conv2d,)",
        ),
        (
            ModelExtractionTestCase(
                ConvolutionWithNotTensorBiasModel,
                (1, 1, 3, 3),
                PTModelExtractionCommand(["conv2d"], ["conv2d", "output_1", "conv2d"]),
            ),
            True,
            "(conv2d, conv2d, conv2d)",
        ),
    ),
    ids=idfn,
)
def test_model_extraction_with_original_output(test_case: ModelExtractionTestCase, tuple_output: bool, ref_output: str):
    captured_model = get_torch_fx_model(test_case.model(), torch.ones(test_case.input_shape))
    if tuple_output:
        output_node = [node for node in captured_model.graph.nodes if node.op == "output"][0]
        output_node.args = (output_node.args[0][0],)
        captured_model.recompile()
    extracted_model, nncf_graph = _extract_model(test_case, captured_model)
    check_graph(nncf_graph, f"{get_test_id(test_case)}.dot", EXTRACTED_GRAPHS_DIR_NAME, extended=True)

    output_node = [node for node in extracted_model.graph.nodes if node.op == "output"][0]
    assert str(output_node.args[0]) == ref_output


def _extract_model(test_case: ModelExtractionTestCase, captured_model: torch.fx.GraphModule):
    layout = TransformationLayout()
    layout.register(test_case.command)
    extracted_model = FXModelTransformer(captured_model).transform(layout)
    return extracted_model, GraphConverter.create_nncf_graph(extracted_model)


MultiBranchesConnectedModelWithConcat_TARGET_POINTS = (
    PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, "conv2d", input_port_id=0),
    PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, "conv2d", input_port_id=1),
    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, "conv2d_1", input_port_id=1),
    PTTargetPoint(TargetType.OPERATOR_POST_HOOK, "conv2d"),
    PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, "cat", input_port_id=1),
    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, "cat", input_port_id=2),
)


@pytest.mark.parametrize("leaf", [False, True], ids=["no_leaf", "leaf"])
def test_model_insertion_transformation(leaf: bool):
    class TestInsertModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    target_points = list(MultiBranchesConnectedModelWithConcat_TARGET_POINTS)
    target_node_name = "TEST_MODULE"
    test_module_instance = TestInsertModule()
    builder = leaf_module_insertion_transformation_builder if leaf else module_insertion_transformation_builder
    transformation = builder(test_module_instance, target_points, target_node_name)

    model = MultiBranchesConnectedModelWithConcat()
    captured_model = get_torch_fx_model(model, torch.ones(MultiBranchesConnectedModelWithConcat.INPUT_SIZE))
    transformation(captured_model)

    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    assert getattr(captured_model, target_node_name) is test_module_instance
    check_graph(nncf_graph, f"model_insertion{'_leaf' if leaf else ''}.dot", TRANSFORMED_GRAPH_DIR_NAME, extended=True)


@pytest.mark.parametrize("concat", [False, True])
def test_constant_update_transformation(concat: bool):
    model = MultiBranchesConnectedModelWithConcat()
    captured_model = get_torch_fx_model(model, torch.ones(MultiBranchesConnectedModelWithConcat.INPUT_SIZE))
    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    target_node_name = "cat" if concat else "add_"
    target_node = nncf_graph.get_node_by_name(target_node_name)
    input_port_id = 2 if concat else 1

    builder = constant_update_transformation_builder
    new_value = torch.tensor((42.0,))
    transformation = builder(target_node, value=new_value, input_port_id=input_port_id)
    transformation(captured_model)

    target_graph_node = get_graph_node_by_name(captured_model.graph, target_node_name)
    new_const_node = _get_node_by_input_port_id(target_graph_node, input_port_id)
    assert get_tensor_constant_from_node(new_const_node, captured_model) == new_value

    transformed_nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    check_graph(
        transformed_nncf_graph,
        f"{'cat_' if concat else ''}constant_update.dot",
        TRANSFORMED_GRAPH_DIR_NAME,
        extended=True,
    )


def test_constant_update_transformation_no_constant():
    model = MultiBranchesConnectedModel()
    captured_model = get_torch_fx_model(model, torch.ones((1, 3, 3, 3)))
    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    target_node = nncf_graph.get_node_by_name("add")

    new_value = torch.tensor((42.0,))
    transformation = constant_update_transformation_builder(target_node, value=new_value, input_port_id=1)
    with pytest.raises(nncf.InternalError):
        transformation(captured_model)


@pytest.mark.parametrize(
    "q_min,q_max,dtype",
    [(-128, 127, torch.qint8), (0, 255, torch.quint8)],
    ids=["int8", "uint8"],
)
class TestQDQInsertion:
    REF_SCALE = torch.tensor([1.0])
    REF_ZERO_POINT = torch.tensor([0.0])

    def _get_quantizer(
        self, per_channel: bool, symmetric: bool, q_min: torch.Tensor, q_max: torch.Tensor, dtype: torch.dtype
    ) -> FakeQuantize:
        if symmetric:
            qscheme = torch.per_channel_symmetric if per_channel else torch.per_tensor_symmetric
        else:
            qscheme = torch.per_channel_affine if per_channel else torch.per_tensor_affine
        observer = PerChannelMinMaxObserver if per_channel else MinMaxObserver

        quantizer = FakeQuantize(
            observer=observer,
            quant_min=q_min,
            quant_max=q_max,
            dtype=dtype,
            qscheme=qscheme,
            eps=1e-5,
        )
        quantizer.scale = self.REF_SCALE
        quantizer.zero_point = self.REF_ZERO_POINT

        return quantizer

    def _check_qdq_params(
        self, captured_model: torch.fx.GraphModule, target_point: PTTargetPoint, dtype: torch.dtype, per_channel: bool
    ):
        target_node = get_graph_node_by_name(captured_model.graph, target_point.target_node_name)
        if target_point.target_type in [TargetType.OPERATION_WITH_WEIGHTS, TargetType.OPERATOR_PRE_HOOK]:
            dq_node = _get_node_by_input_port_id(target_node, target_point.input_port_id)
            q_node = dq_node.args[0]
        else:
            q_node = list(target_node.users)[0]

        ref_dtype = torch.int8 if dtype == torch.qint8 else torch.uint8
        if per_channel:

            def get_value(node: torch.fx.Node):
                return get_tensor_constant_from_node(node, captured_model)

        else:

            def get_value(node: torch.fx.Node):
                return node

        assert q_node.args[-1] == ref_dtype
        assert get_value(q_node.args[1]) == self.REF_SCALE
        assert get_value(q_node.args[2]) == self.REF_ZERO_POINT
        for dq_node in q_node.users:
            assert get_value(dq_node.args[1]) == self.REF_SCALE
            assert get_value(dq_node.args[2]) == self.REF_ZERO_POINT
            assert dq_node.args[-1] == ref_dtype

    @pytest.mark.parametrize("target_point", MultiBranchesConnectedModelWithConcat_TARGET_POINTS)
    def test_one_target_point(
        self,
        is_per_channel: bool,
        quantization_mode: QuantizationMode,
        q_min: int,
        q_max: int,
        dtype: torch.dtype,
        target_point: PTTargetPoint,
    ):
        symmetric = quantization_mode == QuantizationMode.SYMMETRIC
        quantizer = self._get_quantizer(is_per_channel, symmetric, q_min, q_max, dtype)
        transformation = qdq_insertion_transformation_builder(quantizer, [target_point])

        model = MultiBranchesConnectedModelWithConcat()
        captured_model = get_torch_fx_model(model, torch.ones(MultiBranchesConnectedModelWithConcat.INPUT_SIZE))
        transformation(captured_model)

        self._check_qdq_params(captured_model, target_point, dtype, is_per_channel)

        nncf_graph = GraphConverter.create_nncf_graph(captured_model)
        ref_name = (
            f"qdq_insert_{_target_point_to_str(target_point)}"
            f"_{'per_channel' if is_per_channel else 'per_tensor'}.dot"
        )
        check_graph(
            nncf_graph,
            ref_name,
            TRANSFORMED_GRAPH_DIR_NAME,
            extended=True,
        )

    @pytest.mark.parametrize(
        "target_points,weights",
        [
            (
                [
                    PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, "conv2d", input_port_id=0),
                    PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, "conv2d", input_port_id=1),
                    PTTargetPoint(TargetType.OPERATOR_POST_HOOK, "conv2d"),
                ],
                False,
            ),
            (
                [
                    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, "conv2d", input_port_id=1),
                    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, "conv2d_1", input_port_id=1),
                ],
                True,
            ),
        ],
    )
    def test_shared_target_point(
        self,
        is_per_channel: bool,
        quantization_mode: QuantizationMode,
        q_min: int,
        q_max: int,
        dtype: torch.dtype,
        target_points: PTTargetPoint,
        weights: bool,
    ):
        symmetric = quantization_mode == QuantizationMode.SYMMETRIC
        quantizer = self._get_quantizer(is_per_channel, symmetric, q_min, q_max, dtype)
        transformation = qdq_insertion_transformation_builder(quantizer, target_points)

        model = MultiBranchesConnectedModelWithConcat()
        captured_model = get_torch_fx_model(model, torch.ones(MultiBranchesConnectedModelWithConcat.INPUT_SIZE))
        if not weights:
            with pytest.raises(nncf.InternalError):
                transformation(captured_model)
            return

        transformation(captured_model)

        for target_point in target_points:
            self._check_qdq_params(captured_model, target_point, dtype, is_per_channel)

        nncf_graph = GraphConverter.create_nncf_graph(captured_model)
        ref_name = (
            f"qdq_shared_insert_{'weights' if weights else 'activations'}"
            f"_{'per_channel' if is_per_channel else 'per_tensor'}.dot"
        )
        check_graph(
            nncf_graph,
            ref_name,
            TRANSFORMED_GRAPH_DIR_NAME,
            extended=True,
        )


def test_node_removal_transformation():
    model = MultiBranchesConnectedModel()
    captured_model = get_torch_fx_model(model, torch.ones((1, 3, 3, 3)))
    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    node = nncf_graph.get_node_by_name("conv2d")
    transformation = node_removal_transformation_builder(node, input_port_id=0)
    transformation(captured_model)
    transformed_nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    check_graph(transformed_nncf_graph, "node_removal_ref.dot", TRANSFORMED_GRAPH_DIR_NAME, extended=True)


@pytest.mark.parametrize("tuple_output", [False, True], ids=["node_out", "tuple_out"])
@pytest.mark.parametrize("target_point", MultiBranchesConnectedModelWithConcat_TARGET_POINTS)
def test_output_insertion_transformation(tuple_output: bool, target_point: PTTargetPoint):
    model = MultiBranchesConnectedModelWithConcat()
    captured_model = get_torch_fx_model(model, torch.ones(MultiBranchesConnectedModelWithConcat.INPUT_SIZE))

    if not tuple_output:
        output_node = [node for node in captured_model.graph.nodes if node.op == "output"][0]
        output_node.args = (output_node.args[0][0],)
        captured_model.recompile()

    transformation = output_insertion_transformation_builder(target_point)
    transformation(captured_model)

    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    ref_name = f"output_insertion_{_target_point_to_str(target_point)}.dot"
    check_graph(
        nncf_graph,
        ref_name,
        TRANSFORMED_GRAPH_DIR_NAME,
        extended=True,
    )


def count_constants(model: torch.fx.GraphModule) -> int:
    num_constant_nodes = 0
    for node in model.graph.nodes:
        if node.op == "get_attr":
            num_constant_nodes += 1
    return num_constant_nodes


def insert_qdq_nodes(
    model: torch.fx.GraphModule,
    correct_pattern: bool,
    per_channel: bool,
    node_name: str = "conv2d",
    w_const_node_name: str = "conv_a_weight",
):
    const_node = get_graph_node_by_name(model.graph, w_const_node_name)
    if per_channel:
        quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
        dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default
    else:
        quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
        dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default

    conv_node = get_graph_node_by_name(model.graph, node_name)
    if per_channel:
        with model.graph.inserting_before(conv_node):
            scale_node = create_getattr_from_value(
                model,
                model.graph,
                "scale_node",
                torch.ones([3]),
            )
            zp_node = create_getattr_from_value(
                model,
                model.graph,
                "weight_node",
                torch.ones([3]),
            )
        qdq_args = (scale_node, zp_node, 0, -128, 127, torch.int8)
    else:
        qdq_args = (1.0, 1, -128, 127, torch.int8)
    with model.graph.inserting_before(conv_node):
        q_node = model.graph.create_node("call_function", quantize_op, (const_node,) + qdq_args, {})
        if not correct_pattern:
            add_op = torch.ops.aten.add.Tensor
            add_node = model.graph.create_node("call_function", add_op, (q_node, 0), {})
            dq_node = model.graph.create_node("call_function", dequantize_op, (add_node,) + qdq_args, {})
            _set_new_node_meta(q_node, (const_node,) + qdq_args, quantize_op, model)
            _set_new_node_meta(add_node, (q_node, 0), add_op, model)
            _set_new_node_meta(dq_node, (add_node,) + qdq_args, dequantize_op, model)
        else:
            dq_node = model.graph.create_node("call_function", dequantize_op, (q_node,) + qdq_args, {})
            _set_new_node_meta(q_node, (const_node,) + qdq_args, quantize_op, model)
            _set_new_node_meta(dq_node, (q_node,) + qdq_args, dequantize_op, model)
    conv_node.replace_input_with(const_node, dq_node)
    model.graph.eliminate_dead_code()
    model.recompile()


def test_compress_post_quantize_transformation(is_per_channel: bool):
    model = MultiBranchesConnectedModel()
    ex_input = torch.ones(1, 3, 224, 224)

    model_with_correct_pattern = get_torch_fx_model(model, ex_input)
    insert_qdq_nodes(model_with_correct_pattern, correct_pattern=True, per_channel=is_per_channel)
    compress_post_quantize_transformation(model_with_correct_pattern)
    graph_name = f"compress_post_quantize_{'per_channel' if is_per_channel else 'per_tensor'}_valid.dot"
    check_graph(
        NNCFGraphFactory.create(model_with_correct_pattern),
        graph_name,
        TRANSFORMED_GRAPH_DIR_NAME,
        extended=True,
    )

    model_with_incorrect_pattern = get_torch_fx_model(model, ex_input)
    insert_qdq_nodes(model_with_incorrect_pattern, correct_pattern=False, per_channel=is_per_channel)
    compress_post_quantize_transformation(model_with_incorrect_pattern)
    graph_name = f"compress_post_quantize_{'per_channel' if is_per_channel else 'per_tensor'}_invalid.dot"
    check_graph(
        NNCFGraphFactory.create(model_with_incorrect_pattern),
        graph_name,
        TRANSFORMED_GRAPH_DIR_NAME,
        extended=True,
    )


def test_constant_folding():
    model = ConstantFoldingTestModel()
    ex_inputs = (torch.ones(model.INPUT_SIZE), torch.ones((1,)))
    captured_model = get_torch_fx_model(model, ex_inputs)
    folded_model = deepcopy(captured_model)
    constant_fold(folded_model)

    # Check the folded const does not require gradient
    assert not folded_model._frozen_param0.requires_grad

    assert torch.allclose(captured_model(*ex_inputs), folded_model(*ex_inputs))

    nncf_graph = GraphConverter.create_nncf_graph(folded_model)
    check_graph(nncf_graph, "folded_model.dot", TRANSFORMED_GRAPH_DIR_NAME, extended=True)


def test_constant_folding_with_constraints(is_per_channel):
    model = ConstantFoldingTestModel()
    ex_inputs = (torch.ones(model.INPUT_SIZE), torch.ones((1,)))
    model_with_correct_pattern = get_torch_fx_model(model, ex_inputs)

    insert_qdq_nodes(
        model_with_correct_pattern,
        correct_pattern=True,
        per_channel=is_per_channel,
        node_name="linear_1",
        w_const_node_name="linear_act_weight",
    )
    fold_constant_except_qdq(model_with_correct_pattern)

    nncf_graph = GraphConverter.create_nncf_graph(model_with_correct_pattern)
    dot_file_name = f"folded_model_with_constraints_{'per_channel' if is_per_channel else 'per_tensor'}.dot"
    check_graph(nncf_graph, dot_file_name, TRANSFORMED_GRAPH_DIR_NAME, extended=True)
