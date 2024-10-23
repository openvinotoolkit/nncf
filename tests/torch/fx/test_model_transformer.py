# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import pytest
import torch
import torch.ao.quantization
import torch.fx
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.fx.utils import create_getattr_from_value
from torch.ao.quantization.observer import MinMaxObserver
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.quantization.fake_quantize import FakeQuantize

import nncf
from nncf.common.factory import NNCFGraph
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.experimental.torch.fx.model_transformer import FXModelTransformer
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.experimental.torch.fx.transformations import _set_new_node_meta
from nncf.experimental.torch.fx.transformations import bias_update_transformation_builder
from nncf.experimental.torch.fx.transformations import constant_update_transformation_builder
from nncf.experimental.torch.fx.transformations import leaf_module_insertion_transformation_builder
from nncf.experimental.torch.fx.transformations import module_insertion_transformation_builder
from nncf.experimental.torch.fx.transformations import node_removal_transformation_builder
from nncf.experimental.torch.fx.transformations import output_insertion_transformation_builder
from nncf.experimental.torch.fx.transformations import qdq_insertion_transformation_builder
from nncf.experimental.torch.fx.transformations import shared_constants_unification_transformation
from nncf.torch import disable_patching
from nncf.torch.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from tests.torch.fx.test_sanity import count_q_dq
from tests.torch.test_compressed_graph import check_graph
from tests.torch.test_models.synthetic import ConvolutionWithAllConstantInputsModel
from tests.torch.test_models.synthetic import ConvolutionWithNotTensorBiasModel
from tests.torch.test_models.synthetic import MultiBranchesConnectedModel


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


def _capture_model(model: torch.nn.Module, inputs: torch.Tensor) -> torch.fx.GraphModule:
    with torch.no_grad():
        with disable_patching():
            return capture_pre_autograd_graph(model, (inputs,))


@pytest.mark.parametrize("test_case", MODEL_EXTRACTION_CASES, ids=idfn)
def test_model_extraction(test_case: ModelExtractionTestCase):
    captured_model = _capture_model(test_case.model(), torch.ones(test_case.input_shape))

    layout = TransformationLayout()
    layout.register(test_case.command)
    extracted_model = FXModelTransformer(captured_model).transform(layout)
    nncf_graph = GraphConverter.create_nncf_graph(extracted_model)
    check_graph(nncf_graph, f"{get_test_id(test_case)}.dot", EXTRACTED_GRAPHS_DIR_NAME, extended=True)


MultiBranchesConnectedModel_TARGET_POINTS = (
    PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, "conv2d", input_port_id=0),
    PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, "conv2d", input_port_id=1),
    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, "conv2d_1", input_port_id=1),
    PTTargetPoint(TargetType.OPERATOR_POST_HOOK, "conv2d"),
)


@pytest.mark.parametrize("leaf", [False, True], ids=["no_leaf", "leaf"])
def test_model_insertion_transformation(leaf: bool):
    class TestInsertModule(torch.nn.Module):
        def forward(self, x):
            return x + 1

    target_points = list(MultiBranchesConnectedModel_TARGET_POINTS)
    target_node_name = "TEST_MODULE"
    test_module_instance = TestInsertModule()
    builder = leaf_module_insertion_transformation_builder if leaf else module_insertion_transformation_builder
    transformation = builder(test_module_instance, target_points, target_node_name)

    model = MultiBranchesConnectedModel()
    captured_model = _capture_model(model, torch.ones((1, 3, 3, 3)))
    transformation(captured_model)

    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    assert getattr(captured_model, target_node_name) is test_module_instance
    check_graph(nncf_graph, f"model_insertion{'_leaf' if leaf else ''}.dot", TRANSFORMED_GRAPH_DIR_NAME, extended=True)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "constant"])
def test_constant_update_transformation(bias: bool):
    model = MultiBranchesConnectedModel()
    captured_model = _capture_model(model, torch.ones((1, 3, 3, 3)))
    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    target_node = nncf_graph.get_node_by_name("conv2d" if bias else "add_")

    builder = bias_update_transformation_builder if bias else constant_update_transformation_builder
    new_value = torch.tensor((42.0,))
    transformation = builder(target_node, value=new_value, input_port_id=1)
    transformation(captured_model)

    add_node = get_graph_node_by_name(captured_model.graph, "add_")
    assert get_tensor_constant_from_node(add_node.args[1], captured_model) == new_value

    transformed_nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    check_graph(transformed_nncf_graph, "constant_update.dot", TRANSFORMED_GRAPH_DIR_NAME, extended=True)


@pytest.mark.parametrize("bias", [True, False], ids=["bias", "constant"])
def test_constant_update_transformation_no_constant(bias: bool):
    model = MultiBranchesConnectedModel()
    captured_model = _capture_model(model, torch.ones((1, 3, 3, 3)))
    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    target_node = nncf_graph.get_node_by_name("add")

    builder = bias_update_transformation_builder if bias else constant_update_transformation_builder
    new_value = torch.tensor((42.0,))
    transformation = builder(target_node, value=new_value, input_port_id=1)
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
            dq_node = target_node.args[target_point.input_port_id]
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

    @pytest.mark.parametrize("target_point", MultiBranchesConnectedModel_TARGET_POINTS)
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

        model = MultiBranchesConnectedModel()
        captured_model = _capture_model(model, torch.ones((1, 3, 3, 3)))
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

        model = MultiBranchesConnectedModel()
        captured_model = _capture_model(model, torch.ones((1, 3, 3, 3)))
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
    captured_model = _capture_model(model, torch.ones((1, 3, 3, 3)))
    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    node = nncf_graph.get_node_by_name("conv2d")
    transformation = node_removal_transformation_builder(node, input_port_id=0)
    transformation(captured_model)
    transformed_nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    check_graph(transformed_nncf_graph, "node_removal_ref.dot", TRANSFORMED_GRAPH_DIR_NAME, extended=True)


@pytest.mark.parametrize("tuple_output", [False, True], ids=["node_out", "tuple_out"])
@pytest.mark.parametrize("target_point", MultiBranchesConnectedModel_TARGET_POINTS)
def test_output_insertion_transformation(tuple_output: bool, target_point: PTTargetPoint):
    model = MultiBranchesConnectedModel()
    captured_model = _capture_model(model, torch.ones((1, 3, 3, 3)))

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


def test_create_shared_constant_transformation():
    model = MultiBranchesConnectedModel()
    ex_inputs = torch.ones((1, 3, 3, 3))
    captured_model = _capture_model(model, ex_inputs)
    shared_constants_unification_transformation(captured_model)
    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    check_graph(
        nncf_graph, "shared_constants_unification_transformation_test.dot", TRANSFORMED_GRAPH_DIR_NAME, extended=True
    )


def get_shared_constant_nodes(nncf_graph: NNCFGraph):
    """
    Gets a dict of constant nodes as key and consumer nodes as values which are shared in the model.
    eg:
          const
          /   \
    node1     node2

    returns ({const:[node1, node2]})
    """
    shared_const_node_consumer_node = {}
    for node in nncf_graph.get_all_nodes():
        consumer_nodes = nncf_graph.get_next_nodes(node)
        if node.metatype in CONST_NOOP_METATYPES and len(consumer_nodes) > 1:
            shared_const_node_consumer_node[node] = consumer_nodes
    return shared_const_node_consumer_node


def insert_qdq_add_nodes(model: torch.fx.GraphModule):
    const_node = get_graph_node_by_name(model.graph, "_param_constant0")
    quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
    dequantize_op = torch.ops.quantized_decomposed.dequantize_per_channel.default
    add_op = torch.add
    conv_node = get_graph_node_by_name(model.graph, "conv2d")
    with model.graph.inserting_before(conv_node):
        scale_node = create_getattr_from_value(
            model,
            model.graph,
            "scale_node",
            torch.ones(
                [
                    3,
                ]
            ),
        )
        zp_node = create_getattr_from_value(
            model,
            model.graph,
            "weight_node",
            torch.ones(
                [
                    3,
                ]
            ),
        )
        qdq_args = (scale_node, zp_node, 0, -128, 127, torch.int8)
        q_node = model.graph.create_node("call_function", quantize_op, (const_node,) + qdq_args, {})
        add_node = model.graph.create_node("call_function", add_op, (q_node, 0), {})
        dq_node = model.graph.create_node("call_function", dequantize_op, (add_node,) + qdq_args, {})
    _set_new_node_meta(q_node, (const_node,) + qdq_args, quantize_op, model)
    _set_new_node_meta(add_node, (q_node, 0), add_op, model)
    _set_new_node_meta(dq_node, (add_node,) + qdq_args, dequantize_op, model)
    conv_node.replace_input_with(const_node, dq_node)


def test_different_qdq_pattern():
    model = MultiBranchesConnectedModel()
    ex_input = torch.ones(1, 3, 224, 224)
    captured_model = _capture_model(model, ex_input)
    quantized_before_insertion = nncf.quantize(captured_model, nncf.Dataset([ex_input]))
    q_before, dq_before = count_q_dq(quantized_before_insertion)
    insert_qdq_add_nodes(captured_model)
    quantized_after_insertion = nncf.quantize(captured_model, nncf.Dataset([ex_input]))
    q_after, dq_after = count_q_dq(quantized_after_insertion)
    assert q_before == 5
    assert dq_before == 6
    assert q_after == 6
    assert dq_after == 7


def test_update_shared_constant():
    model = MultiBranchesConnectedModel()
    ex_inputs = torch.ones((1, 3, 3, 3))
    captured_model = _capture_model(model, ex_inputs)

    shared_constants_unification_transformation(captured_model)
    nncf_graph = NNCFGraphFactory.create(captured_model)
    shared_constants_consumers_dict = get_shared_constant_nodes(nncf_graph)

    # This returns all the constant nodes as keys and list of consumer as values
    consumer_nodes = list(shared_constants_consumers_dict.values())[0]

    constant_update_transformation_builder(consumer_nodes[0], torch.tensor([100]))(captured_model)

    nncf_graph_updated_constant = NNCFGraphFactory.create(captured_model)
    updated_const_node = nncf_graph_updated_constant.get_previous_nodes(consumer_nodes[0])[1]
    fx_node_to_check_const = get_graph_node_by_name(captured_model.graph, updated_const_node.node_name)
    fx_node_to_check_const_value = get_tensor_constant_from_node(fx_node_to_check_const, captured_model)

    assert fx_node_to_check_const_value == torch.tensor([100])
