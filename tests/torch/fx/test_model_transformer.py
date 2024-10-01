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
from torch._export import capture_pre_autograd_graph

from nncf.common.factory import NNCFGraph
from nncf.common.factory import NNCFGraphFactory
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.torch.fx.model_transformer import FXModelTransformer
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.experimental.torch.fx.transformations import constant_update_transformation_builder
from nncf.experimental.torch.fx.transformations import output_insertion_transformation_builder
from nncf.experimental.torch.fx.transformations import shared_constants_unification_transformation
from nncf.torch import disable_patching
from nncf.torch.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from tests.torch.test_compressed_graph import check_graph
from tests.torch.test_models.synthetic import ConvolutionWithAllConstantInputsModel
from tests.torch.test_models.synthetic import ConvolutionWithNotTensorBiasModel
from tests.torch.test_models.synthetic import MultiBranchesConnectedModel


@dataclass
class ModelExtractionTestCase:
    model: torch.nn.Module
    input_shape: Tuple[int, ...]
    command: PTModelExtractionCommand


EXTRACTED_GRAPHS_DIR_NAME = Path("fx") / "extracted"
TRANSFORMED_GRAPH_DIR_NAME = Path("fx") / "transformed"

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
    check_graph(nncf_graph, f"{get_test_id(test_case)}.dot", str(EXTRACTED_GRAPHS_DIR_NAME), extended=True)


MultiBranchesConnectedModel_TARGET_POINTS = (
    PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, "conv2d", input_port_id=0),
    PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, "conv2d", input_port_id=1),
    PTTargetPoint(TargetType.OPERATION_WITH_WEIGHTS, "conv2d_1", input_port_id=1),
    PTTargetPoint(TargetType.OPERATOR_POST_HOOK, "conv2d"),
)


@pytest.mark.parametrize("tuple_output", [False, True], ids=["node_out", "tuple_out"])
@pytest.mark.parametrize("target_point", MultiBranchesConnectedModel_TARGET_POINTS)
def test_output_insertion_transformation(tuple_output, target_point):
    model = MultiBranchesConnectedModel()
    captured_model = _capture_model(model, torch.ones((1, 3, 3, 3)))

    if not tuple_output:
        output_node = [node for node in captured_model.graph.nodes if node.op == "output"][0]
        output_node.args = (output_node.args[0][0],)
        captured_model.recompile()

    transformation = output_insertion_transformation_builder(target_point)
    transformation(captured_model)

    nncf_graph = GraphConverter.create_nncf_graph(captured_model)
    check_graph(
        nncf_graph,
        f"output_insertion_{_target_point_to_str(target_point)}_ref.dot",
        TRANSFORMED_GRAPH_DIR_NAME,
        extended=True,
    )


def count_constants(model) -> int:
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
    updated_const_node = nncf_graph_updated_constant.get_previous_nodes(consumer_nodes[1])[1]
    fx_node_to_check_const = get_graph_node_by_name(captured_model.graph, updated_const_node.node_name)
    fx_node_to_check_const_value = get_tensor_constant_from_node(fx_node_to_check_const, captured_model)

    assert fx_node_to_check_const_value == torch.tensor([100])
