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

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.torch.fx.model_transformer import FXModelTransformer
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.transformations import output_insertion_transformation_builder
from nncf.torch import disable_patching
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
    check_graph(nncf_graph, f"{get_test_id(test_case)}.dot", str(EXTRACTED_GRAPHS_DIR_NAME))


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
        nncf_graph, f"output_insertion_{_target_point_to_str(target_point)}_ref.dot", TRANSFORMED_GRAPH_DIR_NAME
    )
