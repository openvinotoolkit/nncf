# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import openvino.torch  # noqa
import pytest
import torch
import torch.nn.parallel
import torchvision.models as models
from torch.export.dynamic_shapes import Dim

import nncf
from nncf.common.graph.graph import NNCFNodeName
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.utils.os import safe_open
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.experimental.torch.fx.quantization.backend_parameters import FXBackendParameters
from nncf.experimental.torch.fx.transformations import DEQUANTIZE_NODE_TARGETS
from nncf.experimental.torch.fx.transformations import _get_node_inputs
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch import test_models
from tests.torch.fx.helpers import get_torch_fx_model
from tests.torch.fx.test_sanity import count_q_dq
from tests.torch.test_models.synthetic import EmbeddingSumModel
from tests.torch.test_models.synthetic import MultiBranchesConnectedModel
from tests.torch.test_models.synthetic import ShortTransformer
from tests.torch.test_models.synthetic import YOLO11N_SDPABlock

FX_DIR_NAME = TEST_ROOT / "torch" / "data" / "fx"
FX_QUANTIZED_DIR_NAME = FX_DIR_NAME / "quantized"
FX_QUANTIZED_COMPRESSED_DIR_NAME = FX_DIR_NAME / "post_quantization_compressed"

FX_DYNAMIC_DIR = FX_DIR_NAME / "dynamic_shapes"
FX_DYNAMIC_QUANTIZED_DIR_NAME = FX_DYNAMIC_DIR / "quantized"
FX_DYNAMIC_QUANTIZED_COMPRESSED_DIR_NAME = FX_DYNAMIC_DIR / "post_quantization_compressed"


@dataclass
class ModelCase:
    model_builder: Callable[[], torch.nn.Module]
    model_id: str
    input_shape: tuple[int]


def torchvision_model_case(model_id: str, input_shape: tuple[int,]):
    model = getattr(models, model_id)
    return ModelCase(partial(model, weights=None), model_id, input_shape)


TEST_MODELS = (
    torchvision_model_case("resnet18", (1, 3, 224, 224)),
    torchvision_model_case("mobilenet_v3_small", (1, 3, 224, 224)),
    torchvision_model_case("vit_b_16", (1, 3, 224, 224)),
    torchvision_model_case("swin_v2_t", (1, 3, 224, 224)),
    ModelCase(test_models.UNet, "unet", [1, 3, 224, 224]),
    ModelCase(partial(ShortTransformer, 5, 10), "synthetic_transformer", [5]),
    ModelCase(YOLO11N_SDPABlock, "yolo11n_sdpa_block", YOLO11N_SDPABlock.INPUT_SIZE),
    ModelCase(EmbeddingSumModel, "embedding_bag_model", [1, 1]),
)


def get_dot_filename(model_name):
    return model_name + ".dot"


def get_json_filename(model_name):
    return model_name + ".json"


def get_full_path_to_json(model_json_name: str, attributes: bool = False) -> str:
    property_to_check = "reference_metatypes" if not attributes else "reference_attributes"
    path_to_dir = TEST_ROOT / "torch" / "data" / "fx" / property_to_check
    path_to_json = path_to_dir / model_json_name
    return path_to_json


def get_ref_from_json(
    model_name: str,
    model_metatypes: dict[NNCFNodeName, type[OperatorMetatype] | bool],
    regen_ref_data: bool,
    attributes=False,
) -> dict[NNCFNodeName, type[OperatorMetatype] | bool]:
    model_json_name = get_json_filename(model_name)
    complete_path = get_full_path_to_json(model_json_name, attributes)

    json_parent_dir = Path(complete_path).parent

    if regen_ref_data:
        if not os.path.exists(json_parent_dir):
            os.makedirs(json_parent_dir)
        with safe_open(complete_path, "w") as file:
            json.dump(model_metatypes, file, indent=4)

    with safe_open(complete_path, "r") as file:
        return json.load(file)


@pytest.mark.parametrize("test_case", TEST_MODELS, ids=[m.model_id for m in TEST_MODELS])
def test_model(test_case: ModelCase, regen_ref_data: bool):
    device = torch.device("cpu")
    model_name = test_case.model_id
    model = test_case.model_builder()
    model.to(device)

    dtype = torch.int32 if test_case.model_id in ["synthetic_transformer", "embedding_bag_model"] else torch.float32
    ex_input = torch.ones(test_case.input_shape, dtype=dtype)
    exported_model = get_torch_fx_model(model, ex_input)
    nncf_graph = GraphConverter.create_nncf_graph(exported_model)

    # Check NNCFGrpah
    dot_file_name = get_dot_filename(model_name)
    path_to_dot = FX_DIR_NAME / dot_file_name
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    compare_nx_graph_with_reference(nx_graph, path_to_dot.as_posix())

    # Check metatypes
    model_metatypes = {n.node_name: n.metatype.__name__ for n in nncf_graph.get_all_nodes()}
    ref_metatypes = get_ref_from_json(model_name, model_metatypes, regen_ref_data=regen_ref_data)
    assert model_metatypes == ref_metatypes


TEST_MODELS_QUANIZED = (
    (
        ModelCase(test_models.UNet, "unet", [1, 3, 224, 224]),
        {},
        [(46, 50), (23, 27)],
        [Dim.AUTO, Dim.STATIC, Dim.STATIC, Dim.STATIC],  # This Unet Model is not eligible for dynamic shape capability
    ),
    (
        torchvision_model_case("resnet18", (1, 3, 224, 224)),
        {},
        [(51, 58), (30, 37)],
        [Dim.AUTO, Dim.STATIC, Dim.AUTO, Dim.AUTO],
    ),
    (
        torchvision_model_case("mobilenet_v3_small", (1, 3, 224, 224)),
        {},
        [(97, 112), (61, 76)],
        [Dim.AUTO, Dim.STATIC, Dim.AUTO, Dim.AUTO],
    ),
    (
        torchvision_model_case("vit_b_16", (1, 3, 224, 224)),
        {"model_type": nncf.ModelType.TRANSFORMER},
        [(124, 124), (74, 74)],
        [Dim.AUTO, Dim.STATIC, Dim.STATIC, Dim.STATIC],  # This ViT Model is not eligible for dynamic shape capability
    ),
    (
        torchvision_model_case("swin_v2_t", (1, 3, 224, 224)),
        {"model_type": nncf.ModelType.TRANSFORMER},
        [
            (130, 130),
            (77, 77),
        ],
        [Dim.AUTO, Dim.STATIC, Dim.AUTO, Dim.AUTO],
    ),
    (
        ModelCase(partial(ShortTransformer, 5, 10), "synthetic_transformer", [5]),
        {"model_type": nncf.ModelType.TRANSFORMER},
        [(4, 4), (2, 2)],
        [Dim.AUTO],
    ),
    (
        ModelCase(YOLO11N_SDPABlock, "yolo11n_sdpa_block", YOLO11N_SDPABlock.INPUT_SIZE),
        {"model_type": nncf.ModelType.TRANSFORMER},
        [(4, 4), (3, 3)],
        # (dlyakhov) Last dim has to be static, without that an assert is
        # being inserted to the fx graph to check the last dim is equal to 4
        [Dim.AUTO, Dim.AUTO, Dim.STATIC],
    ),
)


@pytest.mark.parametrize("enable_dynamic_shapes", [True, False])
@pytest.mark.parametrize("compress_weights", [True, False])
@pytest.mark.parametrize(
    ("model_case", "quantization_parameters", "compress_n_qdq", "dynamic_shape_config"),
    TEST_MODELS_QUANIZED,
    ids=[m[0].model_id for m in TEST_MODELS_QUANIZED],
)
def test_quantized_model(
    model_case: ModelCase,
    quantization_parameters,
    compress_weights: bool,
    compress_n_qdq: int,
    enable_dynamic_shapes: bool,
    dynamic_shape_config: list[bool],
):
    model = model_case.model_builder()
    dtype = torch.int32 if model_case.model_id == "synthetic_transformer" else torch.float32
    example_input = torch.ones(model_case.input_shape, dtype=dtype)
    dynamic_shapes = None
    if enable_dynamic_shapes:
        dynamic_shapes = [tuple(dynamic_shape_config)]

    fx_model = get_torch_fx_model(model, example_input, dynamic_shapes=dynamic_shapes)

    def transform_fn(data_item):
        return data_item.to("cpu")

    calibration_dataset = nncf.Dataset([example_input], transform_fn)

    quantization_parameters["advanced_parameters"] = AdvancedQuantizationParameters(
        disable_bias_correction=True, backend_params={FXBackendParameters.COMPRESS_WEIGHTS: compress_weights}
    )
    quantization_parameters["subset_size"] = 1

    quantized_model = nncf.quantize(fx_model, calibration_dataset, **quantization_parameters)
    # Uncomment to visualize torch fx graph
    # from tests.torch.fx.helpers import visualize_fx_model
    # visualize_fx_model(quantized_model, f"{model_case.model_id}_int8.svg")
    if dynamic_shapes:
        save_dir = FX_DYNAMIC_QUANTIZED_COMPRESSED_DIR_NAME if compress_weights else FX_DYNAMIC_QUANTIZED_DIR_NAME
    else:
        save_dir = FX_QUANTIZED_COMPRESSED_DIR_NAME if compress_weights else FX_QUANTIZED_DIR_NAME

    path_to_dot = save_dir / get_dot_filename(model_case.model_id)
    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    compare_nx_graph_with_reference(nx_graph, path_to_dot.as_posix())

    q_nodes, dq_nodes = count_q_dq(quantized_model)
    assert q_nodes == compress_n_qdq[compress_weights][0]
    assert dq_nodes == compress_n_qdq[compress_weights][1]
    check_fq_values(quantized_model)
    check_compressed_post_quantized(quantized_model)


def check_fq_values(quantized_model):
    for node in quantized_model.graph.nodes:
        if node.target not in DEQUANTIZE_NODE_TARGETS:
            continue
        args = []
        quantize_node = node.args[0]
        args = _get_node_inputs(quantize_node, quantized_model)
        if not args:
            continue
        result = node.target(quantize_node.target(*args), *args[1:])
        fp_value = get_tensor_constant_from_node(quantize_node.args[0], quantized_model)
        assert torch.all(result == fp_value)


def check_compressed_post_quantized(quantized_model):
    for node in quantized_model.graph.nodes:
        if node.name[:3] != "mul":
            continue
        args = []
        args = _get_node_inputs(node, quantized_model)
        if not args:
            continue
        assert args[0].dtype == torch.int8
        result = node.target(*args)
        assert result.dtype == torch.float32


def test_is_shared_attribute_default(regen_ref_data: bool):
    model = MultiBranchesConnectedModel()
    ex_inputs = torch.ones((1, 3, 3, 3))
    fx_model = get_torch_fx_model(model, ex_inputs)
    nncf_graph = GraphConverter.create_nncf_graph(fx_model)

    shared_attributes = {n.node_name: n.is_shared() for n in nncf_graph.get_all_nodes()}
    ref_attributes = get_ref_from_json(
        "default_shared_attribute_test_model", shared_attributes, attributes=True, regen_ref_data=regen_ref_data
    )
    assert shared_attributes == ref_attributes
