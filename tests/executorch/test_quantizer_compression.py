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


import dataclasses
import json
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Optional

import pytest
import torch
import torch.fx
from executorch.backends.openvino.quantizer.quantizer import OpenVINOQuantizer
from executorch.backends.openvino.quantizer.quantizer import QuantizationMode
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.utils.os import safe_open
from nncf.experimental.torch.fx import compress_pt2e
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT4SymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.test_models.llama import LlamaDecoderOnly
from tests.torch.test_models.synthetic import ShortTransformer
from tests.torch2.fx.helpers import get_torch_fx_model

FX_PT2E_DIR = TEST_ROOT / "executorch" / "data" / "fx" / "compress_pt2e"
FX_AO_DIR = TEST_ROOT / "executorch" / "data" / "fx" / "ao_export_compression_OpenVINOQuantizer"
INT8_COMPRESSION_MODES = [QuantizationMode.INT8WO_ASYM, QuantizationMode.INT8WO_SYM]


@dataclass
class ModelCase:
    model_builder: Callable[[], torch.nn.Module]
    model_id: str
    input_shape: tuple[int, ...]


def get_dot_filename(model_name: str) -> str:
    return model_name + ".dot"


def get_wc_param_filename(model_name: str) -> str:
    return model_name + "_ref_wc_param.json"


def get_wc_scales_filename(model_name: str) -> str:
    return model_name + "_ref_wc_scales.json"


def build_torch_fx_model(model_case: ModelCase) -> tuple[torch.fx.GraphModule, torch.Tensor]:
    model = model_case.model_builder()
    # ShortTransformer takes token ids; match prior synthetic tests (int32)
    example_input = torch.ones(model_case.input_shape, dtype=torch.int32)
    fx_model = get_torch_fx_model(model, example_input)
    return fx_model, example_input


def _get_calibration_dataset(example_input: torch.Tensor) -> nncf.Dataset:
    torch.manual_seed(42)

    def transform_fn(x):
        return x.to("cpu")

    sample_1 = torch.randint_like(example_input, 0, 10)
    sample_2 = torch.randint_like(example_input, 0, 10)
    return nncf.Dataset([example_input, sample_1, sample_2], transform_fn)


def get_openvino_quantizer(*args, **kwargs) -> OpenVINOQuantizer:
    return OpenVINOQuantizer(*args, **kwargs)


def _string_from_quantizer_params(qparams: dict[str, Any], pt2e_param: Optional[dict[str, Any]] = None) -> str:
    mode = qparams.get("mode")
    gs = qparams.get("group_size", "-1")
    all_layers = qparams.get("all_layers", "False")
    if pt2e_param is None:
        return f"{mode.value}_gs{gs}_all_layers_{all_layers}"
    awq = pt2e_param.get("awq", "False")
    scale_estimation = pt2e_param.get("scale_estimation", "False")
    return f"{mode.value}_gs{gs}_all_layers_{all_layers}_awq_{awq}_scale_estimation_{scale_estimation}"


def check_multiple_isinstance(object_to_check: Any, objects: list[Any]) -> bool:
    if not object_to_check:
        return False
    for obj in objects:
        if isinstance(object_to_check, obj):
            return True
    return False


def get_scale_values_from_model(model: torch.fx.GraphModule) -> dict[str, torch.Tensor]:
    node_to_scale_mapping = {}
    decompressor_modules = [
        INT4AsymmetricWeightsDecompressor,
        INT4SymmetricWeightsDecompressor,
        INT8AsymmetricWeightsDecompressor,
        INT8SymmetricWeightsDecompressor,
    ]
    for node in model.graph.nodes:
        node_module = getattr(model, node.target) if node.op == "call_module" else None
        if not check_multiple_isinstance(node_module, decompressor_modules):
            continue
        state_dict_scale_name = f"{node.target}._scale"
        node_to_scale_mapping[node.name] = model.state_dict()[state_dict_scale_name]

    return node_to_scale_mapping


def get_test_cases() -> list[ModelCase, tuple[dict[str, Any], ...], dict[str, Any]]:
    test_cases = []
    for model in BASE_MODELS:
        for qparam in QUANTIZER_PARAMS:
            pt2e_params = [{}] if qparam.get("mode") in INT8_COMPRESSION_MODES else PT2E_PARAMS
            for pt2e_param in pt2e_params:
                test_cases.append(
                    (
                        model,
                        qparam,
                        pt2e_param,
                    )
                )
    return test_cases


def get_test_model_ids_from_test_cases(
    test_models: list[tuple[ModelCase, tuple[dict[str, Any], ...], Optional[dict[str, Any]]]],
) -> list[str]:
    model_ids = []
    for item in test_models:
        m, qparams, *other_params = item
        pt2e_params = other_params[0] if other_params else None
        model_ids.append(f"{m.model_id}__{_string_from_quantizer_params(qparams, pt2e_params)}")
    return model_ids


BASE_MODELS = (
    ModelCase(LlamaDecoderOnly, "LlamaDecoderOnly", (1, 3, 64)),
    ModelCase(partial(ShortTransformer, 64, 128, True), "short_transformer_shared", (5,)),
)

QUANTIZER_PARAMS = (
    {"mode": QuantizationMode.INT8WO_ASYM},
    {"mode": QuantizationMode.INT4WO_SYM, "group_size": 32},
    {"mode": QuantizationMode.INT4WO_SYM, "group_size": 32, "all_layers": True},
)

PT2E_PARAMS = ({"awq": True, "scale_estimation": True},)


TEST_MODELS = get_test_cases()

TEST_MODEL_IDS = get_test_model_ids_from_test_cases(TEST_MODELS)

TEST_MODELS_NO_PT2E = [(m, qparams) for m, qparams, _ in TEST_MODELS]

TEST_MODEL_IDS_NO_PT2E = get_test_model_ids_from_test_cases(TEST_MODELS_NO_PT2E)


@pytest.mark.parametrize(
    ("model_case", "quantizer_params"),
    TEST_MODELS_NO_PT2E,
    ids=TEST_MODEL_IDS_NO_PT2E,
)
@pytest.mark.parametrize(
    "quantizer_builder",
    [get_openvino_quantizer],
    ids=["OpenVINOQuantizer"],
)
def test_compress_pt2e(
    quantizer_builder: Callable[..., OpenVINOQuantizer],
    model_case: ModelCase,
    quantizer_params: tuple[dict[str, Any], ...],
):
    fx_model, example_input = build_torch_fx_model(model_case)
    with torch.no_grad():
        ref_out = fx_model(example_input)

    calibration_dataset = _get_calibration_dataset(example_input)

    # Build quantizer directly from quantizer_params (already includes mode/group_size)
    quantizer = quantizer_builder(**quantizer_params)
    mode = quantizer_params.get("mode")
    ratio = 1 if mode in INT8_COMPRESSION_MODES else 0.8

    quantized_model = compress_pt2e(fx_model, quantizer=quantizer, ratio=ratio, dataset=calibration_dataset)

    with torch.no_grad():
        out = quantized_model(example_input)
    assert out.shape == ref_out.shape, "Compressed model output shape mismatch."

    nncf_graph: NNCFGraph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    param_string = _string_from_quantizer_params(quantizer_params)
    path_to_dot = (
        FX_PT2E_DIR / quantizer.__class__.__name__ / model_case.model_id / get_dot_filename(param_string)
    ).as_posix()
    compare_nx_graph_with_reference(nx_graph, path_to_dot)


@pytest.mark.parametrize(
    ("model_case", "quantizer_params", "pt2e_params"),
    TEST_MODELS,
    ids=TEST_MODEL_IDS,
)
@pytest.mark.parametrize(
    "quantizer_builder",
    [get_openvino_quantizer],
    ids=["OpenVINOQuantizer"],
)
def test_compress_pt2e_scales(
    quantizer_builder: Callable[..., OpenVINOQuantizer],
    model_case: ModelCase,
    quantizer_params: tuple[dict[str, Any], ...],
    pt2e_params: dict[str, Any],
    regen_ref_data: bool,
):
    fx_model, example_input = build_torch_fx_model(model_case)
    with torch.no_grad():
        ref_out = fx_model(example_input)

    calibration_dataset = _get_calibration_dataset(example_input)

    # Build quantizer directly from quantizer_params (already includes mode/group_size)
    quantizer = quantizer_builder(**quantizer_params)
    mode = quantizer_params.get("mode")
    ratio = 1 if mode in INT8_COMPRESSION_MODES else 0.8
    quantized_model = compress_pt2e(
        fx_model, quantizer=quantizer, ratio=ratio, dataset=calibration_dataset, **pt2e_params
    )

    with torch.no_grad():
        out = quantized_model(example_input)
    assert out.shape == ref_out.shape, "Compressed model output shape mismatch."

    param_string = _string_from_quantizer_params(quantizer_params, pt2e_params)
    ref_json_path = (
        FX_PT2E_DIR / quantizer.__class__.__name__ / model_case.model_id / get_wc_scales_filename(param_string)
    )

    scales_list = get_scale_values_from_model(quantized_model)
    scales_list = to_json_serializable(scales_list)

    if regen_ref_data:
        with safe_open(ref_json_path, "w") as file:
            json.dump(scales_list, file, indent=4)

    with safe_open(ref_json_path, "r") as f:
        json.load(f)


@pytest.mark.parametrize(
    ("model_case", "quantizer_params"),
    TEST_MODELS_NO_PT2E,
    ids=TEST_MODEL_IDS_NO_PT2E,
)
@pytest.mark.parametrize(
    "quantizer_builder",
    [get_openvino_quantizer],
    ids=["OpenVINOQuantizer"],
)
def test_openvino_quantizer(
    model_case: ModelCase,
    quantizer_params: tuple[dict[str, Any], ...],
    quantizer_builder: Callable[..., OpenVINOQuantizer],
):
    fx_model, example_input = build_torch_fx_model(model_case)
    quantizer = quantizer_builder(**quantizer_params)

    prepared = prepare_pt2e(fx_model, quantizer)
    prepared(example_input)
    ao_quantized_model = convert_pt2e(prepared)

    nncf_graph = GraphConverter.create_nncf_graph(ao_quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    param_string = _string_from_quantizer_params(quantizer_params)
    path_to_dot = (FX_AO_DIR / model_case.model_id / get_dot_filename(param_string)).as_posix()
    compare_nx_graph_with_reference(nx_graph, path_to_dot)


def to_json_serializable(obj: Any) -> dict[Any, Any]:
    if dataclasses.is_dataclass(obj):
        return {k: to_json_serializable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, NNCFNode):
        return obj.node_name
    return obj


@pytest.mark.parametrize(
    ("model_case", "quantizer_params"),
    TEST_MODELS_NO_PT2E,
    ids=TEST_MODEL_IDS_NO_PT2E,
)
@pytest.mark.parametrize(
    "quantizer_builder",
    [get_openvino_quantizer],
    ids=["OpenVINOQuantizer"],
)
def test_openvino_wc_params(
    quantizer_builder: Callable[..., OpenVINOQuantizer],
    model_case: ModelCase,
    quantizer_params: tuple[dict[str, Any], ...],
    regen_ref_data: bool,
):
    fx_model, _ = build_torch_fx_model(model_case)
    nncf_graph: NNCFGraph = GraphConverter.create_nncf_graph(fx_model)

    param_string = _string_from_quantizer_params(quantizer_params)
    quantizer = quantizer_builder(**quantizer_params)

    all_weight_params, *_ = quantizer.get_nncf_weight_compression_parameters(fx_model, nncf_graph)

    wc_params = to_json_serializable(all_weight_params)

    ref_json_path = (
        FX_PT2E_DIR / quantizer.__class__.__name__ / model_case.model_id / get_wc_param_filename(param_string)
    )

    if regen_ref_data:
        with safe_open(ref_json_path, "w") as file:
            json.dump(wc_params, file, indent=4)

    with safe_open(ref_json_path, "r") as f:
        ref_data = json.load(f)

    assert wc_params == ref_data, (
        f"Weight compression parameters JSON mismatch for {model_case.model_id} ({param_string}).\nRef: {ref_json_path}"
    )
