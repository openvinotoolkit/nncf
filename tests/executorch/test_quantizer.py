# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional

import dataclasses
import json
from enum import Enum

import pytest
import torch
import torch.fx

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.utils.os import safe_open
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx import compress_pt2e

from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch.test_models.synthetic import ShortTransformer
from tests.torch.test_models.llama import LlamaDecoderOnly
from tests.torch2.fx.helpers import get_torch_fx_model

from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

from executorch.backends.openvino.quantizer.quantizer import (
    OpenVINOQuantizer,
    QuantizationMode,
)
from nncf.common.graph.graph import NNCFNode

FX_PT2E_DIR = TEST_ROOT / "torch2" / "data" / "fx" / "compress_pt2e"
FX_AO_DIR   = TEST_ROOT / "torch2" / "data" / "fx" / "ao_compression_OpenVINOQuantizer"


@dataclass
class ModelCase:
    model_builder: Callable[[], torch.nn.Module]
    model_id: str
    input_shape: tuple[int, ...]


def get_dot_filename(model_name: str) -> str:
    return model_name + ".dot"


def get_wc_param_filename(model_name: str) -> str:
    return model_name + "_ref_wc_param.json"


def _build_torch_fx_model(model_case: ModelCase) -> tuple[torch.fx.GraphModule, torch.Tensor]:
    model = model_case.model_builder()
    # ShortTransformer takes token ids; match prior synthetic tests (int32)
    example_input = torch.ones(model_case.input_shape, dtype=torch.int32)
    fx_model = get_torch_fx_model(model, example_input)
    return fx_model, example_input


def _get_calibration_dataset(example_input: torch.Tensor) -> nncf.Dataset:
    def transform_fn(x):
        return x.to("cpu")
    return nncf.Dataset([example_input], transform_fn)


def get_openvino_quantizer(*args, **kwargs) -> OpenVINOQuantizer:
    return OpenVINOQuantizer(*args, **kwargs)


def _string_from_quantizer_params(qparams: dict[str, Any], pt2e_param: Optional[dict[str, Any]] = None) -> str:
    mode = qparams.get("mode")
    gs = qparams.get("group_size", "-1")
    ratio = qparams.get("ratio", "1")
    all_layers = qparams.get("all_layers", "False")
    if(pt2e_param is None):
        return f"{mode.value}_gs{gs}_ratio{ratio}_all_layers_{all_layers}"
    sensitivity_metric = pt2e_param.get("sensitivity_metric", "None")
    return f"{mode.value}_gs{gs}_ratio{ratio}_all_layers_{all_layers}_sensitivity_metric_{sensitivity_metric}"


BASE_MODELS = (
    ModelCase(LlamaDecoderOnly, "LlamaDecoderOnly", [1,3,64]),
    ModelCase(partial(ShortTransformer, 64, 128, True), "short_transformer_shared", [5]),
)

QUANTIZER_PARAMS = (
    {"mode": QuantizationMode.INT8WO_ASYM},
    {"mode": QuantizationMode.INT4WO_SYM, "group_size": 32, "ratio": 0.8},
    {"mode": QuantizationMode.INT4WO_SYM, "group_size": 32, "ratio": 0.8, "all_layers": True},
)

PT2E_PARAMS = (
    {"sensitivity_metric": nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION},
    {"sensitivity_metric": nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE},
    {"sensitivity_metric": nncf.SensitivityMetric.WEIGHT_QUANTIZATION_ERROR},
    {"sensitivity_metric": nncf.SensitivityMetric.MEAN_ACTIVATION_VARIANCE},
    {"sensitivity_metric": nncf.SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE},
)


TEST_MODELS = tuple(
    (model, qparam, pt2e_param)
    for model in BASE_MODELS
    for qparam in QUANTIZER_PARAMS
    for pt2e_param in (
        [{}] 
        if (
            (qparam.get("mode") in {QuantizationMode.INT8WO_ASYM, QuantizationMode.INT8WO_SYM})
            or (qparam.get("ratio") is None)
        )
        else PT2E_PARAMS
    )
)


TEST_MODEL_IDS = [
    f"{m.model_id}__{_string_from_quantizer_params(qparams, pt2e_param)}" for (m, qparams, pt2e_param) in TEST_MODELS
]


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

def test_compress_pt2e(
    quantizer_builder: Callable[..., OpenVINOQuantizer],
    model_case: ModelCase,
    quantizer_params,
    pt2e_params,
):
    fx_model, example_input = _build_torch_fx_model(model_case)
    with torch.no_grad():
        ref_out = fx_model(example_input)

    calibration_dataset = _get_calibration_dataset(example_input)

    # Build quantizer directly from quantizer_params (already includes mode/group_size)
    quantizer = quantizer_builder(**quantizer_params)

    quantized_model = compress_pt2e(
        fx_model,
        quantizer=quantizer,
        dataset=calibration_dataset,
        **pt2e_params,
    )

    with torch.no_grad():
        out = quantized_model(example_input)
    assert out.shape == ref_out.shape, "Compressed model output shape mismatch."

    nncf_graph: NNCFGraph = GraphConverter.create_nncf_graph(quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    param_string = _string_from_quantizer_params(quantizer_params, pt2e_params)
    path_to_dot = (FX_PT2E_DIR / quantizer.__class__.__name__ / model_case.model_id / get_dot_filename(param_string)).as_posix()
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
def test_openvino_quantizer(
    model_case: ModelCase,
    quantizer_params,
    quantizer_builder: Callable[..., OpenVINOQuantizer],
    pt2e_params,
):
    fx_model, example_input = _build_torch_fx_model(model_case)
    quantizer = quantizer_builder(**quantizer_params)

    prepared = prepare_pt2e(fx_model, quantizer)
    prepared(example_input)
    ao_quantized_model = convert_pt2e(prepared)

    nncf_graph = GraphConverter.create_nncf_graph(ao_quantized_model)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)

    param_string = _string_from_quantizer_params(quantizer_params)
    path_to_dot = (FX_AO_DIR / model_case.model_id / get_dot_filename(param_string)).as_posix()
    compare_nx_graph_with_reference(nx_graph, path_to_dot)


def _serialize_wc_param(wp) -> dict[str, Any]:
    def to_json_serializable(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_json_serializable(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (list, tuple)):
            return [to_json_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, NNCFNode):
            return obj.node_name
        else:
            return obj

    return to_json_serializable(wp)

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
def test_openvino_wc_params(
    quantizer_builder: Callable[..., OpenVINOQuantizer],
    model_case: ModelCase,
    quantizer_params,
    pt2e_params,
    regen_ref_data=False,
):
    fx_model, _ = _build_torch_fx_model(model_case)
    nncf_graph: NNCFGraph = GraphConverter.create_nncf_graph(fx_model)

    param_string = _string_from_quantizer_params(quantizer_params)
    quantizer = quantizer_builder(**quantizer_params)

    all_weight_params, *_ = quantizer.get_nncf_weight_compression_parameters(fx_model, nncf_graph)

    wc_params = _serialize_wc_param(all_weight_params)

    ref_json_path = (FX_PT2E_DIR / quantizer.__class__.__name__ / model_case.model_id / get_wc_param_filename(param_string))

    if regen_ref_data:
        with safe_open(ref_json_path, "w") as file:
            json.dump(wc_params, file, indent=4)

    with safe_open(ref_json_path, "r") as f:
        ref_data = json.load(f)

    assert wc_params == ref_data, (
        f"Weight compression parameters JSON mismatch for {model_case.model_id} ({param_string}).\n"
        f"Ref: {ref_json_path}"
    )
