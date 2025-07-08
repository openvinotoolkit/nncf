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

import json
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import pytest
import torch
import torch.fx
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.ao.quantization.pt2e.utils import _fuse_conv_bn_
from torch.ao.quantization.quantize_pt2e import convert_pt2e
from torch.ao.quantization.quantize_pt2e import prepare_pt2e
from torch.ao.quantization.quantizer import xnnpack_quantizer
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec as TorchAOQuantizationSpec
from torch.ao.quantization.quantizer.quantizer import Quantizer
from torch.ao.quantization.quantizer.quantizer import SharedQuantizationSpec as TorchAOSharedQuantizationSpec
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.ao.quantization.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.utils.os import safe_open
from nncf.experimental.torch.fx import quantize_pt2e
from nncf.experimental.torch.fx.nncf_graph_builder import GraphConverter
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.experimental.torch.fx.quantization.quantizer.openvino_adapter import OpenVINOQuantizerAdapter
from nncf.experimental.torch.fx.quantization.quantizer.openvino_quantizer import OpenVINOQuantizer
from nncf.experimental.torch.fx.quantization.quantizer.torch_ao_adapter import TorchAOQuantizerAdapter
from nncf.experimental.torch.fx.quantization.quantizer.torch_ao_adapter import _get_edge_or_node_to_qspec
from nncf.tensor.definitions import TensorDataType
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch import test_models
from tests.torch.test_models.synthetic import ShortTransformer
from tests.torch.test_models.synthetic import SimpleConcatModel
from tests.torch.test_models.synthetic import YOLO11N_SDPABlock
from tests.torch2.fx.helpers import get_torch_fx_model

FX_QUANTIZED_DIR_NAME = TEST_ROOT / "torch2" / "data" / "fx"


@dataclass
class ModelCase:
    model_builder: Callable[[], torch.nn.Module]
    model_id: str
    input_shape: tuple[int]


def torchvision_model_case(model_id: str, input_shape: tuple[int,]):
    model = getattr(models, model_id)
    return ModelCase(partial(model, weights=None), model_id, input_shape)


def get_dot_filename(model_name: str) -> str:
    return model_name + ".dot"


def get_qconf_filename(model_name: str) -> str:
    return model_name + "_ref_qconfig.json"


def get_x86_quantizer(*args, **kwarsg) -> X86InductorQuantizer:
    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())
    return quantizer


def get_xnnpack_quantizer(*args, **kwargs) -> xnnpack_quantizer.XNNPACKQuantizer:
    quantizer = xnnpack_quantizer.XNNPACKQuantizer()
    quantizer.set_global(xnnpack_quantizer.get_symmetric_quantization_config())
    return quantizer


def get_openvino_quantizer(*args, **kwargs) -> OpenVINOQuantizer:
    return OpenVINOQuantizer(*args, **kwargs)


TEST_MODELS_QUANIZED = (
    (ModelCase(test_models.UNet, "unet", [1, 3, 224, 224]), {}, {}),
    (torchvision_model_case("resnet18", (1, 3, 224, 224)), {}, {}),
    (torchvision_model_case("mobilenet_v3_small", (1, 3, 224, 224)), {}, {}),
    (
        torchvision_model_case("vit_b_16", (1, 3, 224, 224)),
        {"model_type": nncf.ModelType.TRANSFORMER},
        {"smooth_quant": True},
    ),
    (
        torchvision_model_case("swin_v2_t", (1, 3, 224, 224)),
        {"model_type": nncf.ModelType.TRANSFORMER},
        {"smooth_quant": True},
    ),
    (
        ModelCase(partial(ShortTransformer, 5, 10), "synthetic_transformer", [5]),
        {"model_type": nncf.ModelType.TRANSFORMER},
        {"smooth_quant": True},
    ),
    (
        ModelCase(YOLO11N_SDPABlock, "yolo11n_sdpa_block", YOLO11N_SDPABlock.INPUT_SIZE),
        {"model_type": nncf.ModelType.TRANSFORMER},
        {"smooth_quant": True},
    ),
)


def _build_torch_fx_model(model_case: ModelCase) -> tuple[torch.fx.GraphModule, torch.Tensor]:
    model = model_case.model_builder()
    dtype = torch.int32 if model_case.model_id == "synthetic_transformer" else torch.float32
    example_input = torch.ones(model_case.input_shape, dtype=dtype)
    fx_model = get_torch_fx_model(model, example_input)
    return fx_model, example_input


def _get_calibration_dataset(example_input: torch.Tensor) -> nncf.Dataset:
    def transform_fn(data_item):
        return data_item.to("cpu")

    return nncf.Dataset([example_input], transform_fn)


@pytest.mark.parametrize(
    ("model_case", "quantizer_params", "pt2e_params"),
    TEST_MODELS_QUANIZED,
    ids=[m[0].model_id for m in TEST_MODELS_QUANIZED],
)
@pytest.mark.parametrize(
    "quantizer_builder",
    [
        get_xnnpack_quantizer,
        get_x86_quantizer,
        get_openvino_quantizer,
    ],
    ids=["XNNPACKQuantizer", "X86InductorQuantizer", "OpenVINOQuantizer"],
)
def test_quantized_model(
    quantizer_builder: Callable[[tuple[Any, ...]], Quantizer],
    model_case: ModelCase,
    quantizer_params,
    pt2e_params,
):
    fx_model, example_input = _build_torch_fx_model(model_case)
    calibration_dataset = _get_calibration_dataset(example_input)

    quantizer = quantizer_builder(**quantizer_params)
    quantized_model = quantize_pt2e(
        fx_model,
        quantizer,
        calibration_dataset=calibration_dataset,
        fast_bias_correction=None,  # BC is disabled
        fold_quantize=True,
        do_copy=True,
        **pt2e_params,
    )

    # Uncomment to visualize torch fx graph
    # from tests.torch2.fx.helpers import visualize_fx_model
    # visualize_fx_model(quantized_model, f"{quantizer.__class__.__name__}_{model_case.model_id}_int8.svg")

    nncf_graph = GraphConverter.create_nncf_graph(quantized_model)
    path_to_dot = FX_QUANTIZED_DIR_NAME / str(quantizer.__class__.__name__) / get_dot_filename(model_case.model_id)
    nncf_graph = _normalize_nncf_graph(nncf_graph, quantized_model.graph)
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    compare_nx_graph_with_reference(nx_graph, path_to_dot.as_posix())

    # Uncomment to visualize reference graphs
    # from torch.ao.quantization.quantize_pt2e import convert_pt2e
    # from torch.ao.quantization.quantize_pt2e import prepare_pt2e
    # from tests.torch2.fx.helpers import visualize_fx_model
    # prepared_model = prepare_pt2e(fx_model, quantizer)
    # prepared_model(example_input)
    # ao_quantized_model = convert_pt2e(prepared_model)
    # visualize_fx_model(ao_quantized_model, f"{quantizer.__class__.__name__}_{model_case.model_id}_ao_int8.svg")
    # ao_nncf_graph = GraphConverter.create_nncf_graph(ao_quantized_model)
    # ao_nncf_graph.visualize_graph(f"ao_{quantizer.__class__.__name__}_{get_dot_filename(model_case.model_id)}")


@pytest.mark.parametrize(
    ("model_case", "quantizer_params"),
    [case[:2] for case in TEST_MODELS_QUANIZED],
    ids=[m[0].model_id for m in TEST_MODELS_QUANIZED],
)
@pytest.mark.parametrize(
    "quantizer_builder",
    [
        get_xnnpack_quantizer,
        get_x86_quantizer,
        get_openvino_quantizer,
    ],
    ids=["XNNPACKQuantizer", "X86InductorQuantizer", "OpenVINOQuantizer"],
)
def test_quantizer_setup(
    quantizer_builder: Callable[[tuple[Any, ...]], Quantizer],
    model_case: ModelCase,
    quantizer_params,
    regen_ref_data,
):
    fx_model, _ = _build_torch_fx_model(model_case)
    quantizer = quantizer_builder(**quantizer_params)
    ref_qconfig_filename = (
        FX_QUANTIZED_DIR_NAME / quantizer.__class__.__name__ / get_qconf_filename(model_case.model_id)
    )

    _fuse_conv_bn_(fx_model)
    if isinstance(quantizer, OpenVINOQuantizer) or hasattr(quantizer, "get_nncf_quantization_setup"):
        quantizer = OpenVINOQuantizerAdapter(quantizer)
    else:
        quantizer = TorchAOQuantizerAdapter(quantizer)

    # Call transform_prior_quantization before the NNCFGraph creation
    fx_model = quantizer.transform_prior_quantization(fx_model)
    nncf_graph = GraphConverter.create_nncf_graph(fx_model)
    quantizer_setup = quantizer.get_quantization_setup(fx_model, nncf_graph)
    qsetup_config = quantizer_setup.get_state()
    _normalize_qsetup_state(qsetup_config)
    if regen_ref_data:
        with safe_open(ref_qconfig_filename, "w") as file:
            json.dump(qsetup_config, file, indent=4)

    with safe_open(ref_qconfig_filename, "r") as file:
        ref_qsetup_config = json.load(file)
    # helper to find diff in qconfigs
    # pip install dictdiffer
    # from dictdiffer import diff
    # diff_res = list(diff(ref_qsetup_config, qsetup_config))
    assert qsetup_config == ref_qsetup_config


def _normalize_qsetup_state(setup: dict[str, Any]) -> None:
    """
    Normalizes the quantization setup state dictionary in-place to ensure consistent ordering
    of elements for deterministic behavior.

    :param setup: Quantization setup state to normalize.
    """
    for key in ["unified_scale_groups", "shared_input_operation_set_groups"]:
        sorted_usg = {}
        for k, v in setup[key].items():
            sorted_usg[str(k)] = sorted(v)
        setup[key] = sorted_usg
    dq_key = "directly_quantized_operator_node_names"
    sorted_qps = {}
    for qp in setup["quantization_points"].values():
        sorted_dq = sorted(qp[dq_key])
        qconfig = qp["qconfig"].copy()
        if "dest_dtype" in qconfig:
            qconfig["dest_dtype"] = "INT8" if qconfig["dest_dtype"] is TensorDataType.int8 else "UINT8"
        sorted_qps[f"{tuple(sorted_dq)}_{qp['qip_class']}"] = qconfig
    setup["quantization_points"] = sorted_qps


def _normalize_nncf_graph(nncf_graph: NNCFGraph, fx_graph: torch.fx.Graph):
    """
    Normalizes the given NNCFGraph by renaming quantize/dequantize nodes to ensure consistent naming across runs.
    XNNPACKQuantizer and X86InductorQuantizer quantizers insert quantize and dequantize nodes
    with inconsistent names across runs. This function assigns standardized names to such nodes
    to maintain consistency.

    :param nncf_graph: The given NNCFGraph instance.
    :return: The normalized version of the given NNCFGraph.
    """
    idx = 0
    dtypes_map = {}

    q_dq_types = ["quantize_per_tensor", "dequantize_per_tensor", "quantize_per_channel", "dequantize_per_channel"]
    norm_nncf_graph = NNCFGraph()
    node_names_map = {}
    for node in nncf_graph.topological_sort():
        attrs = node._attributes.copy()
        if node.node_type in q_dq_types:
            new_node_name = f"{node.node_type}_{idx}"
            node_names_map[node.node_name] = new_node_name
            attrs[node.NODE_NAME_ATTR] = new_node_name
            idx += 1
            if node.node_type in ["dequantize_per_tensor", "dequantize_per_channel"]:
                source_node = get_graph_node_by_name(fx_graph, node.node_name)
                dtypes_map[new_node_name] = (
                    TensorDataType.int8 if source_node.args[-1] == torch.int8 else TensorDataType.uint8
                )
        norm_nncf_graph.add_nncf_node(
            node_name=attrs[node.NODE_NAME_ATTR],
            node_type=attrs[node.NODE_TYPE_ATTR],
            node_metatype=attrs[node.METATYPE_ATTR],
            layer_attributes=node.layer_attributes,
        )

    for edge in nncf_graph.get_all_edges():
        from_node_name = node_names_map.get(edge.from_node.node_name, edge.from_node.node_name)
        to_node_name = node_names_map.get(edge.to_node.node_name, edge.to_node.node_name)
        from_node, to_node = [norm_nncf_graph.get_node_by_name(name) for name in (from_node_name, to_node_name)]
        dtype = dtypes_map.get(to_node.node_name, edge.dtype)
        norm_nncf_graph.add_edge_between_nncf_nodes(
            from_node.node_id,
            to_node.node_id,
            tensor_shape=edge.tensor_shape,
            input_port_id=edge.input_port_id,
            output_port_id=edge.output_port_id,
            dtype=dtype,
            parallel_input_port_ids=edge.parallel_input_port_ids,
        )
    return norm_nncf_graph


@pytest.mark.parametrize(
    "model_case,quantizer_params",
    [(m[0], m[1]) for m in TEST_MODELS_QUANIZED],
    ids=[m[0].model_id for m in TEST_MODELS_QUANIZED],
)
def test_openvino_quantizer_with_torch_ao_convert_pt2e(model_case: ModelCase, quantizer_params):
    quantizer = get_openvino_quantizer(**quantizer_params)
    fx_model, example_input = _build_torch_fx_model(model_case)
    prepared_model = prepare_pt2e(fx_model, quantizer)
    prepared_model(example_input)
    ao_quantized_model = convert_pt2e(prepared_model)
    nncf_graph = GraphConverter.create_nncf_graph(ao_quantized_model)

    path_to_dot = (
        FX_QUANTIZED_DIR_NAME / "ao_export_quantization_OpenVINOQuantizer" / get_dot_filename(model_case.model_id)
    )
    nx_graph = nncf_graph.get_graph_for_structure_analysis(extended=True)
    compare_nx_graph_with_reference(nx_graph, path_to_dot.as_posix())


TorchAOSharedQuantizationSpecTestCases = (
    (
        ModelCase(SimpleConcatModel, "unified_scales_test_model", SimpleConcatModel.INPUT_SHAPE),
        ("conv2d", "conv2d_1"),
        (0.01176275312, 127, 0, 255, torch.uint8),
    ),
)


@pytest.mark.parametrize(
    "model_case,unified_scale_node_names,ref_fq_params",
    TorchAOSharedQuantizationSpecTestCases,
    ids=[m[0].model_id for m in TorchAOSharedQuantizationSpecTestCases],
)
def test_OVQuantizer_TorchAOSharedQuantizationSpec_handling(
    model_case: ModelCase,
    unified_scale_node_names: tuple[str, str],
    ref_fq_params: tuple[float, int, int, int, torch.dtype],
):
    model_case.model_builder()(torch.ones(model_case.input_shape))
    fx_model, example_input = _build_torch_fx_model(model_case)

    quantizer = OpenVINOQuantizer()
    prepared_model = prepare_pt2e(fx_model, quantizer)

    actual_annotation = _get_edge_or_node_to_qspec(fx_model)
    for edge_or_node, annotation in actual_annotation.items():
        if isinstance(edge_or_node, torch.fx.Node) and edge_or_node.name == unified_scale_node_names[1]:
            assert isinstance(annotation, TorchAOSharedQuantizationSpec)
            assert annotation.edge_or_node.name == unified_scale_node_names[0]
            assert isinstance(actual_annotation[annotation.edge_or_node], TorchAOQuantizationSpec)
            break
    else:
        msg = f"Node {unified_scale_node_names[1]} should be annotated as quantizable"
        raise RuntimeError(msg)

    prepared_model(example_input)
    ao_quantized_model = convert_pt2e(prepared_model)

    nodes_visited = 0
    for node in ao_quantized_model.graph.nodes:
        if node.name in unified_scale_node_names:
            dequantize_args = list(node.users)[0].args
            assert abs(dequantize_args[1] - ref_fq_params[0]) < torch.finfo(torch.float32).eps
            assert dequantize_args[2:] == ref_fq_params[1:]
            nodes_visited += 1
            if nodes_visited == 2:
                break
    else:
        msg = f"Quantizers was not found for the unified scales pair {unified_scale_node_names}"
        raise RuntimeError(msg)
