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


from pathlib import Path
from typing import Dict

import numpy as np
import openvino as ov
import pytest

from nncf import Dataset
from nncf.common.quantization.structs import QuantizationPreset
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.quantization.quantize_model import quantize_impl
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant
from nncf.scopes import IgnoredScope
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.native.common import convert_torch_model
from tests.openvino.native.common import dump_model
from tests.openvino.native.common import get_actual_reference_for_current_openvino
from tests.openvino.native.common import get_dataset_for_test
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.openvino.native.models import DepthwiseConv3DModel
from tests.openvino.native.models import DepthwiseConv4DModel
from tests.openvino.native.models import DepthwiseConv5DModel
from tests.openvino.native.models import GRUSequenceModel
from tests.openvino.native.models import IfModel
from tests.openvino.native.models import IfModel_2
from tests.openvino.native.models import MatmulSoftmaxMatmulBlock
from tests.openvino.native.models import RoPEModel
from tests.openvino.native.models import ScaledDotProductAttentionModel
from tests.openvino.native.models import get_torch_model_info
from tests.openvino.native.quantization.test_fq_params_calculation import quantize_model

QUANTIZED_REF_GRAPHS_DIR = Path("reference_graphs") / "quantized"


@pytest.mark.parametrize("model_creator_func", SYNTHETIC_MODELS.values())
def test_synthetic_models_fq_placement(model_creator_func):
    model = model_creator_func()
    quantized_model = quantize_model(
        model.ov_model, {"preset": QuantizationPreset.PERFORMANCE, "inplace_statistics": True}
    )

    path_ref_graph = get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / model.ref_graph_name)
    compare_nncf_graphs(quantized_model, path_ref_graph)


@pytest.mark.parametrize("model_creator_func", [DepthwiseConv3DModel, DepthwiseConv4DModel, DepthwiseConv5DModel])
def test_depthwise_models_fq_placement(model_creator_func):
    model = model_creator_func()
    quantized_model = quantize_model(
        model.ov_model, {"preset": QuantizationPreset.PERFORMANCE, "inplace_statistics": True}
    )

    path_ref_graph = get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / model.ref_graph_name)
    compare_nncf_graphs(quantized_model, path_ref_graph)


MODELS_QUANTIZE_PARAMS = (
    ("mobilenet-v2", {"preset": QuantizationPreset.PERFORMANCE}),
    ("mobilenet-v3-small", {"preset": QuantizationPreset.PERFORMANCE}),
    ("resnet-18", {"preset": QuantizationPreset.PERFORMANCE}),
    ("resnet-18", {"preset": QuantizationPreset.PERFORMANCE, "target_device": TargetDevice.CPU_SPR}),
    ("ssd-vgg-300", {"preset": QuantizationPreset.PERFORMANCE}),
    ("swin-block", {"preset": QuantizationPreset.PERFORMANCE, "model_type": ModelType.TRANSFORMER}),
)


@pytest.mark.parametrize("model_name_params", MODELS_QUANTIZE_PARAMS)
def test_real_models_fq_placement(model_name_params, tmp_path):
    model_name, q_params = model_name_params
    params_str = "_".join([param.value for param in q_params.values()])
    model_cls, input_shape = get_torch_model_info(model_name)
    ov_model = convert_torch_model(model_cls(), input_shape, tmp_path)

    quantized_model = quantize_model(ov_model, q_params)

    result_name = f"{model_name}_{params_str}"
    path_ref_graph = get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / f"{result_name}.dot")
    xml_path = tmp_path / (result_name + ".xml")
    bin_path = tmp_path / (result_name + ".bin")
    dump_model(quantized_model, str(xml_path), str(bin_path))
    compare_nncf_graphs(quantized_model, path_ref_graph)


@pytest.mark.parametrize("model_creator_func", [MatmulSoftmaxMatmulBlock, RoPEModel])
def test_transformer_models_fq_placement(model_creator_func, tmp_path):
    model = model_creator_func()
    quantized_model = quantize_model(
        model.ov_model,
        {"preset": QuantizationPreset.PERFORMANCE, "inplace_statistics": True, "model_type": ModelType.TRANSFORMER},
    )

    path_ref_graph = get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / model.ref_graph_name)
    xml_path = tmp_path / (model.ref_model_name + ".xml")
    bin_path = tmp_path / (model.ref_model_name + ".bin")
    dump_model(quantized_model, str(xml_path), str(bin_path))
    compare_nncf_graphs(quantized_model, path_ref_graph)


MODELS_SQ_PARAMS = (("swin-block", {"preset": QuantizationPreset.PERFORMANCE, "model_type": ModelType.TRANSFORMER}),)


@pytest.mark.parametrize("model_name_params", MODELS_SQ_PARAMS)
def test_real_models_sq_placement(model_name_params, tmp_path):
    model_name, q_params = model_name_params

    model_cls, input_shape = get_torch_model_info(model_name)
    ov_model = convert_torch_model(model_cls(), input_shape, tmp_path)

    quantized_model = smooth_quant_model(ov_model, q_params, quantize=False)

    path_ref_graph = get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / f"{model_name}_sq.dot")
    xml_path = tmp_path / (model_name + ".xml")
    bin_path = tmp_path / (model_name + ".bin")
    dump_model(quantized_model, str(xml_path), str(bin_path))
    compare_nncf_graphs(quantized_model, path_ref_graph)


def smooth_quant_model(ov_model: ov.Model, q_params: Dict, quantize=True):
    dataset = get_dataset_for_test(ov_model)
    graph = GraphConverter.create_nncf_graph(ov_model)

    smooth_quant_algo = SmoothQuant(subset_size=1)
    statistics_aggregator = OVStatisticsAggregator(dataset)
    statistic_points = smooth_quant_algo.get_statistic_points(ov_model, graph)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(ov_model, graph)
    modified_model = smooth_quant_algo.apply(ov_model, graph, statistics_aggregator.statistic_points)

    if quantize:
        modified_model = quantize_model(modified_model, q_params)
    return modified_model


@pytest.mark.parametrize(
    "linear_before_reset", [True, False], ids=["linear_before_reset_True", "linear_before_reset_False"]
)
def test_ignore_nodes_by_attribues(linear_before_reset):
    model = GRUSequenceModel(**{"linear_before_reset": linear_before_reset}).ov_model
    quantized_model = quantize_model(model, {})
    postfix = "T" if linear_before_reset else "F"
    path_ref_graph = get_actual_reference_for_current_openvino(
        QUANTIZED_REF_GRAPHS_DIR / f"GRUSequenceModel_linear_before_reset_{postfix}.dot"
    )
    compare_nncf_graphs(quantized_model, path_ref_graph)


def get_dataset_for_if_model(model: ov.Model, size: int = 2) -> Dataset:
    rng = np.random.default_rng(seed=0)
    dataitems = []
    for i in range(size):
        input_data = {}
        for param in model.get_parameters():
            if param.get_element_type().get_type_name() == "boolean":
                input_data[param.get_output_tensor(0).get_any_name()] = i < size // 2
            else:
                input_shape = param.partial_shape.get_max_shape()
                input_data[param.get_output_tensor(0).get_any_name()] = rng.uniform(0, 1, input_shape)
        dataitems.append(input_data)
    dataset = Dataset(dataitems)
    return dataset


def test_if_model_fq_placement():
    if_model = IfModel()
    ov_model = if_model.ov_model
    dataset = get_dataset_for_if_model(ov_model)
    quantized_model = quantize_impl(
        ov_model,
        dataset,
        subset_size=2,
        fast_bias_correction=True,
    )
    if_ops = [op for op in quantized_model.get_ops() if op.get_type_name() == "If"]
    assert len(if_ops) == 1
    if_op = if_ops[0]
    main_model_path = if_model.ref_model_name + "_main.dot"
    then_body_path = if_model.ref_model_name + "_then.dot"
    else_body_path = if_model.ref_model_name + "_else.dot"

    compare_nncf_graphs(
        quantized_model, get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / main_model_path)
    )
    compare_nncf_graphs(
        if_op.get_function(0), get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / then_body_path)
    )
    compare_nncf_graphs(
        if_op.get_function(1), get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / else_body_path)
    )


def test_if_model_fq_placement_ignored_scope():
    if_model = IfModel_2()
    ov_model = if_model.ov_model
    dataset = get_dataset_for_if_model(ov_model)
    quantized_model = quantize_impl(
        ov_model, dataset, subset_size=2, fast_bias_correction=True, ignored_scope=IgnoredScope(names=["MatMul"])
    )
    if_ops = [op for op in quantized_model.get_ops() if op.get_type_name() == "If"]
    assert len(if_ops) == 1
    if_op = if_ops[0]
    main_model_path = if_model.ref_model_name + "_ignored_scope_main.dot"
    then_body_path = if_model.ref_model_name + "_ignored_scope_then.dot"
    else_body_path = if_model.ref_model_name + "_ignored_scope_else.dot"

    compare_nncf_graphs(
        quantized_model, get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / main_model_path)
    )
    compare_nncf_graphs(
        if_op.get_function(0), get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / then_body_path)
    )
    compare_nncf_graphs(
        if_op.get_function(1), get_actual_reference_for_current_openvino(QUANTIZED_REF_GRAPHS_DIR / else_body_path)
    )


@pytest.mark.parametrize("q_params", [{}, {"model_type": ModelType.TRANSFORMER}], ids=["default", "transformer"])
def test_scaled_dot_product_attention_placement(q_params, tmp_path):
    model = ScaledDotProductAttentionModel().ov_model
    quantized_model = quantize_model(model, q_params)

    if q_params:
        params_str = "_".join([param.value for param in q_params.values()])
    else:
        params_str = "default"

    path_ref_graph = get_actual_reference_for_current_openvino(
        QUANTIZED_REF_GRAPHS_DIR / "scaled_dot_product_attention.dot"
    )
    result_name = f"scaled_dot_product_attention_{params_str}"
    xml_path = tmp_path / (result_name + ".xml")
    bin_path = tmp_path / (result_name + ".bin")
    dump_model(quantized_model, str(xml_path), str(bin_path))
    compare_nncf_graphs(quantized_model, path_ref_graph)


@pytest.mark.parametrize(
    "model_creator_func",
    [SYNTHETIC_MODELS.get("LinearModel"), SYNTHETIC_MODELS.get("ConvModel"), SYNTHETIC_MODELS.get("SharedConvModel")],
)
def test_synthetic_models_fc_placement(model_creator_func):
    model = model_creator_func()
    quantized_model = quantize_model(
        model.ov_model,
        {
            "preset": QuantizationPreset.PERFORMANCE,
            "inplace_statistics": True,
            "mode": QuantizationMode.FP8_E4M3,
        },
    )

    path_ref_graph = get_actual_reference_for_current_openvino(
        QUANTIZED_REF_GRAPHS_DIR / f"{model.ref_model_name}_FC.dot"
    )
    compare_nncf_graphs(quantized_model, path_ref_graph)
