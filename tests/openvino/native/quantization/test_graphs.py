# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict

import numpy as np
import openvino.runtime as ov
import pytest

from nncf import Dataset
from nncf.common.quantization.structs import QuantizationPreset
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.quantization.quantize_model import quantize_impl
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant
from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.native.common import dump_model
from tests.openvino.native.common import get_dataset_for_test
from tests.openvino.native.common import get_openvino_version
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.openvino.native.models import DepthwiseConv3DModel
from tests.openvino.native.models import DepthwiseConv4DModel
from tests.openvino.native.models import DepthwiseConv5DModel
from tests.openvino.native.models import GRUSequenceModel
from tests.openvino.native.models import IfModel
from tests.openvino.native.models import MatmulSoftmaxMatmulBlock
from tests.openvino.native.quantization.test_fq_params_calculation import quantize_model
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model

OV_VERSION = get_openvino_version()
QUANTIZED_REF_GRAPHS_DIR = OPENVINO_NATIVE_TEST_ROOT / "data" / OV_VERSION / "reference_graphs" / "quantized"


@pytest.mark.parametrize("model_creator_func", SYNTHETIC_MODELS.values())
def test_synthetic_models_fq_placement(model_creator_func):
    model = model_creator_func()
    quantized_model = quantize_model(
        model.ov_model, {"preset": QuantizationPreset.PERFORMANCE, "inplace_statistics": True}
    )

    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / model.ref_graph_name
    compare_nncf_graphs(quantized_model, path_ref_graph)


@pytest.mark.parametrize("model_creator_func", [DepthwiseConv3DModel, DepthwiseConv4DModel, DepthwiseConv5DModel])
def test_depthwise_models_fq_placement(model_creator_func):
    model = model_creator_func()
    quantized_model = quantize_model(
        model.ov_model, {"preset": QuantizationPreset.PERFORMANCE, "inplace_statistics": True}
    )

    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / model.ref_graph_name
    compare_nncf_graphs(quantized_model, path_ref_graph)


OMZ_MODELS_QUANTIZE_PARAMS = {
    "swin-tiny-patch4-window7-224": {"preset": QuantizationPreset.PERFORMANCE, "model_type": ModelType.TRANSFORMER},
    "mobilenet-v2-pytorch": {"preset": QuantizationPreset.PERFORMANCE},
    "mobilenet-v3-small-1.0-224-tf": {"preset": QuantizationPreset.PERFORMANCE},
    "resnet-18-pytorch": {"preset": QuantizationPreset.PERFORMANCE},
    "resnet-50-pytorch": {"preset": QuantizationPreset.PERFORMANCE, "target_device": TargetDevice.CPU_SPR},
    "yolo-v4-tiny-tf": {"preset": QuantizationPreset.PERFORMANCE},
}


@pytest.mark.parametrize("model_name_params", OMZ_MODELS_QUANTIZE_PARAMS.items(), ids=list(OMZ_MODELS_QUANTIZE_PARAMS))
def test_omz_models_fq_placement(model_name_params, tmp_path, omz_cache_dir):
    model_name, q_params = model_name_params
    params_str = "_".join([param.value for param in q_params.values()])
    q_params.update({"inplace_statistics": True})
    download_model(model_name, tmp_path, omz_cache_dir)
    convert_model(model_name, tmp_path)
    model_path = tmp_path / "public" / model_name / "FP32" / f"{model_name}.xml"
    model = ov.Core().read_model(model_path)
    quantized_model = quantize_model(model, q_params)

    result_name = f"{model_name}_{params_str}"
    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / f"{result_name}.dot"
    xml_path = tmp_path / (result_name + ".xml")
    bin_path = tmp_path / (result_name + ".bin")
    dump_model(quantized_model, str(xml_path), str(bin_path))
    compare_nncf_graphs(quantized_model, path_ref_graph)


@pytest.mark.parametrize("model_creator_func", [MatmulSoftmaxMatmulBlock])
def test_transformer_models_fq_placement(model_creator_func, tmp_path):
    model = model_creator_func()
    quantized_model = quantize_model(
        model.ov_model,
        {"preset": QuantizationPreset.PERFORMANCE, "inplace_statistics": True, "model_type": ModelType.TRANSFORMER},
    )

    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / model.ref_graph_name
    xml_path = tmp_path / (model.ref_model_name + ".xml")
    bin_path = tmp_path / (model.ref_model_name + ".bin")
    dump_model(quantized_model, str(xml_path), str(bin_path))
    compare_nncf_graphs(quantized_model, path_ref_graph)


OMZ_MODELS_SQ_PARAMS = {
    "swin-tiny-patch4-window7-224": {"preset": QuantizationPreset.PERFORMANCE, "model_type": ModelType.TRANSFORMER}
}


@pytest.mark.parametrize("model_name_params", OMZ_MODELS_SQ_PARAMS.items(), ids=list(OMZ_MODELS_SQ_PARAMS))
def test_omz_models_sq_placement(model_name_params, tmp_path, omz_cache_dir):
    model_name, q_params = model_name_params
    q_params.update({"inplace_statistics": True})
    download_model(model_name, tmp_path, omz_cache_dir)
    convert_model(model_name, tmp_path)
    model_path = tmp_path / "public" / model_name / "FP32" / f"{model_name}.xml"
    model = ov.Core().read_model(model_path)

    quantized_model = smooth_quant_model(model, q_params, quantize=False)

    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / f"{model_name}_sq.dot"
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
    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / f"GRUSequenceModel_linear_before_reset_{postfix}.dot"
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

    compare_nncf_graphs(quantized_model, QUANTIZED_REF_GRAPHS_DIR / main_model_path)
    compare_nncf_graphs(if_op.get_function(0), QUANTIZED_REF_GRAPHS_DIR / then_body_path)
    compare_nncf_graphs(if_op.get_function(1), QUANTIZED_REF_GRAPHS_DIR / else_body_path)
