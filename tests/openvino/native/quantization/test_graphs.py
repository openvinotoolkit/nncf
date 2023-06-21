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

import openvino.runtime as ov
import pytest

from nncf.common.quantization.structs import QuantizationPreset
from nncf.openvino.statistics.aggregator import OVStatisticsAggregator
from nncf.parameters import ModelType
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant
from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.native.common import dump_model
from tests.openvino.native.common import get_dataset_for_test
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.openvino.native.models import DepthwiseConv3DModel
from tests.openvino.native.models import DepthwiseConv4DModel
from tests.openvino.native.models import DepthwiseConv5DModel
from tests.openvino.native.models import MatmulSoftmaxMatmulBlock
from tests.openvino.native.quantization.test_fq_params_calculation import quantize_model
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model

QUANTIZED_REF_GRAPHS_DIR = OPENVINO_NATIVE_TEST_ROOT / "data" / "reference_graphs" / "quantized"


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
    "yolo-v4-tiny-tf": {"preset": QuantizationPreset.PERFORMANCE},
}


@pytest.mark.parametrize("model_name_params", OMZ_MODELS_QUANTIZE_PARAMS.items(), ids=list(OMZ_MODELS_QUANTIZE_PARAMS))
def test_omz_models_fq_placement(model_name_params, tmp_path):
    model_name, q_params = model_name_params
    q_params.update({"inplace_statistics": True})
    download_model(model_name, tmp_path)
    convert_model(model_name, tmp_path)
    model_path = tmp_path / "public" / model_name / "FP32" / f"{model_name}.xml"
    model = ov.Core().read_model(model_path)
    quantized_model = quantize_model(model, q_params)

    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / f"{model_name}.dot"
    xml_path = tmp_path / (model_name + ".xml")
    bin_path = tmp_path / (model_name + ".bin")
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
def test_omz_models_sq_placement(model_name_params, tmp_path):
    model_name, q_params = model_name_params
    q_params.update({"inplace_statistics": True})
    download_model(model_name, tmp_path)
    convert_model(model_name, tmp_path)
    model_path = tmp_path / "public" / model_name / "FP32" / f"{model_name}.xml"
    model = ov.Core().read_model(model_path)

    quantized_model = smooth_quant_model(model, q_params, quantize=False)

    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / f"{model_name}_sq.dot"
    xml_path = tmp_path / (model_name + ".xml")
    bin_path = tmp_path / (model_name + ".bin")
    dump_model(quantized_model, str(xml_path), str(bin_path))
    compare_nncf_graphs(quantized_model, path_ref_graph)


# pylint: disable=protected-access
def smooth_quant_model(ov_model: ov.Model, q_params: Dict, quantize=True):
    dataset = get_dataset_for_test(ov_model)

    smooth_quant_algo = SmoothQuant(subset_size=1)
    statistics_aggregator = OVStatisticsAggregator(dataset)
    statistic_points = smooth_quant_algo.get_statistic_points(ov_model)
    statistics_aggregator.register_statistic_points(statistic_points)
    statistics_aggregator.collect_statistics(ov_model)
    modified_model = smooth_quant_algo._apply(ov_model, statistics_aggregator.statistic_points)

    if quantize:
        modified_model = quantize_model(modified_model, q_params)
    return modified_model
