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

import os

import openvino.runtime as ov
import pytest

import nncf
from nncf.common.utils.os import is_windows
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from tests.openvino.conftest import AC_CONFIGS_DIR
from tests.openvino.datasets_helpers import get_dataset_for_test
from tests.openvino.datasets_helpers import get_nncf_dataset_from_ac_config
from tests.openvino.omz_helpers import calculate_metrics
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model

OMZ_MODELS = [
    ("resnet-18-pytorch", "imagenette2-320", {"accuracy@top1": 0.777, "accuracy@top5": 0.949}),
    ("mobilenet-v3-small-1.0-224-tf", "imagenette2-320", {"accuracy@top1": 0.744, "accuracy@top5": 0.922}),
    ("googlenet-v3-pytorch", "imagenette2-320", {"accuracy@top1": 0.91, "accuracy@top5": 0.994}),
    ("mobilefacedet-v1-mxnet", "wider", {"map": 0.7750216770678978}),
    ("retinaface-resnet50-pytorch", "wider", {"map": 0.91875950512032}),
]


@pytest.mark.parametrize("model, dataset, ref_metrics", OMZ_MODELS, ids=[model[0] for model in OMZ_MODELS])
def test_compression(data_dir, tmp_path, model, dataset, ref_metrics):
    if is_windows() and model == "mobilefacedet-v1-mxnet":
        pytest.xfail("OMZ for Windows has version 1.2.0 pinned that is incompatible with Python 3.8+")
    if model == "mobilenet-v3-small-1.0-224-tf":
        pytest.xfail("Disables until OMZ fix model conversion pipeline.")
    extracted_data_dir = os.path.dirname(get_dataset_for_test(dataset, data_dir))
    config_path = AC_CONFIGS_DIR / f"{model}.yml"

    download_model(model, tmp_path)
    convert_model(model, tmp_path)
    model_path = tmp_path / "public" / model / "FP32" / f"{model}.xml"

    fp_model_dir = os.path.dirname(model_path)
    int8_ir_path = os.path.join(fp_model_dir, f"{model}_int8.xml")

    calibration_dataset = get_nncf_dataset_from_ac_config(model_path, config_path, extracted_data_dir)

    ov_model = ov.Core().read_model(str(model_path))
    quantized_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        advanced_parameters=AdvancedQuantizationParameters(backend_params={"use_pot": True}),
    )
    ov.serialize(quantized_model, int8_ir_path)

    report_path = tmp_path / f"{model}.csv"
    metrics = calculate_metrics(int8_ir_path, config_path, extracted_data_dir, report_path, eval_size=1000)
    for metric_name, metric_val in ref_metrics.items():
        assert metrics[metric_name] == pytest.approx(metric_val, abs=0.006)
