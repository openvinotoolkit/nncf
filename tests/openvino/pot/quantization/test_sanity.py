"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import pytest
import os
import nncf
import openvino.runtime as ov
from openvino.tools.accuracy_checker.config import ConfigReader
from openvino.tools.accuracy_checker.argparser import build_arguments_parser
from openvino.tools.accuracy_checker.evaluators import ModelEvaluator

from tests.openvino.conftest import AC_CONFIGS_DIR
from tests.openvino.omz_helpers import DATASET_DEFINITIONS_PATH
from tests.openvino.omz_helpers import calculate_metrics
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model

OMZ_MODELS = {
    'mobilenet-v2-pytorch': {'accuracy@top1': '0.7153', 'accuracy@top5': '0.9016'},
    'mobilenet-v3-small-1.0-224-tf': {'accuracy@top1': '0.64536', 'accuracy@top5': '0.85656'},
    'resnet-18-pytorch': {'accuracy@top1': '0.69532', 'accuracy@top5': '0.89022'},
    'googlenet-v3-pytorch': {'accuracy@top1': '0.77534', 'accuracy@top5': '0.9357'},
    'ssd_mobilenet_v1_coco': {'coco_precision': '0.23112447531324606'},
    'yolo-v4-tiny-tf': {'map': '0.3981337453540408', 'AP@0.5': '0.4598277813211308', 'AP@0.5:0.05:95': '0.22052182158380668'}
}


def quantize_model(model_path, config_path, data_dir, framework='openvino', device='CPU'):
    args = [
        "-c", str(config_path),
        "-m", str(model_path),
        "-d", str(DATASET_DEFINITIONS_PATH),
        "-s", str(data_dir),
        "-tf", framework,
        "-td", device,
    ]
    parser = build_arguments_parser()
    args = parser.parse_args(args)

    config, mode = ConfigReader.merge(args)
    model_evaluator = ModelEvaluator.from_configs(config[mode][0])

    def transform_fn(data_item):
        _, batch_annotation, batch_input, _ = data_item
        filled_inputs, _, _ = model_evaluator._get_batch_input(batch_annotation, batch_input)
        return filled_inputs

    calibration_dataset = nncf.Dataset(model_evaluator.dataset, transform_fn)
    model = ov.Core().read_model(str(model_path))
    quantized_model = nncf.quantize(model, calibration_dataset)
    return quantized_model


@pytest.mark.parametrize('model_description', OMZ_MODELS.items())
def test_compression(tmp_path, model_description, data_dir):
    model_name, ref_metrics = model_description
    _ = download_model(model_name, tmp_path)
    model_path = convert_model(model_name, tmp_path)

    config_path = AC_CONFIGS_DIR / f'{model_name}.yml'
    quantized_model = quantize_model(model_path, config_path, data_dir)

    fp_model_dir = os.path.basename(os.path.dirname(model_path))
    int8_ir_path = f'{fp_model_dir}/{model_name}_int8.xml'
    ov.serialize(quantized_model, int8_ir_path)

    report_path = tmp_path / f'{model_name}.csv'
    metrics = calculate_metrics(int8_ir_path, config_path, data_dir, report_path)

    for metric_name, metric_val in ref_metrics.items():
        print(f'{metric_name}: {metric_val}')
        assert metrics[metric_name] == pytest.approx(ref_metrics[metric_name], abs=0.006)
