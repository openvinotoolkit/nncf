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
from tests.openvino.omz_helpers import OPENVINO_DATASET_DEFINITIONS_PATH
from tests.openvino.omz_helpers import calculate_metrics
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model
from tests.openvino.datasets_helpers import get_dataset_for_test

OMZ_MODELS = [
    ('resnet-18-pytorch', 'imagenette2-320', {'accuracy@top1': '0.778', 'accuracy@top5': '0.948'}),
    ('mobilenet-v3-small-1.0-224-tf', 'imagenette2-320', {'accuracy@top1': '0.746', 'accuracy@top5': '0.92'}),
    ('googlenet-v3-pytorch', 'imagenette2-320', {'accuracy@top1': '0.909', 'accuracy@top5': '0.994'}),
    ('mobilefacedet-v1-mxnet', 'wider', {'map': '0.7750224587055485'}),
    ('retinaface-resnet50-pytorch', 'wider', {'map': '0.9170155131056823'}),
]


# pylint: disable=protected-access
def quantize_model(model_path, config_path, data_dir, framework='openvino', device='CPU'):
    args = [
        "-c", str(config_path),
        "-m", str(model_path),
        "-d", str(OPENVINO_DATASET_DEFINITIONS_PATH),
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
        return filled_inputs[0]

    calibration_dataset = nncf.Dataset(model_evaluator.dataset, transform_fn)
    model = ov.Core().read_model(str(model_path))
    quantized_model = nncf.quantize(model, calibration_dataset)
    return quantized_model


@pytest.mark.parametrize('model, dataset, ref_metrics',
                         OMZ_MODELS, ids=[model[0] for model in OMZ_MODELS])
def test_compression(tmp_path, model, dataset, ref_metrics):
    data_dir = os.path.dirname(get_dataset_for_test(dataset))
    config_path = AC_CONFIGS_DIR / f'{model}.yml'

    _ = download_model(model, tmp_path)
    model_path = convert_model(model, tmp_path)

    fp_model_dir = os.path.dirname(model_path)
    int8_ir_path = os.path.join(fp_model_dir, f'{model}_int8.xml')

    quantized_model = quantize_model(model_path, config_path, data_dir)
    ov.serialize(quantized_model, int8_ir_path)

    report_path = tmp_path / f'{model}.csv'
    metrics = calculate_metrics(int8_ir_path, config_path, data_dir, report_path, eval_size=1000)
    for metric_name, metric_val in ref_metrics.items():
        assert metrics[metric_name] == pytest.approx(metric_val, abs=0.006)
