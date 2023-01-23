"""
 Copyright (c) 2023 Intel Corporation
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
import openvino.runtime as ov

from nncf.common.quantization.structs import QuantizationPreset

from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT
from tests.openvino.omz_helpers import convert_model
from tests.openvino.omz_helpers import download_model
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.native.models import SYNTHETIC_MODELS
from tests.openvino.native.models import WeightsModel
from tests.openvino.native.quantization.test_fq_params_calculation import quantize_model

QUANTIZED_REF_GRAPHS_DIR = OPENVINO_NATIVE_TEST_ROOT / 'data' / 'reference_graphs' / 'quantized'


@pytest.mark.parametrize('model_creator_func', SYNTHETIC_MODELS.values())
def test_syntetic_models_fq_placement(model_creator_func):
    if model_creator_func == WeightsModel:
        pytest.skip('OpenVINO backend does not support MatMul op without weights.')
    model = model_creator_func()
    quantized_model = quantize_model(model.ov_model, QuantizationPreset.PERFORMANCE)

    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / model.ref_graph_name
    compare_nncf_graphs(quantized_model, path_ref_graph)


OMZ_MODELS = [
    'mobilenet-v2-pytorch',
    'mobilenet-v3-small-1.0-224-tf',
    'resnet-18-pytorch',
    'yolo-v4-tiny-tf',
]


@pytest.mark.skip(reason='Ticket 100948')
@pytest.mark.parametrize('model_name', OMZ_MODELS)
def test_omz_models_fq_placement(model_name, tmp_path):
    _ = download_model(model_name, tmp_path)
    model_path = convert_model(model_name, tmp_path)
    model = ov.Core().read_model(model_path)
    quantized_model = quantize_model(model, QuantizationPreset.PERFORMANCE)

    path_ref_graph = QUANTIZED_REF_GRAPHS_DIR / f'{model_name}.dot'
    compare_nncf_graphs(quantized_model, path_ref_graph)
