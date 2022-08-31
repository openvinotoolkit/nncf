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

# pylint: disable=no-member, redefined-outer-name, no-name-in-module

import pytest

from tests.onnx.quantization.common import infer_model, min_max_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.models import ReshapeWeightModel
from tests.onnx.models import WeightSharingModel
from tests.onnx.models import OneInputPortQuantizableModel
from tests.onnx.models import ManyInputPortsQuantizableModel


@pytest.mark.parametrize('model_to_test', [ReshapeWeightModel(), WeightSharingModel(), OneInputPortQuantizableModel(),
                                           ManyInputPortsQuantizableModel()])
def test_syntetic_models_graph(model_to_test):
    quantized_model = min_max_quantize_model(model_to_test.input_shape[0], model_to_test.onnx_model)
    infer_model(model_to_test.input_shape[0], quantized_model)
    compare_nncf_graph(quantized_model, model_to_test.path_ref_graph)

