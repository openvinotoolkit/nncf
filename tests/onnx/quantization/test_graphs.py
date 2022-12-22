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
from unittest.mock import patch

from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.aggregator import StatisticsAggregator
from tests.onnx.quantization.common import infer_model, min_max_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import mock_calculate_activation_quantizer_parameters
from tests.onnx.quantization.common import mock_get_statistics
from tests.onnx.quantization.common import mock_collect_statistics
from tests.onnx.models import ALL_SYNTHETIC_MODELS
from tests.onnx.models import MultiInputOutputModel

@patch('nncf.quantization.algorithms.min_max.onnx_backend.calculate_activation_quantizer_parameters',
       new=mock_calculate_activation_quantizer_parameters)
@pytest.mark.parametrize('model_cls_to_test', ALL_SYNTHETIC_MODELS.values())
def test_syntetic_models_graph(model_cls_to_test):
    mockObject = StatisticsAggregator
    mockObject.collect_statistics = mock_collect_statistics
    mockObject_ = TensorStatisticCollectorBase
    mockObject_.get_statistics = mock_get_statistics
    if model_cls_to_test == MultiInputOutputModel:
        pytest.skip('min_max_quantize_model does not support many inputs for now.')
    model_to_test = model_cls_to_test()
    quantized_model = min_max_quantize_model(model_to_test.input_shape[0], model_to_test.onnx_model)
    compare_nncf_graph(quantized_model, 'synthetic/' + model_to_test.path_ref_graph)
    infer_model(model_to_test.input_shape[0], quantized_model)
