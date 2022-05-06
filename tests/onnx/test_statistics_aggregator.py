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

import numpy as np

from nncf.experimental.post_training.compression_builder import CompressionBuilder
from nncf.experimental.onnx.algorithms.quantization.min_max_quantization import ONNXMinMaxQuantization
from nncf.experimental.post_training.algorithms.quantization import MinMaxQuantizationParameters
from nncf.experimental.onnx.graph.onnx_graph import ONNXGraph
from nncf.experimental.post_training.algorithms.quantization.min_max_quantization import RangeType

from tests.onnx.models import OneConvolutionalModel
from tests.onnx.test_samplers import TestDataset

INPUT_SHAPE = [3, 10, 10]

DATASET_SAMPLES = [(np.zeros(INPUT_SHAPE), 0),
                   (np.zeros(INPUT_SHAPE), 1),
                   (np.zeros(INPUT_SHAPE), 2)]

DATASET_SAMPLES[0][0][0, 0, 0] = 128  # max
DATASET_SAMPLES[0][0][0, 0, 1] = -128  # min

DATASET_SAMPLES[1][0][0, 0, 0] = 1  # max
DATASET_SAMPLES[1][0][0, 0, 1] = -10  # min

DATASET_SAMPLES[2][0][0, 0, 0] = 0.1  # max
DATASET_SAMPLES[2][0][0, 0, 1] = -1  # min


class TestParameters:
    def __init__(self, number_samples, activation_float_range, weight_float_range):
        self.number_samples = number_samples
        self.activation_float_range = activation_float_range
        self.weight_float_range = weight_float_range


@pytest.mark.parametrize('range_type, test_parameters',
                         ((RangeType.MEAN_MINMAX, (TestParameters(1, 128, 1))),
                          (RangeType.MEAN_MINMAX, (TestParameters(2, 69, 1))),
                          (RangeType.MEAN_MINMAX, (TestParameters(3, 46.333, 1))),
                          (RangeType.MINMAX, (TestParameters(1, 128, 1))),
                          (RangeType.MINMAX, (TestParameters(2, 128, 1))),
                          (RangeType.MINMAX, (TestParameters(3, 128, 1)))
                          )
                         )
def test_statistics_aggregator(range_type, test_parameters):
    model = OneConvolutionalModel().onnx_model

    dataset = TestDataset(DATASET_SAMPLES)
    compression_builder = CompressionBuilder()

    quantization = ONNXMinMaxQuantization(MinMaxQuantizationParameters(
        number_samples=test_parameters.number_samples,
        range_type=range_type
    ))

    compression_builder.add_algorithm(quantization)
    quantized_model = compression_builder.apply(model, dataset)

    onnx_graph = ONNXGraph(quantized_model)
    num_q = 0
    for node in quantized_model.graph.node:
        if node.name == 'QuantizeLinear_X':
            num_q += 1
            activation_scale = test_parameters.activation_float_range / ((2 ** 8 - 1) / 2)
            assert np.allclose(onnx_graph.get_initializers_value(node.input[1]), np.array(activation_scale))
        if node.name == 'QuantizeLinear_Conv1_W':
            num_q += 1
            weight_scale = test_parameters.weight_float_range / ((2 ** 8 - 1) / 2)
            assert np.allclose(onnx_graph.get_initializers_value(node.input[1]),
                               np.array(weight_scale))
    assert num_q == 2
