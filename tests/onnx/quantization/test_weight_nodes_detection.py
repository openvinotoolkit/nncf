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
from collections import Counter

from nncf.experimental.post_training.algorithms.quantization.min_max.algorithm import MinMaxQuantization, \
    MinMaxQuantizationParameters
from nncf.experimental.onnx.graph.nncf_graph_builder import GraphConverter
from tests.onnx.models import WEIGHT_DETECTION_MODELS
from tests.onnx.models import MulWithWeightsFirstInputModel, MulWithWeightsZeroInputModel, ReshapeWeightModel

# pylint: disable=protected-access


@pytest.mark.parametrize('model_to_test', WEIGHT_DETECTION_MODELS.values())
def test_weight_nodes_detection(model_to_test):
    if model_to_test in [MulWithWeightsFirstInputModel, MulWithWeightsZeroInputModel, ReshapeWeightModel]:
        pytest.skip('Currently there are some limitations of quantizing the weights of the elementwise operations.'
                    'The ticket is 94688.')
    model_to_test = model_to_test()
    onnx_model = model_to_test.onnx_model
    quantization_algo = MinMaxQuantization(MinMaxQuantizationParameters(number_samples=1))
    nncf_graph = GraphConverter.create_nncf_graph(onnx_model)
    quantization_algo._set_backend_entity(onnx_model)
    quantizer_setup = quantization_algo._get_quantizer_setup(onnx_model, nncf_graph)

    quantized_weight_nodes = []
    for quantization_point in quantizer_setup.quantization_points.values():
        if quantization_point.is_weight_quantization_point():
            quantized_weight_nodes.append(quantization_point.insertion_point.target_node_name)
    assert Counter(quantized_weight_nodes) == Counter(model_to_test.weight_nodes)
