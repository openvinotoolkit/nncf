# Copyright (c) 2025 Intel Corporation
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

import pytest

from tests.torch import test_models
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import register_bn_adaptation_init_args
from tests.torch.test_compressed_graph import QUANTIZERS
from tests.torch.test_compressed_graph import QuantizeTestCaseConfiguration
from tests.torch.test_compressed_graph import check_model_graph
from tests.torch.test_compressed_graph import get_basic_quantization_config

TEST_MODELS = [
    (("alexnet.dot", "lenet.dot"), (test_models.AlexNet, test_models.LeNet), ([1, 3, 32, 32], [1, 3, 32, 32]))
]


@pytest.fixture(scope="function", params=QUANTIZERS)
def _case_config(request):
    quantization_type = request.param
    graph_dir = os.path.join("quantized", quantization_type)
    return QuantizeTestCaseConfiguration(quantization_type, graph_dir)


@pytest.mark.parametrize("model_name, model_builder, input_size", TEST_MODELS)
def test_context_independence(model_name, model_builder, input_size, _case_config):
    config = get_basic_quantization_config(_case_config.quant_type, input_sample_sizes=input_size[0])
    register_bn_adaptation_init_args(config)

    compressed_models = [
        create_compressed_model_and_algo_for_test(model_builder[0](), config)[0],
        create_compressed_model_and_algo_for_test(model_builder[1](), config)[0],
    ]

    for i, compressed_model in enumerate(compressed_models):
        check_model_graph(compressed_model, model_name[i], _case_config.graph_dir)
