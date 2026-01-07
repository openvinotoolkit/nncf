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

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as opset

import nncf
from nncf import CompressWeightsMode


def get_transpose_b_false_model():
    """Creates model with [In, Out] weight layout (transpose_b=False)"""
    input_shape = [1, 32]
    input_node = opset.parameter(input_shape, name="Input")
    # Weight shape [32, 16] -> Input=32, Output=16
    weight_data = np.random.rand(32, 16).astype(np.float32)
    matmul_node = opset.matmul(input_node, weight_data, transpose_a=False, transpose_b=False, name="MatMul")
    result_node = opset.result(matmul_node, name="Result")
    return ov.Model([result_node], [input_node], "transpose_b_false_model")


@pytest.mark.parametrize(
    "params", [{"awq": True}, {"gptq": True}, {"scale_estimation": True}, {"lora_correction": True}]
)
def test_compress_weights_algorithms_transpose_b_false(params):
    """
    Checks that ALL data-aware algorithms support transpose_b=False
    without crashing.
    """
    model = get_transpose_b_false_model()

    # Dummy dataset for calibration
    dataset = nncf.Dataset([np.random.rand(1, 32).astype(np.float32) for _ in range(3)])

    # We use INT4_ASYM as it supports all these advanced algorithms
    try:
        nncf.compress_weights(
            model,
            mode=CompressWeightsMode.INT4_ASYM,
            dataset=dataset,
            subset_size=2,
            **params,  # Unpacks to awq=True, gptq=True, etc.
        )
    except Exception as e:
        pytest.fail(f"Algorithm {list(params.keys())[0]} failed for transpose_b=False. Error: {e}")
