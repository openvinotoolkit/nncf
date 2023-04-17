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
import numpy as np
import onnx
import pytest
import torch

from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from tests.onnx.quantization.common import get_random_dataset_for_test
from tests.torch.ptq.helpers import ConvTestModel
from tests.torch.ptq.helpers import get_min_max_and_fbc_algo_for_test


def get_data_from_node(model: onnx.ModelProto, node_name: str):
    data = [t for t in model.graph.initializer if t.name == node_name]
    if data:
        return onnx.numpy_helper.to_array(data[0])
    return None


@pytest.mark.parametrize("with_bias, ref_bias", ((False, None), (True, np.array([-2.0181384, -2.0181384]))))
def test_fast_bias_correction_algo(with_bias, ref_bias, tmpdir):
    """
    Check working on fast bias correction algorithm and compare bias in quantized model with reference
    """
    model = ConvTestModel(bias=with_bias)
    input_shape = [1, 1, 4, 4]

    onnx_path = f"{tmpdir}/model.onnx"
    torch.onnx.export(model, torch.rand(input_shape), onnx_path, opset_version=13)
    onnx_model = onnx.load(onnx_path)

    quantization_algorithm = get_min_max_and_fbc_algo_for_test()

    np.random.seed(42)
    dataset = get_random_dataset_for_test(onnx_model, False)
    quantized_model = quantization_algorithm.apply(onnx_model, dataset=dataset)

    if ref_bias is None:
        assert get_data_from_node(quantized_model, "conv.bias") is None
    else:
        assert np.all(
            np.isclose(get_data_from_node(quantized_model, "conv.bias"), ref_bias)
        )



@pytest.mark.parametrize(
    "bias_value, bias_shift, channel_axis, ref_shape",
    (
        (np.array([1, 1]), np.array([0.1, 0.1]), 1, [2]),
        (np.array([[1, 1]]), np.array([0.1, 0.1]), -1, [1, 2]),
        (np.array([[1, 1]]), np.array([0.1, 0.1]), 1, [1, 2]),
    ),
)
def test_reshape_bias_shift(bias_value, bias_shift, channel_axis, ref_shape):
    """
    Checks the result of the FastBiasCorrection.reshape_bias_shift method if np.array is used.
    """
    new_bias_shift = FastBiasCorrection.reshape_bias_shift(bias_shift, bias_value, channel_axis)
    assert list(new_bias_shift.shape) == ref_shape
