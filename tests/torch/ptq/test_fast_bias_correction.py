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
import torch
from torch import Tensor

from nncf.data import Dataset
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.torch.nncf_network import ExtraCompressionModuleType
from tests.torch.helpers import RandomDatasetMock
from tests.torch.ptq.helpers import ConvTestModel
from tests.torch.ptq.helpers import get_min_max_and_fbc_algo_for_test
from tests.torch.ptq.helpers import get_nncf_network


@pytest.mark.parametrize("with_bias", (False, True))
def test_fast_bias_correction_algo(with_bias):
    model = ConvTestModel(bias=with_bias)
    input_shape = [1, 1, 4, 4]
    nncf_network = get_nncf_network(model, input_shape)
    nncf_network.register_compression_module_type(ExtraCompressionModuleType.EXTERNAL_QUANTIZER)
    quantization_algorithm = get_min_max_and_fbc_algo_for_test()

    def transform_fn(data_item):
        images, _ = data_item
        return images

    dataset = Dataset(RandomDatasetMock(input_shape), transform_fn)
    quantized_model = quantization_algorithm.apply(nncf_network, dataset=dataset)

    if with_bias:
        assert not torch.equal(model.conv.bias.data, quantized_model.conv.bias.data)
    else:
        assert quantized_model.conv.bias is None


@pytest.mark.parametrize(
    "bias_value, bias_shift, channel_axis, ref_shape",
    (
        (Tensor([1, 1]), Tensor([0.1, 0.1]), 1, [2]),
        (Tensor([[1, 1]]), Tensor([0.1, 0.1]), -1, [1, 2]),
        (Tensor([[1, 1]]), Tensor([0.1, 0.1]), 1, [1, 2]),
    ),
)
def test_reshape_bias_shift(bias_value, bias_shift, channel_axis, ref_shape):
    new_bias_shit = FastBiasCorrection.reshape_bias_shift(bias_value, bias_shift, channel_axis)
    assert list(new_bias_shit.shape) == ref_shape
