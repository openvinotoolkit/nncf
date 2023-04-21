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
from typing import List

import torch
from torch import Tensor

from nncf.data import Dataset
from nncf.quantization.algorithms.fast_bias_correction.torch_backend import PTFastBiasCorrectionAlgoBackend
from nncf.torch.utils import manual_seed
from tests.post_training.test_fast_bias_correction import TemplateTestFBCAlgorithm
from tests.torch.helpers import RandomDatasetMock
from tests.torch.ptq.helpers import ConvBNTestModel
from tests.torch.ptq.helpers import ConvTestModel
from tests.torch.ptq.helpers import FCTestModel
from tests.torch.ptq.helpers import get_min_max_and_fbc_algo_for_test
from tests.torch.ptq.helpers import get_nncf_network


def get_random_dataset(model: torch.nn.Module):
    manual_seed(42)

    def transform_fn(data_item):
        images, _ = data_item
        return images

    dataset = Dataset(RandomDatasetMock(model.INPUT_SIZE), transform_fn)
    return dataset


class TestTorchFBCAlgorithm(TemplateTestFBCAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> torch.Tensor:
        return torch.Tensor(data)

    @staticmethod
    def get_backend() -> PTFastBiasCorrectionAlgoBackend:
        return PTFastBiasCorrectionAlgoBackend

    @staticmethod
    def get_model(with_bias: bool, tmp_dir: str):
        return get_nncf_network(ConvTestModel(bias=with_bias), [1, 1, 4, 4])

    @staticmethod
    def get_dataset(model: torch.nn.Module):
        return get_random_dataset(model)

    @staticmethod
    def check_bias(model: torch.nn.Module, with_bias: bool):
        if with_bias:
            assert all(torch.isclose(model.conv.bias.data, Tensor([-1.9895, -1.9895]), rtol=0.0001))
        else:
            assert model.conv.bias is None


def test_fast_bias_correction_conv_bn_algo():
    """
    Check FBC for model with conv+bn layer.
    """
    model = get_nncf_network(ConvBNTestModel(), [1, 1, 4, 4])

    dataset = get_random_dataset(model)

    quantization_algorithm = get_min_max_and_fbc_algo_for_test()
    quantized_model = quantization_algorithm.apply(model, dataset=dataset)

    assert all(torch.isclose(quantized_model.bn.bias.data, Tensor([0.1105, 0.1105]), atol=0.0001))


def test_fast_bias_correction_fc_algo():
    """
    Check FBC for model with conv+bn layer.
    """
    model = get_nncf_network(FCTestModel(), [1, 1, 4, 4])

    dataset = get_random_dataset(model)

    quantization_algorithm = get_min_max_and_fbc_algo_for_test()
    quantized_model = quantization_algorithm.apply(model, dataset=dataset)

    assert all(torch.isclose(quantized_model.fc.bias.data, Tensor([0.1158, 0.1158]), atol=0.0001))
