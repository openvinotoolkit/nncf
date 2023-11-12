# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import pytest
import torch
from torch import nn

import nncf
from tests.torch.test_models.alexnet import AlexNet as ModelWithSingleInput


class ModelWithMultipleInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self._conv2d_0 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self._conv2d_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)

    def forward(self, input_0, input_1):
        output_0 = self._conv2d_0(input_0)
        output_1 = self._conv2d_1(input_1)
        return output_0 + output_1


dataset = [
    [
        torch.zeros((3, 32, 32), dtype=torch.float32),
        torch.ones((3, 32, 32), dtype=torch.float32),
    ]
]

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)


def single_input_transform_fn(data_item, device):
    return data_item[0].to(device)


def test_transform_fn_single_input(use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("There are no available CUDA devices")

    device = torch.device("cuda" if use_cuda else "cpu")
    model = ModelWithSingleInput().to(device)
    input_data = single_input_transform_fn(next(iter(dataloader)), device)

    # Check the transformation function
    model(input_data)
    # Start quantization
    calibration_dataset = nncf.Dataset(dataloader, partial(single_input_transform_fn, device=device))
    nncf.quantize(model, calibration_dataset)


def multiple_inputs_transform_tuple_fn(data_item, device):
    return data_item[0].to(device), data_item[1].to(device)


def multiple_inputs_transform_dict_fn(data_item, device):
    return {"input_0": data_item[0].to(device), "input_1": data_item[1].to(device)}


@pytest.mark.parametrize(
    "transform_fn", (multiple_inputs_transform_tuple_fn, multiple_inputs_transform_dict_fn), ids=["tuple", "dict"]
)
def test_transform_fn_multiple_inputs(transform_fn, use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("There are no available CUDA devices")

    device = torch.device("cuda" if use_cuda else "cpu")
    model = ModelWithMultipleInputs().to(device)
    input_data = transform_fn(next(iter(dataloader)), device)

    # Check the transformation function
    if isinstance(input_data, tuple):
        model(*input_data)
    if isinstance(input_data, dict):
        model(**input_data)

    # Start quantization
    calibration_dataset = nncf.Dataset(dataloader, partial(transform_fn, device=device))
    nncf.quantize(model, calibration_dataset)
