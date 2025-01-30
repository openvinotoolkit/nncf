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


def single_input_transform_fn(data_item, use_cuda):
    if use_cuda:
        return data_item[0].cuda()
    return data_item[0]


def test_transform_fn_single_input(use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("There are no available CUDA devices")

    model = ModelWithSingleInput()
    input_data = single_input_transform_fn(next(iter(dataloader)), use_cuda)
    if use_cuda:
        model = model.cuda()

    # Check the transformation function
    model(input_data)
    # Start quantization
    calibration_dataset = nncf.Dataset(dataloader, partial(single_input_transform_fn, use_cuda=use_cuda))
    nncf.quantize(model, calibration_dataset)


def multiple_inputs_transform_tuple_fn(data_item, use_cuda):
    if use_cuda:
        return data_item[0].cuda(), data_item[1].cuda()
    return data_item[0], data_item[1]


def multiple_inputs_transform_dict_fn(data_item, use_cuda):
    if use_cuda:
        return {"input_0": data_item[0].cuda(), "input_1": data_item[1].cuda()}
    return {"input_0": data_item[0], "input_1": data_item[1]}


@pytest.mark.parametrize(
    "transform_fn", (multiple_inputs_transform_tuple_fn, multiple_inputs_transform_dict_fn), ids=["tuple", "dict"]
)
def test_transform_fn_multiple_inputs(transform_fn, use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("There are no available CUDA devices")

    model = ModelWithMultipleInputs()
    input_data = transform_fn(next(iter(dataloader)), use_cuda)
    if use_cuda:
        model = model.cuda()

    # Check the transformation function
    if isinstance(input_data, tuple):
        model(*input_data)
    if isinstance(input_data, dict):
        model(**input_data)

    # Start quantization
    calibration_dataset = nncf.Dataset(dataloader, partial(transform_fn, use_cuda=use_cuda))
    nncf.quantize(model, calibration_dataset)
