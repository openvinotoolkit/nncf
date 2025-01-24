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
import openvino.runtime as ov
import pytest

import nncf
from tests.openvino.native.models import ConvModel as ModelWithMultipleInputs
from tests.openvino.native.models import LinearModel as ModelWithSingleInput

dataset = [
    {
        "input_0": np.zeros((1, 3, 4, 2), dtype=np.float32),
        "input_1": np.zeros((1, 3, 2, 4), dtype=np.float32),
    }
]


def single_input_transform_fn(data_item):
    return data_item["input_0"]


def multiple_inputs_transform_fn(data_item):
    return data_item["input_0"], data_item["input_1"]


def multiple_inputs_as_dict_transform_fn(data_item):
    return {
        "Input_1": data_item["input_0"],
        "Input_2": data_item["input_1"],
    }


@pytest.mark.parametrize(
    "model,transform_fn",
    [
        [ModelWithSingleInput(), single_input_transform_fn],
        [ModelWithMultipleInputs(), multiple_inputs_transform_fn],
        [ModelWithMultipleInputs(), multiple_inputs_as_dict_transform_fn],
    ],
    ids=[
        "single_input_native",
        "multiple_inputs_native",
        "multiple_inputs_as_dict_native",
    ],
)
def test_transform_fn(model, transform_fn):
    # Check the transformation function
    compiled_model = ov.compile_model(model.ov_model, device_name="CPU")
    _ = compiled_model(transform_fn(next(iter(dataset))))

    # Start quantization
    calibration_dataset = nncf.Dataset(dataset, transform_fn)
    _ = nncf.quantize(model.ov_model, calibration_dataset)
