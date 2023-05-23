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

import numpy as np
import openvino.runtime as ov
import pytest

import nncf
from nncf.openvino.quantization.backend_parameters import BackendParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
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
    "model,transform_fn,use_pot",
    [
        [ModelWithSingleInput(), single_input_transform_fn, False],
        [ModelWithSingleInput(), single_input_transform_fn, True],
        [ModelWithMultipleInputs(), multiple_inputs_transform_fn, False],
        [ModelWithMultipleInputs(), multiple_inputs_transform_fn, True],
        [ModelWithMultipleInputs(), multiple_inputs_as_dict_transform_fn, False],
        [ModelWithMultipleInputs(), multiple_inputs_as_dict_transform_fn, True],
    ],
    ids=[
        "single_input_native",
        "signle_input_pot",
        "multiple_inputs_native",
        "multiple_inputs_pot",
        "multiple_inputs_as_dict_native",
        "multiple_inputs_as_dict_pot",
    ],
)
def test_transform_fn(model, transform_fn, use_pot: bool):
    # Check the transformation function
    compiled_model = ov.compile_model(model.ov_model)
    _ = compiled_model(transform_fn(next(iter(dataset))))

    # Start quantization
    params = AdvancedQuantizationParameters(
        backend_params={
            BackendParameters.USE_POT: use_pot,
        }
    )
    calibration_dataset = nncf.Dataset(dataset, transform_fn)
    _ = nncf.quantize(model.ov_model, calibration_dataset, advanced_parameters=params)
