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
import onnxruntime as rt
import pytest

import nncf
from tests.onnx.models import LinearModel as ModelWithSingleInput
from tests.onnx.models import MultiInputOutputModel as ModelWithMultipleInputs

dataset = [
    {
        "X": np.zeros((1, 3, 32, 32), dtype=np.float32),
        "X_1": np.zeros((1, 6, 3, 3), dtype=np.float32),
        "X_2": np.zeros((2, 6, 3, 3), dtype=np.float32),
        "X_3": np.zeros((3, 6, 3, 3), dtype=np.float32),
    }
]


def single_input_transform_fn(data_item):
    return {"X": data_item["X"]}


def multiple_inputs_transform_fn(data_item):
    return {
        "X_1": data_item["X_1"],
        "X_2": data_item["X_2"],
        "X_3": data_item["X_3"],
    }


@pytest.mark.parametrize(
    "model,transform_fn",
    [[ModelWithSingleInput(), single_input_transform_fn], [ModelWithMultipleInputs(), multiple_inputs_transform_fn]],
    ids=["single_input", "multiple_inputs"],
)
def test_transform_fn(model, transform_fn):
    # Check the transformation function
    session = rt.InferenceSession(model.onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
    session.run([], transform_fn(next(iter(dataset))))

    # Start quantization
    calibration_dataset = nncf.Dataset(dataset, transform_fn)
    _ = nncf.quantize(model.onnx_model, calibration_dataset)
