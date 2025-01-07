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
import pytest
from openvino.runtime import Model
from openvino.runtime import Shape
from openvino.runtime import Type
from openvino.runtime import op
from openvino.runtime import opset13 as opset

import nncf
from nncf import Dataset
from tests.cross_fw.shared.datasets import MockDataset

INPUT_SHAPE = [2, 1, 1, 1]


def get_mock_model() -> Model:
    param_node = op.Parameter(Type.f32, Shape(INPUT_SHAPE))
    softmax_axis = 1
    softmax_node = opset.softmax(param_node, softmax_axis)
    return Model(softmax_node, [param_node], "mock")


def test_non_positive_subset_size():
    model_to_test = get_mock_model()

    with pytest.raises(nncf.ValidationError) as e:
        nncf.quantize(model_to_test, Dataset(MockDataset(INPUT_SHAPE)), subset_size=0)
        assert "Subset size must be positive." in e.info
