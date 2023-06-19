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

from typing import Callable, Dict, Tuple

import numpy as np
import openvino.runtime as ov
import pytest

from tests.openvino.native.models import LinearModel
from tests.post_training.test_templates.test_smooth_quantize import TemplateTestSQAlgorithm


class TestOVFBCAlgorithm(TemplateTestSQAlgorithm):
    @staticmethod
    def fn_to_type(tensor) -> np.ndarray:
        return np.array(tensor)

    @staticmethod
    def get_transform_fn() -> Callable:
        def transform_fn(data_item):
            tensor, _ = data_item
            return {"Input": tensor}

        return transform_fn

    @staticmethod
    def check_scales(model: ov.Model, reference_values: Dict[str, np.ndarray]) -> None:
        ops_list = {op.get_friendly_name(): op for op in model.get_ops()}
        for ref_name, ref_value in reference_values.items():
            node = ops_list[ref_name]
            const_node = node.input(1).get_source_output().get_node()

            assert const_node.get_type_name() == "Constant"

            value = const_node.data
            assert np.all(np.isclose(value, ref_value, atol=0.0001)), f"{value} != {ref_value}"

    @staticmethod
    def get_dataset_shape(model: ov.Model) -> Tuple[int]:
        return tuple(model.input(0).shape)

    @pytest.mark.parametrize(
        "model, reference_values",
        (
            (
                LinearModel().ov_model,
                {"Reshape/smooth_quant_multiply": np.array([0.984319, 1.032351, 1.1578148, 1.0598988])},
            ),
        ),
    )
    def test_smooth_quant_algo(self, model, reference_values):
        return super().test_smooth_quant_algo(model, reference_values)
