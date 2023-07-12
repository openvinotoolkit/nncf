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

from typing import List

import numpy as np
import onnx
import torch

from tests.post_training.test_templates.test_min_max import TemplateTestMinMaxAlgorithm


class TestTorchMinMaxAlgorithm(TemplateTestMinMaxAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> np.ndarray:
        return np.array(data)

    @staticmethod
    def backend_specific_model(model: bool, tmp_dir: str):
        onnx_path = f"{tmp_dir}/model.onnx"
        torch.onnx.export(model, torch.rand(model.INPUT_SIZE), onnx_path, opset_version=13, input_names=["input.1"])
        onnx_model = onnx.load(onnx_path)
        return onnx_model

    @staticmethod
    def fn_to_type(tensor):
        return np.array(tensor)

    @staticmethod
    def get_transform_fn():
        def transform_fn(data_item):
            tensor, _ = data_item
            return {"input.1": tensor}

        return transform_fn
