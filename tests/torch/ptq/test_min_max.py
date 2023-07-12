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

import pytest
import torch

from tests.post_training.test_templates.test_min_max import TemplateTestMinMaxAlgorithm
from tests.torch.ptq.helpers import get_nncf_network


class TestTorchMinMaxAlgorithm(TemplateTestMinMaxAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> torch.Tensor:
        return torch.Tensor(data)

    @staticmethod
    def backend_specific_model(model: bool, tmp_dir: str):
        return get_nncf_network(model, model.INPUT_SIZE)

    @staticmethod
    def fn_to_type(tensor):
        return torch.Tensor(tensor)

    @staticmethod
    def get_transform_fn():
        def transform_fn(data_item):
            tensor, _ = data_item
            return tensor

        return transform_fn


class TestTorchMinMaxCudaAlgorithm(TemplateTestMinMaxAlgorithm):
    @staticmethod
    def list_to_backend_type(data: List) -> torch.Tensor:
        return torch.Tensor(data)

    @staticmethod
    def backend_specific_model(model: bool, tmp_dir: str):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        return get_nncf_network(model.cuda(), model.INPUT_SIZE)

    @staticmethod
    def fn_to_type(tensor):
        return torch.Tensor(tensor).cuda()

    @staticmethod
    def get_transform_fn():
        def transform_fn(data_item):
            tensor, _ = data_item
            return tensor

        return transform_fn
