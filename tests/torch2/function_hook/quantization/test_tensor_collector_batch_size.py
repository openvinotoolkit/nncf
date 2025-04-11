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
import pytest
import torch

from nncf.experimental.common.tensor_statistics.collectors import AGGREGATORS_MAP
from nncf.experimental.common.tensor_statistics.collectors import REDUCERS_MAP
from tests.common.experimental.test_tensor_collector_batch_size import TemplateTestTensorCollectorBatchSize


class TestTensorCollectorBatchSize(TemplateTestTensorCollectorBatchSize):
    @pytest.fixture(params=REDUCERS_MAP.values())
    def reducers(self, request) -> bool:
        return request.param

    @pytest.fixture(params=AGGREGATORS_MAP.values())
    def aggregators(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[False])
    def inplace(self, request):
        return request.param

    @staticmethod
    def to_backend_tensor(tensor: np.ndarray):
        return torch.tensor(tensor)
