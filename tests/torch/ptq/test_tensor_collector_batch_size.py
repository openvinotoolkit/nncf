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

from nncf.experimental.common.tensor_statistics.collectors import AGGREGATORS_MAP
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.tensor_statistics.collectors import PT_REDUCERS_MAP
from nncf.torch.tensor_statistics.collectors import PTNNCFCollectorTensorProcessor
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from tests.common.experimental.test_tensor_collector_batch_size import TemplateTestTensorCollectorBatchSize


class TestTensorCollectorBatchSize(TemplateTestTensorCollectorBatchSize):
    @staticmethod
    def get_tensor_statistics_class():
        return PTMinMaxTensorStatistic

    @staticmethod
    def get_tensor_processor():
        return PTNNCFCollectorTensorProcessor()

    @staticmethod
    def get_nncf_tensor_class():
        return PTNNCFTensor

    @pytest.fixture(params=PT_REDUCERS_MAP.values())
    def reducers(self, request) -> bool:
        return request.param

    @pytest.fixture(params=AGGREGATORS_MAP.values())
    def aggregators(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[False])
    def inplace(self, request):
        return request.param

    def create_dataitems_without_batch_dim(self, input_shape: List[int], length: int = 100) -> List[torch.TensorType]:
        torch.random.manual_seed(seed=0)

        data_items = []
        for _ in range(length):
            data_items.append(torch.rand(input_shape))
        return data_items

    def add_batch_dim_to_dataitems(self, data_items: List[torch.TensorType], batch_size: int) -> List[torch.TensorType]:
        assert batch_size >= 1
        dataset = []
        item = []
        cnt = 0
        for data_item in data_items:
            if batch_size == 1:
                dataset.append(torch.unsqueeze(data_item, 0))
            else:
                item.append(data_item)
                if cnt == batch_size - 1:
                    dataset.append(torch.stack(item))
                    item = []
                    cnt = -1
                cnt += 1

        return dataset
