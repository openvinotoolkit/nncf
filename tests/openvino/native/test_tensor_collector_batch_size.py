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

from nncf.experimental.common.tensor_statistics.collectors import AGGREGATORS_MAP
from nncf.experimental.common.tensor_statistics.collectors import HistogramAggregator
from nncf.experimental.common.tensor_statistics.collectors import RawReducer
from nncf.openvino.statistics.collectors import OV_REDUCERS_MAP
from tests.common.experimental.test_tensor_collector_batch_size import TemplateTestTensorCollectorBatchSize


class TestTensorCollectorBatchSize(TemplateTestTensorCollectorBatchSize):
    @pytest.fixture(params=[r for r in OV_REDUCERS_MAP.values() if r is not RawReducer])
    def reducers(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[a for a in AGGREGATORS_MAP.values() if a is not HistogramAggregator])
    def aggregators(self, request) -> bool:
        return request.param

    @pytest.fixture(params=[True, False])
    def inplace(self, request):
        return request.param

    @staticmethod
    def to_backend_tensor(tensor: np.ndarray):
        return tensor
