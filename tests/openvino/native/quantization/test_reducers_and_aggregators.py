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
import pytest

from nncf.openvino.statistics.collectors import OVAbsMaxReducer
from nncf.openvino.statistics.collectors import OVAbsQuantileReducer
from nncf.openvino.statistics.collectors import OVBatchMeanReducer
from nncf.openvino.statistics.collectors import OVMaxReducer
from nncf.openvino.statistics.collectors import OVMeanPerChanelReducer
from nncf.openvino.statistics.collectors import OVMeanReducer
from nncf.openvino.statistics.collectors import OVMinReducer
from nncf.openvino.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.openvino.statistics.collectors import OVNoopReducer
from nncf.openvino.statistics.collectors import OVQuantileReducer
from nncf.openvino.tensor import OVNNCFTensor
from tests.experimental.common.test_reducers_and_aggregators import TemplateTestReducersAggreagtors


class TestReducersAggregators(TemplateTestReducersAggreagtors):
    @pytest.fixture
    def tensor_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_nncf_tensor(self, x: np.array):
        return OVNNCFTensor(x)

    @pytest.fixture(scope="module")
    def reducers(self):
        return {
            "noop": OVNoopReducer,
            "min": OVMinReducer,
            "max": OVMaxReducer,
            "abs_max": OVAbsMaxReducer,
            "mean": OVMeanReducer,
            "quantile": OVQuantileReducer,
            "abs_quantile": OVAbsQuantileReducer,
            "batch_mean": OVBatchMeanReducer,
            "mean_per_ch": OVMeanPerChanelReducer,
        }

    def all_close(self, val, ref) -> bool:
        val_ = np.array(val)
        ref_ = np.array(ref)
        return np.allclose(val_, ref_) and val_.shape == ref_.shape
