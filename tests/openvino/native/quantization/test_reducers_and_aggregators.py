# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from nncf.common.graph.layer_attributes import Dtype
from nncf.openvino.statistics.collectors import OVAbsMaxReducer
from nncf.openvino.statistics.collectors import OVAbsQuantileReducer
from nncf.openvino.statistics.collectors import OVBatchMeanReducer
from nncf.openvino.statistics.collectors import OVMaxReducer
from nncf.openvino.statistics.collectors import OVMeanPerChanelReducer
from nncf.openvino.statistics.collectors import OVMeanReducer
from nncf.openvino.statistics.collectors import OVMinReducer
from nncf.openvino.statistics.collectors import OVQuantileReducer
from nncf.tensor import Tensor
from tests.common.experimental.test_reducers_and_aggregators import TemplateTestReducersAggregators


class TestReducersAggregators(TemplateTestReducersAggregators):

    def get_nncf_tensor(self, x: np.array, dtype: Optional[Dtype] = None):
        if dtype is Dtype.INTEGER:
            x = x.astype(np.int64)
        if dtype is Dtype.FLOAT:
            x = x.astype(np.float32)
        return Tensor(x)

    @pytest.fixture(scope="module")
    def reducers(self):
        return {
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

    def squeeze_tensor(self, ref_tensor: List[Any], axes: Optional[Tuple[int]] = None):
        return np.squeeze(np.array(ref_tensor), axes)

    def cast_tensor(self, tensor, dtype: Dtype):
        return tensor
