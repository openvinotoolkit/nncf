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

from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

from nncf.common.graph.layer_attributes import Dtype
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
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
            "min": MinReducer,
            "max": MaxReducer,
            "abs_max": AbsMaxReducer,
            "mean": MeanReducer,
            "quantile": QuantileReducer,
            "abs_quantile": AbsQuantileReducer,
            "batch_mean": BatchMeanReducer,
            "mean_per_ch": MeanPerChReducer,
        }

    def all_close(self, val, ref) -> bool:
        val_ = np.array(val)
        ref_ = np.array(ref)
        return np.allclose(val_, ref_) and val_.shape == ref_.shape

    def squeeze_tensor(self, ref_tensor: List[Any], axes: Optional[Tuple[int]] = None):
        return np.squeeze(np.array(ref_tensor), axes)

    def cast_tensor(self, tensor, dtype: Dtype):
        return tensor
